import os
import json
import argparse
import numpy as np
from typing import Union

import jax
from jax import random
import jax.numpy as jnp
import orbax.checkpoint as ocp
from src.common_utils import load_tokenizer
from src.jax_implementation.utils import init_train_state
from src.jax_implementation.model import CrossAttentionModelFLAX


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir',
                        help='Directory containing checkpoints',
                        type=str, default='output/ddp_kv/output/checkpoints')
    parser.add_argument('--output_dir',
                        help='Directory to store output',
                        type=str, default='./output')
    parser.add_argument('--tokenizer_filepath',
                        type=str,
                        help='Path to tokenizer which is to be used',
                        default='./output/tokenizer.joblib')
    parser.add_argument('--random_state',
                        help='Random state for weights initialization',
                        type=int, default=42)
    parser.add_argument('--embed_dim',
                        help='Size of embedding',
                        type=int, default=64)
    parser.add_argument('--hidden_dim',
                        type=int,
                        help='Number of Neurons in Hidden Layers',
                        default=64)
    parser.add_argument('--num_heads',
                        help='Number of Attention Heads',
                        type=int, default=4)
    parser.add_argument('--bidirectional',
                        action='store_true',
                        help='Use bidirectional model')
    return parser.parse_args()


def quantize_tensor_int8(tensor: jnp.ndarray, num_bits=8):
    """Quantizes a float32 JAX tensor to int8 using uniform affine
    quantization."""
    qmin = -2 ** (num_bits - 1)
    qmax = 2 ** (num_bits - 1) - 1

    min_val = jnp.min(tensor)
    max_val = jnp.max(tensor)

    scale = jnp.where(max_val != min_val, (max_val - min_val) / (qmax - qmin),
                      1.0)
    zero_point = jnp.where(
        max_val != min_val,
        jnp.round(qmin - min_val / scale).astype(jnp.int32),
        0
    )

    quantized = jnp.where(
        max_val != min_val,
        jnp.clip(jnp.round(tensor / scale + zero_point),
                 qmin, qmax).astype(jnp.int8),
        jnp.zeros_like(tensor, dtype=jnp.int8)
    )

    return {
        'quantized': quantized,
        'scale': scale,
        'zero_point': zero_point
    }


def quantize_tensor_int32(tensor: jnp.ndarray):
    """Quantizes a float32 JAX tensor to int32 using uniform affine
    quantization (per-tensor). This implementation avoids integer overflow
    by performing range arithmetic in floating point and only casting at the
    end.
    """
    # int32 range constants as Python ints
    qmin_int = -2 ** 31
    qmax_int = 2 ** 31 - 1

    # Convert to JAX-friendly floats for range arithmetic
    qmin = jnp.array(float(qmin_int), dtype=jnp.float32)
    qmax = jnp.array(float(qmax_int), dtype=jnp.float32)
    qrange = qmax - qmin  # float32 safe

    min_val = jnp.min(tensor).astype(jnp.float32)
    max_val = jnp.max(tensor).astype(jnp.float32)

    # Compute scale in float32 to avoid overflow/underflow inside JAX
    scale = jnp.where(max_val != min_val, (max_val - min_val) / qrange,
                      jnp.array(1.0, dtype=jnp.float32))

    # Avoid division by zero if scale is zero
    safe_scale = jnp.where(scale == 0.0, jnp.array(1.0, dtype=jnp.float32),
                           scale)

    # Compute zero_point in float then round and cast to int64 before final
    # cast
    zero_point_f = jnp.round(qmin - min_val / safe_scale)
    zero_point = zero_point_f.astype(jnp.int64)

    # Quantize in float then clip and cast to int32
    quant_f = jnp.round(tensor.astype(jnp.float32) / safe_scale + zero_point_f)
    quant_clipped = jnp.clip(quant_f, qmin, qmax).astype(jnp.int32)

    # If the tensor had no dynamic range, return zeros
    quantized = jnp.where(max_val != min_val, quant_clipped,
                          jnp.zeros_like(tensor, dtype=jnp.int32))

    return {
        'quantized': quantized,
        'scale': scale,
        'zero_point': zero_point
    }


def recursively_quantize(params: Union[dict], parent_key: str = '') -> dict:
    """
    Recursively processes all JAX float32 arrays:
      - embeddings → BF16 (no quantization, just cast)
      - weights   → INT8 (with scale/zero_point)
      - bias      → INT32 (quantized)
    """
    quantized_params = {}

    for k, v in params.items():
        full_key = f"{parent_key}/{k}" if parent_key else k

        if isinstance(v, dict):
            quantized_params[k] = recursively_quantize(v, full_key)

        elif isinstance(v, jnp.ndarray) and v.dtype == jnp.float32:
            lower_key = full_key.lower()

            if "embedding" in lower_key:
                # Embeddings: store as BF16
                quantized_params[k] = v.astype(jnp.bfloat16)

            elif "bias" in lower_key:
                # Bias: quantize to INT32 (UNCHANGED)
                quantized_params[k] = quantize_tensor_int32(v)

            elif "kernel" in lower_key:
                # Weights: transpose to store as column-major
                # JAX/NumPy are row-major by default.
                # Saving the transpose makes it effectively column-major
                v_transposed = jnp.transpose(v, (1, 0))
                quantized_params[k] = quantize_tensor_int8(v_transposed)

            else:
                # Other float32 tensors (if any)
                quantized_params[k] = quantize_tensor_int8(v)

        else:
            quantized_params[k] = v

    return quantized_params


def export_quantized_params_to_bin_json(quantized_params, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    bin_data = bytearray()
    metadata = {}
    offset = 0

    def process_node(node, meta_subtree):
        nonlocal bin_data, offset

        for k, v in node.items():
            # Quantized weights or biases (INT8 or INT32)
            if isinstance(v, dict) and 'quantized' in v:
                # Determine dtype from numpy dtype of quantized array
                quant_np = np.array(v['quantized']).flatten()
                np_dtype = quant_np.dtype

                if np_dtype == np.int8:
                    dtype = 'int8'
                elif np_dtype == np.int32:
                    dtype = 'int32'
                else:
                    # Fallback to string representation
                    dtype = str(np_dtype)

                quantized_np = quant_np.astype(np_dtype)
                size = quantized_np.size
                bin_data += quantized_np.tobytes()

                # Convert scale and zero_point to Python scalars if they are
                # arrays
                scale_val = float(v['scale']) if not \
                    isinstance(v['scale'], (jnp.ndarray, np.ndarray)) \
                    else float(np.array(v['scale']).item())
                zp_val = int(v['zero_point']) if not \
                    isinstance(v['zero_point'], (jnp.ndarray, np.ndarray)) \
                    else int(np.array(v['zero_point']).item())

                meta_subtree[k] = {
                    'shape': list(v['quantized'].shape),
                    'scale': scale_val,
                    'zero_point': zp_val,
                    'dtype': dtype,
                    'offset': offset,
                    'size': size
                }
                offset += quantized_np.nbytes

            # Raw arrays: embeddings (BF16) or others left as-is
            elif isinstance(v, jnp.ndarray):
                if v.dtype == jnp.bfloat16:
                    raw_np = np.array(v).flatten().astype(np.float16)
                    # store as float16 on disk
                    dtype = "bfloat16"
                elif v.dtype == jnp.float32:
                    raw_np = np.array(v).flatten().astype(np.float32)
                    dtype = "float32"
                else:
                    # Fallback: store in original dtype
                    raw_np = np.array(v).flatten()
                    dtype = str(raw_np.dtype)

                size = raw_np.size
                bin_data += raw_np.tobytes()
                scale = float(np.max(raw_np) - np.min(raw_np)) if size > 0 \
                    else 0.0

                meta_subtree[k] = {
                    'shape': list(v.shape),
                    'dtype': dtype,
                    'scale': scale,
                    'offset': offset,
                    'size': size
                }

                offset += raw_np.nbytes

            elif isinstance(v, dict):
                meta_subtree[k] = {}
                process_node(v, meta_subtree[k])

            else:
                meta_subtree[k] = str(v)

    process_node(quantized_params, metadata)

    with open(os.path.join(output_dir, 'weights.bin'), 'wb') as f:
        f.write(bin_data)

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    print("Saved hierarchical metadata and weights to "
          f"{output_dir}/weights.bin and metadata.json")


def quantize_weights_to_int8(args):
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    output_dir = args.output_dir
    tokenizer_filepath = args.tokenizer_filepath
    tokenizer = load_tokenizer(tokenizer_filepath)

    random_state = args.random_state
    embed_size = args.embed_dim
    hidden_size = args.hidden_dim
    num_heads = args.num_heads
    bidirectional = args.bidirectional

    checkpoint_manager = ocp.CheckpointManager(ckpt_dir)
    step = checkpoint_manager.latest_step()

    model = CrossAttentionModelFLAX(
        embed_size, hidden_size, tokenizer.vocab_size, num_heads,
        tokenizer.sos_token_id, bidirectional
    )

    prng_key = random.PRNGKey(random_state)

    train_state = init_train_state(
        model, prng_key, seq_len=tokenizer.MAX_SEQUENCE_LENGTH
    )

    abstract_state = jax.tree_util.tree_map(
        ocp.utils.to_shape_dtype_struct, train_state
    )
    state = checkpoint_manager.restore(
        step, args=ocp.args.StandardRestore(abstract_state)
    )
    params = state.params

    quantized_params = recursively_quantize(params)
    export_quantized_params_to_bin_json(quantized_params, output_dir)


def main():
    args = get_arguments()
    quantize_weights_to_int8(args)


if __name__ == "__main__":
    main()
