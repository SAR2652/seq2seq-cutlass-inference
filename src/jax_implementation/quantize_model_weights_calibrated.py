import os
import json
import argparse
from typing import Union

import numpy as np
import pandas as pd
import jax
from jax import random
import jax.numpy as jnp
import orbax.checkpoint as ocp
from torch.utils.data import DataLoader
from src.common_utils import load_tokenizer, collate_fn
from src.jax_implementation.utils import init_train_state
from src.jax_implementation.model import CrossAttentionModelFLAX
from src.dataset import PolynomialDataset


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir',
                        help='Directory containing checkpoints',
                        type=str, default='output/normal/checkpoints')
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
    # Calibration arguments
    parser.add_argument('--calib_data_path',
                        type=str,
                        help='Path to CSV file used for calibration '
                             '(e.g. validation.csv)',
                        default='./output/validation.csv')
    parser.add_argument('--num_calib_samples',
                        type=int,
                        help='Number of samples to use for calibration',
                        default=512)
    parser.add_argument('--calib_batch_size',
                        type=int,
                        help='Batch size for calibration forward passes',
                        default=64)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def quantize_tensor_int8(tensor: jnp.ndarray, num_bits=8):
    """Symmetric int8 quantization (zero_point always 0).
    x_float = x_int8 * scale, so accumulations can be dequantized with a
    single scale_x * scale_W multiply."""
    qmax = 2 ** (num_bits - 1) - 1   # 127

    abs_max = jnp.max(jnp.abs(tensor))
    scale = jnp.where(abs_max != 0, abs_max / qmax, 1.0)

    quantized = jnp.clip(
        jnp.round(tensor / scale), -qmax, qmax
    ).astype(jnp.int8)

    return {'quantized': quantized, 'scale': float(scale), 'zero_point': 0}


def quantize_bias_int32(bias: jnp.ndarray, accum_scale: float):
    """Quantize bias to int32 at a pre-determined accumulation scale.

    accum_scale = scale_x * scale_W, matching the scale of the int32
    accumulation (x@Wx + h@Wh) so all three terms can be dequantized with
    a single multiply. zero_point is always 0 (symmetric)."""
    b_float = bias.astype(jnp.float32)
    b_int32 = jnp.clip(
        jnp.round(b_float / accum_scale),
        -(2 ** 31), 2 ** 31 - 1
    ).astype(jnp.int32)
    return {'quantized': b_int32, 'scale': accum_scale, 'zero_point': 0}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def build_calib_dataloader(calib_data_path, tokenizer, num_calib_samples,
                           calib_batch_size):
    """Load calibration CSV and return a DataLoader over the first
    num_calib_samples rows."""
    df = pd.read_csv(calib_data_path)
    if num_calib_samples > 0:
        df = df.iloc[:num_calib_samples, :]
    factors = df['factor'].tolist()
    expansions = df['expansion'].tolist()
    dataset = PolynomialDataset(factors, tokenizer, expansions)
    return DataLoader(dataset, shuffle=False, batch_size=calib_batch_size,
                      collate_fn=collate_fn)


def calibrate_input_scale(params, model, calib_loader, num_bits=8):
    """Run calibration batches through the encoder embedding layer to find
    the abs-max of the bfloat16 embedding activations, then derive scale_x.

    scale_x = abs_max / qmax  (symmetric int8 convention)

    Args:
        params:       Restored Flax model params dict.
        model:        CrossAttentionModelFLAX instance.
        calib_loader: DataLoader yielding (inputs, targets, _, _) batches of
                      token-id sequences [B, seq_len].
        num_bits:     Quantization bit-width (default 8).

    Returns:
        scale_x (float): Dequantization scale for the int8 embedding output.
    """
    qmax = 2 ** (num_bits - 1) - 1   # 127

    # JIT-compile a single embedding forward pass.
    # Flax setup() fields are only accessible inside apply/init, so we use
    # model.apply with a method lambda instead of model.encoder.embedding.

    @jax.jit
    def embed_fn(inputs):
        return model.apply(
            {'params': params},
            inputs,
            method=lambda m, x: m.encoder.embedding(x)
        )

    abs_max = 0.0
    num_batches = 0

    for inputs, _, _, _ in calib_loader:
        inputs = jnp.array(inputs)              # [B, seq_len]  int32
        embeddings = embed_fn(inputs)           # [B, seq_len, embed_dim] bf16
        batch_abs_max = float(jnp.max(jnp.abs(embeddings.astype(jnp.float32))))
        abs_max = max(abs_max, batch_abs_max)
        num_batches += 1

    print(f"Calibration: {num_batches} batches processed, "
          f"embedding abs_max = {abs_max:.6f}")

    scale_x = abs_max / qmax
    print(f"Derived scale_x = {scale_x:.8f}")
    return scale_x


def calibrate_hidden_scale(params, model, calib_loader, num_bits=8):
    """Run calibration batches through the full encoder to find the abs-max of
    the LSTM hidden state h across all timesteps and batches, then derive
    h_scale.

    h_scale is used at inference time to transiently quantize the float32
    hidden state h to int8 immediately before each h @ Wh GEMM.  It must
    be calibrated separately from scale_x because the two GEMMs (x @ Wx and
    h @ Wh) accumulate on different numeric ranges and require independent
    dequantization scales.

    The encoder is expected to return the sequence of LSTM hidden states with
    shape [B, seq_len, hidden_dim].  If the model is bidirectional the output
    may be a concatenation of forward and backward states; the abs-max over
    the full tensor is a safe upper bound for both directions.

    Args:
        params:       Restored Flax model params dict.
        model:        CrossAttentionModelFLAX instance.
        calib_loader: DataLoader yielding (inputs, targets, _, _) batches.
        num_bits:     Quantization bit-width (default 8).

    Returns:
        h_scale (float): Dequantization scale for the transiently-quantized
                         int8 hidden state.
    """
    qmax = 2 ** (num_bits - 1) - 1  # 127

    @jax.jit
    def encoder_fn(inputs):
        # encoder(x) runs embedding + LSTM and returns the hidden-state
        # sequence [B, seq_len, hidden_dim] (or concatenation for
        # bidirectional).
        return model.apply(
            {'params': params},
            inputs,
            method=lambda m, x: m.encoder(x)
        )

    abs_max = 0.0
    num_batches = 0

    for inputs, _, _, _ in calib_loader:
        inputs = jnp.array(inputs)
        # encoder returns (outputs, hidden, cell); outputs [B, seq_len,
        # hidden_dim] is the stacked fwd_hidden (and bkwd_hidden concatenated
        # if bidirectional) at every timestep — exactly the values fed back as
        # h in the C++ forward.
        outputs, _, _ = encoder_fn(inputs)
        batch_abs_max = float(jnp.max(jnp.abs(outputs.astype(jnp.float32))))
        abs_max = max(abs_max, batch_abs_max)
        num_batches += 1

    print(f"Calibration: {num_batches} batches processed, "
          f"hidden state abs_max = {abs_max:.6f}")

    h_scale = abs_max / qmax
    print(f"Derived h_scale = {h_scale:.8f}")
    return h_scale


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


# ---------------------------------------------------------------------------
# Recursive quantization (two-pass per level: kernels first, then biases)
# ---------------------------------------------------------------------------

def recursively_quantize(params: Union[dict], scale_x: float,
                         h_scale: float, parent_key: str = '') -> dict:
    """Recursively quantize all parameters.

      - embeddings → BF16 (unchanged)
      - kernels    → INT8 symmetric  (zero_point=0, scale derived from tensor)
      - biases     → INT32 symmetric (zero_point=0, accum_scale chosen below)

    Bias accum_scale selection:
      LSTM hidden-kernel biases (under hf/hg/hi/ho) are fused with the
      h_int8 @ Wh GEMM at inference time, so their accum_scale must be
      h_scale * sibling_kernel_scale.  All other biases use scale_x *
      sibling_kernel_scale.

    Two-pass per dict level ensures kernel scales are available when
    quantizing sibling biases."""

    # LSTM hidden-state projection blocks (hf/hg/hi/ho).  Biases at these
    # levels are fused with h @ Wh and must use h_scale, not scale_x.
    LSTM_HIDDEN_KEYS = {'hf', 'hg', 'hi', 'ho'}

    quantized_params = {}
    kernel_scales = {}   # key → float scale, populated in pass 1

    # ------------------------------------------------------------------
    # Pass 1: recurse into sub-dicts and quantize kernels/embeddings
    # ------------------------------------------------------------------
    for k, v in params.items():
        full_key = f"{parent_key}/{k}" if parent_key else k
        lower_key = full_key.lower()

        if isinstance(v, dict):
            quantized_params[k] = recursively_quantize(
                v, scale_x, h_scale, parent_key=full_key)

        elif isinstance(v, jnp.ndarray) and v.dtype == jnp.float32:
            if "embedding" in lower_key:
                quantized_params[k] = v.astype(jnp.bfloat16)

            elif "kernel" in lower_key:
                # Transpose to column-major before quantizing
                v_transposed = jnp.transpose(v, (1, 0))
                result = quantize_tensor_int8(v_transposed)
                quantized_params[k] = result
                kernel_scales[k] = result['scale']

            # biases handled in pass 2

        else:
            quantized_params[k] = v

    # ------------------------------------------------------------------
    # Pass 2: quantize biases using the correct accumulation scale.
    #
    # The immediate parent key (last path component) identifies whether we are
    # inside an LSTM hidden-projection block (hf/hg/hi/ho).  Biases there are
    # fused with h_int8 @ Wh at inference, so they must be pre-scaled with
    # h_scale * kernel_scale, not scale_x * kernel_scale.
    # ------------------------------------------------------------------
    immediate_parent = parent_key.split('/')[-1] if parent_key else ''
    in_lstm_hidden_block = immediate_parent in LSTM_HIDDEN_KEYS

    for k, v in params.items():
        if not (isinstance(v, jnp.ndarray) and v.dtype == jnp.float32):
            continue
        full_key = f"{parent_key}/{k}" if parent_key else k
        if "bias" not in full_key.lower():
            continue

        # Find the corresponding kernel scale at the same dict level.
        # Convention: key named "bias" pairs with key named "kernel".
        sibling_kernel_key = k.replace('bias', 'kernel')
        w_scale = kernel_scales.get(sibling_kernel_key, None)

        if w_scale is None:
            # No sibling kernel found — fall back to independent int32 quant
            # and warn so the user knows the scale may not match.
            print(f"Warning: no sibling kernel found for bias '{full_key}'. "
                  f"Falling back to independent int32 quantization.")
            quantized_params[k] = quantize_tensor_int32(v)
        else:
            input_scale = h_scale if in_lstm_hidden_block else scale_x
            accum_scale = input_scale * w_scale
            quantized_params[k] = quantize_bias_int32(v, accum_scale)

    return quantized_params


# ---------------------------------------------------------------------------
# Binary + JSON export (unchanged from original)
# ---------------------------------------------------------------------------

def export_quantized_params_to_bin_json(quantized_params, output_dir,
                                        scale_x: float, h_scale: float):
    os.makedirs(output_dir, exist_ok=True)

    bin_data = bytearray()
    metadata = {}
    offset = 0

    def process_node(node, meta_subtree):
        nonlocal bin_data, offset

        for k, v in node.items():
            if isinstance(v, dict) and 'quantized' in v:
                quant_np = np.array(v['quantized']).flatten()
                np_dtype = quant_np.dtype

                if np_dtype == np.int8:
                    dtype = 'int8'
                elif np_dtype == np.int32:
                    dtype = 'int32'
                else:
                    dtype = str(np_dtype)

                quantized_np = quant_np.astype(np_dtype)
                size = quantized_np.size
                bin_data += quantized_np.tobytes()

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

            elif isinstance(v, jnp.ndarray):
                if v.dtype == jnp.bfloat16:
                    raw_np = np.array(v).flatten().astype(np.float16)
                    dtype = "bfloat16"
                elif v.dtype == jnp.float32:
                    raw_np = np.array(v).flatten().astype(np.float32)
                    dtype = "float32"
                else:
                    raw_np = np.array(v).flatten()
                    dtype = str(raw_np.dtype)

                size = raw_np.size
                bin_data += raw_np.tobytes()
                scale = float(np.max(raw_np) - np.min(raw_np)) \
                    if size > 0 else 0.0

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

    # Store calibration scales so the inference server can load them.
    metadata['calibration'] = {'scale_x': scale_x, 'h_scale': h_scale}

    # Inject h_scale into every LSTM cell metadata block.
    # The inference-server LSTMCell constructor reads lstm_metadata["h_scale"]
    # to know the fixed scale for transiently quantizing h → int8 before each
    # h @ Wh GEMM.  We write the same value into both forward and backward
    # blocks (if present) since both pass h through tanh and share the same
    # theoretical output range.
    encoder_meta = metadata.get('encoder', {})
    for lstm_key in ('forward_lstm', 'backward_lstm'):
        if lstm_key in encoder_meta:
            encoder_meta[lstm_key]['h_scale'] = h_scale

    with open(os.path.join(output_dir, 'weights.bin'), 'wb') as f:
        f.write(bin_data)

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved weights and metadata to {output_dir}/weights.bin "
          f"and metadata.json")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def quantize_weights_to_int8(args):
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    output_dir = args.output_dir
    tokenizer = load_tokenizer(args.tokenizer_filepath)

    model = CrossAttentionModelFLAX(
        args.embed_dim, args.hidden_dim, tokenizer.vocab_size, args.num_heads,
        tokenizer.sos_token_id, args.bidirectional
    )

    prng_key = random.PRNGKey(args.random_state)
    train_state = init_train_state(model, prng_key,
                                   seq_len=tokenizer.MAX_SEQUENCE_LENGTH)

    abstract_state = jax.tree_util.tree_map(
        ocp.utils.to_shape_dtype_struct, train_state
    )
    checkpoint_manager = ocp.CheckpointManager(ckpt_dir)
    step = checkpoint_manager.latest_step()
    state = checkpoint_manager.restore(
        step, args=ocp.args.StandardRestore(abstract_state)
    )
    params = state.params

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    calib_loader = build_calib_dataloader(
        args.calib_data_path, tokenizer,
        args.num_calib_samples, args.calib_batch_size
    )
    # scale_x: dequant scale for int8 embedding output (x @ Wx GEMM input)
    scale_x = calibrate_input_scale(params, model, calib_loader)
    # h_scale: dequant scale for transiently-quantized hidden state
    # (h @ Wh GEMM input)
    h_scale = calibrate_hidden_scale(params, model, calib_loader)

    # ------------------------------------------------------------------
    # Quantize
    # ------------------------------------------------------------------
    quantized_params = recursively_quantize(params, scale_x=scale_x,
                                            h_scale=h_scale)

    export_quantized_params_to_bin_json(quantized_params, output_dir,
                                        scale_x=scale_x, h_scale=h_scale)


def main():
    args = get_arguments()
    quantize_weights_to_int8(args)


if __name__ == "__main__":
    main()
