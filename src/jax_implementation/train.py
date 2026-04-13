import os

import optax
import argparse
import pandas as pd
import jax
from jax import random
import jax.numpy as jnp
from torch.utils.data import DataLoader
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager, \
    CheckpointManagerOptions
from flax.training import train_state, orbax_utils
from src.dataset import PolynomialDataset
from src.common_utils import load_tokenizer, collate_fn
from src.jax_implementation.utils import init_train_state
from src.jax_implementation.model import CrossAttentionModelFLAX


def get_training_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',
                        type=str, help='Directory containing training, '
                        'validation and test CSV files',
                        default='./output')
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
    parser.add_argument('--learning_rate',
                        type=float,
                        help='Learning Rate at which the model is to be '
                        'trained', default=1e-4)
    parser.add_argument('--output_dir',
                        type=str,
                        help='Directory to save output',
                        default='./output')
    parser.add_argument('--batch_size',
                        help='Batch size for model training',
                        type=int, default=768)
    parser.add_argument('--epochs',
                        type=int,
                        help='Number of Epochs to train the model',
                        default=1000)
    parser.add_argument('--tokenizer_filepath',
                        type=str,
                        help='Path to tokenizer which is to be used',
                        default='./output/tokenizer.joblib')
    parser.add_argument('--ckpt_dir',
                        help='Directory containing checkpoints',
                        type=str, default='checkpoints')
    parser.add_argument('--bidirectional',
                        help='Activate Bidirectionla LSTM in encoder',
                        action='store_true')
    parser.add_argument('--use_cache',
                        help='Activate Key-Value Caching',
                        action='store_true')
    return parser.parse_args()


@jax.jit
def train_step(state: train_state.TrainState, inputs: jnp.ndarray,
               targets: jnp.ndarray):

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, inputs)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits,
                                                               targets)
        loss = loss.mean()
        return loss, logits

    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = gradient_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, grads


def train_model(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    random_state = args.random_state
    embed_size = args.embed_dim
    hidden_size = args.hidden_dim
    num_heads = args.num_heads
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    tokenizer_filepath = args.tokenizer_filepath
    tokenizer = load_tokenizer(tokenizer_filepath)
    use_cache = args.use_cache
    bidirectional = args.bidirectional
    ckpt_dir = os.path.join(args.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_dir = os.path.abspath(ckpt_dir)

    train_path = os.path.join(input_dir, 'training.csv')
    val_path = os.path.join(input_dir, 'validation.csv')
    test_path = os.path.join(input_dir, 'test.csv')

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    # df = df.iloc[:12800, :]

    train_factors = train_df['factor'].tolist()
    train_expansions = train_df['expansion'].tolist()

    val_factors = val_df['factor'].tolist()
    val_expansions = val_df['expansion'].tolist()

    test_factors = test_df['factor'].tolist()
    test_expansions = test_df['expansion'].tolist()

    train_dataset = PolynomialDataset(train_factors, tokenizer,
                                      train_expansions)
    val_dataset = PolynomialDataset(val_factors, tokenizer,
                                    val_expansions)
    test_dataset = PolynomialDataset(test_factors, tokenizer,
                                     test_expansions)

    train_dataloader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=batch_size, collate_fn=collate_fn)

    val_dataloader = DataLoader(val_dataset, shuffle=True,
                                batch_size=batch_size, collate_fn=collate_fn)

    test_dataloader = DataLoader(test_dataset, shuffle=True,
                                 batch_size=batch_size, collate_fn=collate_fn)

    model = CrossAttentionModelFLAX(
        embed_size, hidden_size, tokenizer.vocab_size, num_heads,
        tokenizer.sos_token_id, bidirectional, use_cache
    )

    prng_key = random.PRNGKey(random_state)
    # dummy_inputs = jnp.ones((batch_size, tokenizer.MAX_SEQUENCE_LENGTH),
    #                         dtype=jnp.int32)
    # dummy_targets = jnp.ones((batch_size, tokenizer.MAX_SEQUENCE_LENGTH),
    #                          dtype=jnp.int32)
    # params = model.init(prng_key, dummy_inputs, dummy_targets)
    # jax.tree_map(lambda x: x.shape, params)     # Check the parameters

    state = init_train_state(model, prng_key, batch_size,
                             tokenizer.MAX_SEQUENCE_LENGTH, learning_rate)

    orbax_checkpointer = PyTreeCheckpointer()
    options = CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = CheckpointManager(ckpt_dir, orbax_checkpointer,
                                           options)

    name = 'best_model_saca'
    if bidirectional:
        name += '_bidirect'
    name += '_'

    min_loss = float('inf')
    for epoch in range(epochs):

        running_loss = 0
        for i, batch in enumerate(train_dataloader):
            inputs, targets, _, _ = batch
            inputs = jnp.array(inputs, dtype=jnp.int32)
            targets = jnp.array(targets, dtype=jnp.int32)
            state, loss, _ = train_step(state, inputs, targets)
            running_loss += loss
            if (i + 1) % (len(train_dataloader) // 100) == 0:
                print(f'Running Loss after {i + 1} batches = '
                      f'{running_loss:.4f}')

        avg_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        if avg_loss < min_loss:
            min_loss = avg_loss
            ckpt = {'state': state}
            save_args = orbax_utils.save_args_from_target(ckpt)
            checkpoint_manager.save(epoch + 1, ckpt,
                                    save_kwargs={
                                        'save_args': save_args
                                    })


def main():
    args = get_training_arguments()
    train_model(args)


if __name__ == '__main__':
    main()
