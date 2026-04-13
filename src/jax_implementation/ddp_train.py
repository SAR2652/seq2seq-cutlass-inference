import os
import argparse
import time
from typing import Tuple

import jax
from jax import random
import jax.numpy as jnp
import optax
import wandb

from flax.training import train_state
from flax.jax_utils import replicate, unreplicate
import orbax.checkpoint as ocp

import pandas as pd
from torch.utils.data import DataLoader

from src.dataset import PolynomialDataset
from src.jax_implementation.model import CrossAttentionModelFLAX
from src.common_utils import load_tokenizer, collate_fn, WandbCSVLogger
from src.jax_implementation.utils import eval_step, train_epoch_or_evaluate, \
    is_replicated, init_train_state

# compute_equivalence_accuracy, score


def get_training_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',
                        type=str, help='Directory containing Input Files',
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
                        default=1)
    parser.add_argument('--tokenizer_filepath',
                        type=str,
                        help='Path to tokenizer which is to be used',
                        default='./output/tokenizer.joblib')
    parser.add_argument('--bidirectional',
                        action='store_true',
                        help='Use bidirectional model')
    parser.add_argument('--continue_from_ckpt',
                        action='store_true',
                        help='Continue training from a checkpoint')
    parser.add_argument('--ckpt_dir',
                        help='Directory containing checkpoints',
                        type=str, default='checkpoints')
    parser.add_argument('--use_cache',
                        help='Use KV caching during model training/inference',
                        action='store_true')
    parser.add_argument('--ddp',
                        help='Activate Distributed Data Parallel',
                        action='store_true')
    parser.add_argument('--teacher_force_ratio',
                        help='Teacher force ratio',
                        type=float, default=0.5)
    parser.add_argument('--warmup_steps',
                        help='Number of warm up steps before training',
                        type=int, default=10)
    parser.add_argument('--warmup_epochs',
                        help='Number of warm up epochs for teacher forcing',
                        type=int, default=25)
    parser.add_argument('--profile',
                        help='Profile model training using wandb',
                        action='store_true')
    parser.add_argument('--disable_wandb',
                        help='Disable wandb logging',
                        action='store_true')
    return parser.parse_args()


@jax.jit
def apply_gradient_update(state, grads):
    return state.apply_gradients(grads=grads)


def create_train_step_fn(ddp: bool = False):
    """
    Returns a unified training step function.

    Args:
        ddp (bool): If True, performs DDP-style training using `jax.pmap`.

    Returns:
        train_step: (state, inputs, targets) -> (state, loss, grads)
    """

    def train_step(state: train_state.TrainState, inputs: jnp.ndarray,
                   targets: jnp.ndarray, curr_epoch: int, warmup_epochs: int
                   ) -> Tuple[train_state.TrainState, jnp.ndarray, dict]:

        def loss_fn(params):
            logits = state.apply_fn({'params': params}, inputs, targets,
                                    curr_epoch=curr_epoch,
                                    warmup_epochs=warmup_epochs)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits,
                                                                   targets)
            return loss.mean(), logits

        (loss, logits), grads = jax.value_and_grad(loss_fn,
                                                   has_aux=True)(state.params)

        if ddp:
            loss = jax.lax.pmean(loss, axis_name='num_devices')
            grads = jax.lax.pmean(grads, axis_name='num_devices')

        # Note: We return grads separately, letting the user decide when to
        # apply
        return state, loss, grads

    # Compile for performance
    return jax.pmap(train_step, axis_name='num_devices',
                    static_broadcasted_argnums=(3, 4)) if ddp else \
        jax.jit(train_step, static_argnums=(3, 4))


def load_data_and_return_dataloader(filepath, tokenizer, batch_size,
                                    return_dataset: bool = False,
                                    num_samples: int = 0):
    df = pd.read_csv(filepath)
    if num_samples > 0:
        df = df.iloc[:num_samples, :]
    factors = df['factor'].tolist()
    expansions = df['expansion'].tolist()
    dataset = PolynomialDataset(factors, tokenizer, expansions)
    dataloader = DataLoader(dataset, shuffle=True,
                            batch_size=batch_size, collate_fn=collate_fn)

    if return_dataset:
        return dataloader, dataset

    return dataloader


def train_model(args):

    # input/output directories
    input_dir = args.input_dir
    output_dir = args.output_dir
    ckpt_dir = os.path.join(output_dir, args.ckpt_dir)
    logs_dir = os.path.join(output_dir, 'logs')
    for directory in [output_dir, ckpt_dir, logs_dir]:
        os.makedirs(directory, exist_ok=True)
    ckpt_dir = os.path.abspath(ckpt_dir)
    log_file = os.path.join(logs_dir, 'metrics_log.csv')
    tokenizer_filepath = args.tokenizer_filepath
    tokenizer = load_tokenizer(tokenizer_filepath)

    # hyperparameters
    random_state = args.random_state
    embed_size = args.embed_dim
    hidden_size = args.hidden_dim
    num_heads = args.num_heads
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    bidirectional = args.bidirectional
    teacher_force_ratio = args.teacher_force_ratio
    warmup_steps = args.warmup_steps
    warmup_epochs = args.warmup_epochs
    profile = args.profile
    disable_wandb = args.disable_wandb
    use_wandb = True if not disable_wandb else False

    # performance optimizations
    use_cache = args.use_cache
    ddp = args.ddp
    num_devices = jax.local_device_count()
    print(f'Number of Devices = {num_devices}')

    model = CrossAttentionModelFLAX(
        embed_size, hidden_size, tokenizer.vocab_size, num_heads,
        tokenizer.sos_token_id, bidirectional, use_cache, teacher_force_ratio
    )

    prng_key = random.PRNGKey(random_state)
    # param_shapes = jax.tree_map(lambda x: x.shape, params)
    # print(f"Model parameter shapes: {param_shapes}")

    state = init_train_state(model, prng_key, batch_size,
                             tokenizer.MAX_SEQUENCE_LENGTH, learning_rate)

    if ddp:     # replicate model state on all available GPUs
        state = replicate(state)

    # Initialize model checkpointing requirements
    options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = ocp.CheckpointManager(
        ocp.test_utils.erase_and_create_empty(ckpt_dir),
        options=options
    )

    name = 'best_model_saca'
    if bidirectional:
        name += '_bidirect'
    name += '_'

    # initialize model training/evaluation and update functions
    train_step = create_train_step_fn(ddp)
    update_model = jax.pmap(apply_gradient_update, axis_name='num_devices') \
        if ddp else apply_gradient_update
    optimized_eval_step = jax.pmap(eval_step, axis_name='num_devices',
                                   static_broadcasted_argnums=(0,)) \
        if ddp else eval_step

    train_path = os.path.join(input_dir, 'training.csv')
    val_path = os.path.join(input_dir, 'validation.csv')

    train_dataloader, train_dataset = load_data_and_return_dataloader(
        train_path, tokenizer, batch_size, return_dataset=True
    )

    # create a fresh iterator
    train_iter = iter(train_dataloader)

    # model warmup
    for _ in range(warmup_steps):
        inputs, targets, _, _ = next(train_iter)
        if ddp:
            inputs = inputs.reshape(num_devices, -1,
                                    tokenizer.MAX_SEQUENCE_LENGTH)
            targets = targets.reshape(num_devices, -1,
                                      tokenizer.MAX_SEQUENCE_LENGTH)
        _, _, _ = train_step(state, inputs, targets, 0, warmup_epochs)

    # recreate dataloader for training data
    train_dataloader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=batch_size, collate_fn=collate_fn)

    # load validation data
    val_dataloader = load_data_and_return_dataloader(val_path, tokenizer,
                                                     batch_size)

    best_val_acc = float('-inf')
    global_step = 0
    start = time.perf_counter()

    if profile:
        if use_wandb:
            wandb.init(
                project="polynomial-expansion",
                name="encoder-decoder-training",
                config={
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "kv_caching": use_cache,
                    "DDP": ddp
                }
            )

        logger = WandbCSVLogger(log_file, use_wandb)
        logger.start()
        global_step = 0
    else:
        logger = None

    for epoch in range(epochs):

        if profile:
            epoch_start = time.perf_counter()

        state, running_loss, global_step = train_epoch_or_evaluate(
            state, train_dataloader, tokenizer, ddp, train_step,
            update_model, num_devices, "train", epoch, warmup_epochs,
            profile, logger, global_step
        )

        if profile:
            val_start = time.perf_counter()

        model_params = state.params

        val_preds, _, val_gt = train_epoch_or_evaluate(
            (model, model_params), val_dataloader, tokenizer, ddp,
            optimized_eval_step, None, num_devices, "eval",
            profile=profile, logger=logger
        )

        # val_expansions = tokenizer.batch_decode_expressions(val_preds)
        # val_acc = compute_equivalence_accuracy(val_expansions, val_gt)
        val_acc = (val_preds.flatten() == val_gt.flatten()).sum() * 100 / \
            val_gt.size

        print(f"Epoch {epoch + 1}: Training Loss = {running_loss:.4f}, "
              f"Validation Accuracy = {val_acc:.2f}%")

        # save model state on only one GPU
        if val_acc > best_val_acc and \
                (not ddp or (ddp and jax.process_index() == 0)):
            if ddp:     # get a single copy of model training state
                save_state = unreplicate(state)
            else:
                save_state = state
            checkpoint_manager.save(
                epoch + 1,
                args=ocp.args.StandardSave(save_state)
            )
            best_val_acc = val_acc

        if profile:
            epoch_end = time.perf_counter()
            epoch_time_total = epoch_end - epoch_start
            epoch_train_only_time = val_start - epoch_start
            epoch_val_only_time = epoch_end - val_start
            logger.log({
                "epoch_time_total": epoch_time_total,
                "epoch_train_only_time": epoch_train_only_time,
                "epoch_val_only_time": epoch_val_only_time,
                "epoch": epoch + 1
            }, step=global_step)

    if profile:
        logger.finish()

        if use_wandb:
            wandb.finish()

    # checkpoint manager saves model training checkpoints asynchronously
    checkpoint_manager.wait_until_finished()

    end = time.perf_counter()

    elapsed = end - start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60

    print(f"Elapsed time: {hours:02d}:{minutes:02d}:{seconds:06.3f}")

    # load best performing checkpoint
    step = checkpoint_manager.latest_step()
    state = checkpoint_manager.restore(step)
    params = state['params']
    print(f'Model Parameters:\n'
          f'{jax.tree_map(lambda x: x.shape, params)}')

    if ddp and not is_replicated(params):
        model_params = replicate(params)
        print(f'Model Parameters after DDP:\n'
              f'{jax.tree_map(lambda x: x.shape, model_params)}')
    else:
        model_params = params

    # load and evaluate test set
    test_path = os.path.join(input_dir, 'test.csv')
    test_dataloader = load_data_and_return_dataloader(test_path, tokenizer,
                                                      batch_size)

    test_preds, _, test_gt = train_epoch_or_evaluate(
        (model, model_params), test_dataloader, tokenizer, ddp,
        optimized_eval_step, None, num_devices, "eval",
    )

    # test_expansions = tokenizer.batch_decode_expressions(test_preds)
    # test_acc = compute_equivalence_accuracy(test_expansions, test_gt)
    test_acc = (test_preds.flatten() == test_gt.flatten()).sum() * 100 / \
        test_gt.size

    print(f"Test Accuracy = {test_acc:.2f}%")


def main():
    args = get_training_arguments()
    train_model(args)


if __name__ == '__main__':
    main()
