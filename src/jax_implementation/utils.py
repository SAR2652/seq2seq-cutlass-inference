import time
import functools
from typing import Union, Tuple, Literal

import optax
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
from flax.jax_utils import replicate
from torch.utils.data import DataLoader


@functools.partial(jax.jit, static_argnums=0)
def eval_step(model, params, inputs):
    logits = model.apply({'params': params}, inputs, targets=None, eval=True)
    probs = jax.nn.softmax(logits, axis=-1)
    preds = jnp.argmax(probs, axis=-1)
    return preds, probs


def is_replicated(params):
    """Checks if the parameters are replicated across devices."""
    leaves, _ = jax.tree_util.tree_flatten(params)
    for leaf in leaves:
        if not hasattr(leaf, "shape"):
            continue
        if leaf.shape[0] == jax.local_device_count():
            return True
    return False


def init_train_state(model, random_key, batch_size: int = 1, seq_len: int = 30,
                     learning_rate: float = 0.001) -> train_state.TrainState:

    dummy_inputs = jnp.ones((batch_size, seq_len),
                            dtype=jnp.int32)
    dummy_targets = jnp.ones((batch_size, seq_len),
                             dtype=jnp.int32)

    # Initialize the Model
    variables = model.init(random_key, dummy_inputs, dummy_targets)
    # Create the optimizer
    optimizer = optax.adam(learning_rate)
    # Create a State
    return train_state.TrainState.create(
        apply_fn=model.apply,
        tx=optimizer,
        params=variables['params']
    )


def train_epoch_or_evaluate(
        state_or_model: Union[train_state.TrainState, Tuple],
        dataloader: DataLoader, tokenizer, ddp: bool,
        step_function, update_model=None, num_devices: int = 1,
        mode: Literal["train", "eval", "infer"] = "train",
        curr_epoch: int = None, warmup_epochs: int = None,
        profile: bool = False, logger=None, global_step: int = 0):

    if isinstance(state_or_model, tuple):
        model, params = state_or_model
        replicate_flag = False
        if ddp and not is_replicated(params):
            replicated_params = replicate(params)
            replicate_flag = True
    else:
        state = state_or_model

    if mode == "train":
        running_loss = 0
        assert update_model is not None, "update_model() must have a " \
            "function as value in 'train' mode"

    if mode in ["eval", "infer"]:
        predictions_list, probabilities_list = [], []

        if mode == "eval":
            ground_truth_list = list()

    if profile:
        step_start = time.perf_counter()
        log_interval = len(dataloader) // 100
        token_count = 0

    for step, batch in enumerate(dataloader, 0):

        if mode == "train":
            global_step += 1

        inputs, targets, _, _ = batch

        if mode != "infer":
            assert all(x is not None for x in targets), \
                "Targets can be None ONLY in inference mode!"

        inputs_jnp = jnp.array(inputs, dtype=jnp.int32)
        targets_jnp = jnp.array(targets, dtype=jnp.int32)

        if ddp:

            inputs_jnp = inputs_jnp.reshape(
                num_devices, -1, tokenizer.MAX_SEQUENCE_LENGTH
            )

            if mode == "train":
                targets_jnp = targets_jnp.reshape(
                    num_devices, -1, tokenizer.MAX_SEQUENCE_LENGTH
                )

        if mode == "train":
            state, loss, grads = step_function(state, inputs_jnp, targets_jnp,
                                               curr_epoch, warmup_epochs)

            if profile:
                loss = loss.block_until_ready()

            running_loss += loss.mean().item()

            if (step + 1) % log_interval == 0:
                print(f'Running Loss after {step + 1} batches = '
                      f'{running_loss:.4f}')

            state = update_model(state, grads)

        else:

            if ddp and replicate_flag:
                batch_preds, batch_probs = step_function(
                    model, replicated_params, inputs
                )
            else:
                batch_preds, batch_probs = step_function(model, params,
                                                         inputs_jnp)

            if profile:
                batch_preds.block_until_ready()
                batch_probs.block_until_ready()

            # print(f'Processed {i + 1} batches for evaluation')

            # Handle DDP output shapes: (num_devices, batch_size_per_device,
            # ...)
            if ddp:
                batch_preds = batch_preds.reshape(
                    -1, batch_preds.shape[-1]
                )
                batch_probs = batch_probs.reshape(
                    -1, batch_probs.shape[-2], batch_probs.shape[-1]
                )

            predictions_list.append(batch_preds)
            probabilities_list.append(batch_probs)

            if mode == "eval":
                ground_truth_list.append(targets)

        if profile:
            input_tokens = np.sum(inputs != tokenizer.pad_token_id)

            if mode != "infer":
                target_tokens = np.sum(targets != tokenizer.pad_token_id)

            token_count += (input_tokens + target_tokens)

            if step % log_interval == 0 and step > 0:
                elapsed = time.perf_counter() - step_start
                steps_per_sec = log_interval / elapsed
                tokens_per_sec = token_count / elapsed

                metric_dict = {
                    "steps/sec": steps_per_sec,
                    "tokens/sec": tokens_per_sec,
                }

                if mode == "train":
                    metric_dict["train/loss"] = loss.mean().item()
                    metric_dict["epoch"] = curr_epoch

                if mode == "train":
                    save_step = global_step
                else:
                    save_step = None

                logger.log(metric_dict, save_step)

                # Reset interval counters
                step_start = time.perf_counter()
                token_count = 0

    if mode == "train":
        return state, running_loss, global_step
    else:
        predictions_jnp = jnp.concatenate(predictions_list, axis=0)
        probabilities_jnp = jnp.concatenate(probabilities_list, axis=0)

        if profile:
            predictions_jnp.block_until_ready()
            probabilities_jnp.block_until_ready()

        predictions = np.asarray(jax.device_get(predictions_jnp))
        probabilities = np.asarray(jax.device_get(probabilities_jnp))

        return_vals = [predictions, probabilities]

        if mode == "eval":

            ground_truth_jnp = jnp.concatenate(ground_truth_list, axis=0)
            # print(ground_truth_jnp.shape)

            if profile:
                ground_truth_jnp.block_until_ready()

            ground_truth = np.asarray(jax.device_get(ground_truth_jnp))
            # print(ground_truth.shape)

            return_vals.append(ground_truth)

        return tuple(return_vals)
