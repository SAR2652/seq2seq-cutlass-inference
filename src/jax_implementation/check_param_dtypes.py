import os
import argparse

import jax
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir',
                        help='Model checkpoint filepath',
                        type=str, default='./output/checkpoints')
    return parser.parse_args()


def print_param_dtypes(params):
    for path, value in jax.tree_util.tree_flatten_with_path(params)[0]:
        key = "/".join(str(p) for p in path)
        print(f"{key}: dtype={value.dtype}, shape={value.shape}")


def display_params(args):

    ckpt_dir = os.path.abspath(args.ckpt_dir)
    orbax_checkpointer = PyTreeCheckpointer()
    checkpoint_manager = CheckpointManager(ckpt_dir, orbax_checkpointer)
    step = checkpoint_manager.latest_step()
    checkpoint = checkpoint_manager.restore(step)
    state = checkpoint['state']
    params = state['params']
    print_param_dtypes(params)


def main():
    args = get_arguments()
    display_params(args)


if __name__ == '__main__':
    main()
