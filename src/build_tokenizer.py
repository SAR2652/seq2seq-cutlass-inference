import os
import argparse

import joblib
from src.common_utils import load_file, Tokenizer


def get_vocabulary_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filepath',
                        type=str,
                        help='Path to Input File',
                        default='./output/train.txt')
    parser.add_argument('--output_dir',
                        type=str,
                        help='Directory to save tokenizer file',
                        default='./output')
    return parser.parse_args()


def build_tokenizer(args):
    input_file = args.input_filepath
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    factors, expressions = load_file(input_file)

    tokenizer = Tokenizer()
    tokenizer.expand_vocabulary(factors)
    tokenizer.expand_vocabulary(expressions)

    tokenizer_filepath = os.path.join(output_dir, 'tokenizer.joblib')
    joblib.dump(tokenizer, tokenizer_filepath)

    print('Successfully built Tokenizer!')


def main():
    args = get_vocabulary_arguments()
    build_tokenizer(args)


if __name__ == '__main__':
    main()
