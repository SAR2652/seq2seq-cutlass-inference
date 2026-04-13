import pickle
from typing import Tuple, Iterable, Dict, Any

import csv
import wandb
import joblib
import numpy as np
from sympy import simplify, sympify



def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


class Tokenizer:
    MAX_POLYNOMIAL_LENGTH = 29
    MAX_SEQUENCE_LENGTH = MAX_POLYNOMIAL_LENGTH + 1     # Append EOS Token
    pad_token = '<pad>'
    pad_token_id = 0
    sos_token = '<s>'
    sos_token_id = 1
    eos_token = '</s>'
    eos_token_id = 2
    special_token_ids = [pad_token_id, sos_token_id, eos_token_id]

    def __init__(self):
        self.vocab_dict = dict()
        self.vocab_dict[self.sos_token] = self.sos_token_id
        self.vocab_dict[self.eos_token] = self.eos_token_id
        self.vocab_dict[self.pad_token] = self.pad_token_id
        self.current_token_idx = 3
        self.vocab_size = len(self.vocab_dict)
        self.id_dict = dict((v, k) for k, v in self.vocab_dict.items())

    def expand_vocabulary(self, expressions: Iterable):
        """Create Vocabulary, i.e. mapping for each unique token and its
        corresponding index"""
        for expression in expressions:
            tokens = list(expression)
            for token in tokens:
                if token not in self.vocab_dict.keys():
                    self.vocab_dict[token] = self.current_token_idx
                    self.id_dict[self.current_token_idx] = token
                    self.current_token_idx += 1
        self.vocab_size = len(self.vocab_dict)

    def convert_tokens_to_ids(self, expression):
        """Convert Tokens into their corresponding Integer mappings"""
        tokens = list(expression)
        # print(tokens)
        input_ids = [self.vocab_dict[token] for token in tokens]
        return input_ids

    def encode(self, factor, expansion=None):
        """Tokenize a single factor and its corresponding expansion and
        append EOS token"""
        factor_input_ids = self.convert_tokens_to_ids(factor)
        factor_input_ids.append(self.eos_token_id)
        factor_padding_length = self.MAX_SEQUENCE_LENGTH - \
            len(factor_input_ids)
        factor_input_ids.extend([self.pad_token_id] * factor_padding_length)

        if expansion is not None:
            expansion_label_ids = self.convert_tokens_to_ids(expansion)
            expansion_label_ids.append(self.eos_token_id)
            expansion_padding_length = self.MAX_SEQUENCE_LENGTH - \
                len(expansion_label_ids)

            expansion_label_ids.extend([self.pad_token_id] *
                                       expansion_padding_length)

            return factor_input_ids, expansion_label_ids

        return factor_input_ids

    def encode_expression(self, expression):
        """Encode a single expression into its corresponding numeric ids"""
        input_ids = self.convert_tokens_to_ids(expression)
        input_ids.append(self.eos_token)
        padding_length = self.MAX_SEQUENCE_LENGTH - len(input_ids)
        input_ids.extend([self.pad_token_id] * padding_length)
        return input_ids

    def decode_expression(self, expression):
        """Convert IDs to their corresponding tokens"""
        output_sequence = list()
        for id in expression:
            if id == self.eos_token_id:
                break
            elif id in self.special_token_ids:
                continue
            output_sequence.append(self.id_dict[id])

        return output_sequence

    def batch_decode_expressions(self, expressions):
        """Convert IDs to their corresponding tokens"""

        batch_decoded_expressions_map = map(self.decode_expression,
                                            expressions)

        batch_concat_operation_map = map(''.join,
                                         batch_decoded_expressions_map)

        return list(batch_concat_operation_map)

    def validate(self):
        for k, v in self.vocab_dict.items():
            if self.id_dict[v] != k:
                return False
        return True


def load_tokenizer(tokenizer_filepath: str):
    if tokenizer_filepath.endswith('.pickle'):
        with open(tokenizer_filepath, 'rb') as tokenizer_binary:
            tokenizer = pickle.load(tokenizer_binary)
    elif tokenizer_filepath.endswith('.joblib'):
        tokenizer = joblib.load(tokenizer_filepath)
    return tokenizer


def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)


def collate_fn(batch):
    # Batch shape: (seq_len, batch_size) -> (batch_size, seq_len)
    input_ids = [item["input_ids"] for item in batch]
    factors = [item["factor"] for item in batch]
    target_ids = [item["target_ids"] for item in batch]
    expansions = [item["expansion"] for item in batch]

    # print(target_ids)
    target_flag = False
    if not any(x is None for x in target_ids):
        target_flag = True

    # print(input_ids)
    # print(target_ids)

    return np.stack(input_ids).transpose(0, 1), \
        np.stack(target_ids).transpose(0, 1) if target_flag else target_ids, \
        factors, expansions


def is_equivalent(expr1, expr2):
    try:
        return simplify(sympify(expr1) - sympify(expr2)) == 0
    except Exception:
        return False


def compute_equivalence_accuracy(preds, targets):
    """
    Compute percentage of symbolic equivalence between predicted and target
    expressions.

    Args:
        preds (list of str): Predicted expressions
        targets (list of str): Ground truth expressions

    Returns:
        float: Percentage of correct predictions
    """
    assert len(preds) == len(targets), "Prediction and target lists must have"
    "same length"
    correct = sum(is_equivalent(p, t) for p, t in zip(preds, targets))
    return 100.0 * correct / len(preds)


class WandbCSVLogger:
    def __init__(self, csv_path: str = "metrics_log.csv",
                 use_wandb: bool = False):
        self.csv_path = csv_path
        self.use_wandb = use_wandb
        self.csv_file = None
        self.csv_writer = None
        self.header_written = False

        if self.use_wandb:
            self.wandb = wandb

    def start(self):
        # Initialize the CSV file
        self.csv_file = open(self.csv_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

    def log(self, metrics: Dict[str, Any], step: int = None):
        # Log to wandb (optional)
        if self.use_wandb:
            self.wandb.log(metrics, step=step)

        # Log to CSV
        if not self.header_written:
            self.csv_writer.writerow(["step"] + list(metrics.keys()))
            self.header_written = True
        self.csv_writer.writerow([step] + list(metrics.values()))
        self.csv_file.flush()

    def finish(self):
        if self.csv_file:
            self.csv_file.close()
