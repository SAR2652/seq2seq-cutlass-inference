import os
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common_utils import load_file


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_txt_file',
                        help='Text file containing polynomials and expansions',
                        type=str, default='./train.txt')
    parser.add_argument('--output_dir',
                        help='Dirctory to store output',
                        type=str, default='./output')
    parser.add_argument('--random_state',
                        help='Random state for initialization',
                        type=int, default=42)
    parser.add_argument('--vt_size',
                        help='Size of validation partition',
                        type=float, default=0.2)
    return parser.parse_args()


def split_data(args):

    data_txt_file = args.data_txt_file
    output_dir = args.output_dir
    random_state = args.random_state
    vt_size = args.vt_size

    os.makedirs(output_dir, exist_ok=True)

    factors, expansions = load_file(data_txt_file)

    X_train, X_vt, y_train, y_vt = train_test_split(
        factors, expansions, test_size=vt_size, random_state=random_state
    )

    df_train = pd.DataFrame()
    df_train['factor'] = np.asarray(X_train).T
    df_train['expansion'] = np.asarray(y_train).T

    training_filepath = os.path.join(output_dir, 'training.csv')
    df_train.to_csv(training_filepath, index=False)

    X_val, X_test, y_val, y_test = train_test_split(
        X_vt, y_vt, test_size=0.5, random_state=random_state
    )

    df_val = pd.DataFrame()
    df_val['factor'] = np.asarray(X_val).T
    df_val['expansion'] = np.asarray(y_val).T

    validation_filepath = os.path.join(output_dir, 'validation.csv')
    df_val.to_csv(validation_filepath, index=False)

    df_test = pd.DataFrame()
    df_test['factor'] = np.asarray(X_test).T
    df_test['expansion'] = np.asarray(y_test).T

    test_filepath = os.path.join(output_dir, 'test.csv')
    df_test.to_csv(test_filepath, index=False)


def main():
    args = get_arguments()
    split_data(args)


if __name__ == '__main__':
    main()
