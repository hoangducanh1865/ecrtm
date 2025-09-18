import os
import numpy as np
import scipy.io
import yaml
import argparse
from src.trainer.trainer import Trainer

from src.utils.TextData import TextData
from src.utils import file_utils
from src.evaluator.evaluator import Evaluator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('-d', '--dataset')
    parser.add_argument('-m', '--model')
    parser.add_argument('-c', '--config')
    parser.add_argument('-k', '--num_topic', type=int, default=50)
    parser.add_argument('--num_top_word', type=int, default=15)
    parser.add_argument('--test_index', type=int, default=1)
    parser.add_argument('--path')
    parser.add_argument('--label_path')
    parser.add_argument('--data_path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.mode == 'train':
        trainer = Trainer(args)
        trainer.fit()

    else:
        evaluator = Evaluator(args)
        evaluator.evaluate()


if __name__ == '__main__':
    main()
