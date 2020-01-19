import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
import os
import argparse
from datasets import save_imgs
import fastai
from fastai.vision import *


def train(data_dir: str,
          bs,
          sz=128,
          nfolds=4,
          fold=0,
          stats=(
              [0.0692],
              [0.2051],
          )):
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    label_cols = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
    data = ImageList.from_df(
        train_df,
        path='',
        folder=data_dir,
        suffix='.png',
        cols='image_id',
        convert_mode='L') \
        .split_by_idx(range(fold * len(train_df) // nfolds, (fold + 1) * len(train_df) // nfolds)) \
        .label_from_df(cols=label_cols) \
        .transform(get_transforms(do_flip=False, max_warp=0.1), size=sz, padding_mode='zeros') \
        .databunch(bs=bs)
    data = data.normalize(stats)
    data.show_batch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="Data")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    if (not any(map(lambda f: f.endswith(".png"), os.listdir(args.data_dir)))):
        save_imgs(args.data_dir)
    train(args.data_dir, bs=args.batch_size)
