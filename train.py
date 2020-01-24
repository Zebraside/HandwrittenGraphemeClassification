import os
import argparse
from data import save_imgs, load_imgs
from fastai.data_block import ItemList
from models.fastai import Rnet_1ch
import torch.nn.functional as F
from torch import nn
from fastai.vision import Learner
from models.fastai import Metric_grapheme, Metric_consonant, Metric_vowel, \
    Metric_tot
from fastai.callbacks import CSVLogger
from utils import SaveModelCallback, Loss_combine


def train(data: ItemList, fold=0):
    model = Rnet_1ch()
    learn = Learner(data,
                    model,
                    loss_func=Loss_combine(),
                    metrics=[
                        Metric_grapheme(),
                        Metric_vowel(),
                        Metric_consonant(),
                        Metric_tot()
                    ])
    logger = CSVLogger(learn, f'log{fold}')
    learn.clip_grad = 1.0
    learn.split([model.head1])
    learn.unfreeze()
    learn.fit_one_cycle(10,
                        max_lr=slice(0.2e-2, 1e-2),
                        wd=[1e-3, 0.1e-1],
                        pct_start=0.0,
                        div_factor=100,
                        callbacks=[
                            logger,
                            SaveModelCallback(learn,
                                              monitor='metric_tot',
                                              mode='max',
                                              name=f'model_{fold}')
                        ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="Data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    if (not any(map(lambda f: f.endswith(".png"), os.listdir(args.data_dir)))):
        save_imgs(args.data_dir)
    data = load_imgs(args.data_dir, bs=args.batch_size, fold=args.fold)
    train(data, args.fold)
