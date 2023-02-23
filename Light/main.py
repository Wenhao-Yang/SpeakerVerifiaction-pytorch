#!/usr/bin/env python
# encoding: utf-8
'''
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: VS Code
@File: main.py
@Time: 2023/02/23 18:23
@Overview: 
'''


from argparse import ArgumentParser
# from operator import mod
import torch
import os
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from Light.dataset import SubDatasets, SubLoaders
from Light.model import SpeakerModule
from argparse import ArgumentParser
from hyperpyyaml import load_hyperpyyaml

# torch.multiprocessing.set_sharing_strategy('file_system')

parser = ArgumentParser()
# Trainer arguments
# Hyperparameters for the model

parser.add_argument('--config-yaml', type=str,
                    default='TrainAndTest/Fbank/ResNets/cnc1_resnet_simple.yaml')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed (default: 0)')
parser.add_argument('--gpus', type=str, default='0,1',
                    help='random seed (default: 0)')
args = parser.parse_args()

# seed
pl.seed_everything(args.seed)

# load train config file


def main():
    with open(args.config_yaml, 'r') as f:
        config_args = load_hyperpyyaml(f)

    # Dataset
    train_dir, valid_dir, train_extract_dir = SubDatasets(config_args)
    train_loader, valid_loader, train_extract_loader = SubLoaders(
        train_dir, valid_dir, train_extract_dir, config_args)

    # Model
    model = SpeakerModule(config_args)

    trainer = Trainer(max_epochs=config_args['epochs'], gpus=args.gpus,
                      accelerator='ddp',)
    trainer.fit(model=model, train_dataloader=train_loader,
                val_dataloaders=valid_loader)

    # val_dataloaders=[valid_loader, train_extract_loader])

    # return 0


if __name__ == '__main__':  # pragma: no cover
    main()
