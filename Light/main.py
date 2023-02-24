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
from pytorch_lightning.callbacks import ModelCheckpoint

from Light.dataset import SubDatasets, SubLoaders
from Light.model import SpeakerModule
from argparse import ArgumentParser
from hyperpyyaml import load_hyperpyyaml

# torch.multiprocessing.set_sharing_strategy('file_system')

parser = ArgumentParser()
# Trainer arguments
# Hyperparameters for the model

parser.add_argument('--config-yaml', type=str,
                    default='TrainAndTest/Fbank/ResNets/cnc1_resnet_light.yaml')
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
    model = SpeakerModule(config_args=config_args, train_dir=train_dir)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          filename='%s-%s-{epoch:02d}-{val_loss:.2f}' % (
                                              config_args['datasets'], config_args['loss']),
                                          save_top_k=3,
                                          mode='min',
                                          save_last=True)

    trainer = Trainer(max_epochs=config_args['epochs'],
                      gpus=args.gpus,
                      accelerator='ddp', num_sanity_val_steps=5000,
                      default_root_dir=config_args['check_path'], callbacks=[
                          checkpoint_callback],
                      val_check_interval=0.25,)

    trainer.fit(model=model, train_dataloader=train_loader,
                val_dataloaders=[train_extract_loader, valid_loader])

    # val_dataloaders=[valid_loader, train_extract_loader])

    # return 0


if __name__ == '__main__':  # pragma: no cover
    main()
