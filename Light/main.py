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
from pytorch_lightning.profiler.profilers import AdvancedProfiler
from Light.callback import ShufTrainset
from pytorch_lightning.callbacks import LearningRateMonitor

from Light.dataset import SubDatasets, SubLoaders, SubScriptDatasets
from Light.model import SpeakerModule
from argparse import ArgumentParser
from hyperpyyaml import load_hyperpyyaml
# from pytorch_lightning.strategies import DDPStrategy

torch.multiprocessing.set_sharing_strategy('file_system')

parser = ArgumentParser()
# Trainer arguments
# Hyperparameters for the model

parser.add_argument('--config-yaml', type=str,
                    default='TrainAndTest/Fbank/ResNets/cnc1_resnet_light.yaml')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed (default: 0)')
parser.add_argument('--gpus', type=str, default='0,1',
                    help='gpus(default: 0)')
parser.add_argument('--dataset-type', type=str,
                    default='scripts', help='gpus(default: 0)')
parser.add_argument('--test', action='store_true',
                    default=False, help='gpus(default: 0)')

args = parser.parse_args()
# seed
pl.seed_everything(args.seed)

# load train config file


def main():
    with open(args.config_yaml, 'r') as f:
        config_args = load_hyperpyyaml(f)

    # Dataset
    if args.dataset_type == 'egs':
        train_dir, valid_dir, train_extract_dir = SubDatasets(config_args)
    else:
        train_dir, valid_dir, train_extract_dir = SubScriptDatasets(
            config_args)

    train_loader, valid_loader, train_extract_loader = SubLoaders(
        train_dir, valid_dir, train_extract_dir, config_args)

    # Model
    model = SpeakerModule(config_args=config_args)
    # model._set_hparams(config_args=config_args, train_dir=train_dir)
    model._set_hparams({'config_args': config_args})

    checkpoint_callback = ModelCheckpoint(monitor='val_eer',
                                          filename='%s-{epoch:02d}-{val_eer:.2f}' % (
                                              config_args['loss']),
                                          save_top_k=3,
                                          mode='min',
                                          save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    shuf_train_callback = ShufTrainset(train_dir=train_dir)

    # profiler = AdvancedProfiler(
    # output_filename=config_args['check_path']+'/profilers')
    # profiler=profiler,
    trainer = Trainer(max_epochs=config_args['epochs'],
                      accelerator='ddp', gpus=args.gpus,
                      num_sanity_val_steps=0,
                      callbacks=[checkpoint_callback,
                                 shuf_train_callback, lr_monitor],
                      default_root_dir=config_args['check_path'],
                      val_check_interval=0.5, gradient_clip_val=1.0,
                      weights_summary='full')

    trainer.fit(model=model, train_dataloader=train_loader,
                val_dataloaders=[train_extract_loader, valid_loader])

    # val_dataloaders=[valid_loader, train_extract_loader])

    # return 0


if __name__ == '__main__':  # pragma: no cover
    main()
