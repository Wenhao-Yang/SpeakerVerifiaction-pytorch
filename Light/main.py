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
import math
# from operator import mod
import torch
import os
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.profilers import AdvancedProfiler, PyTorchProfiler
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
                    default='TrainAndTest/wav/ecapa/vox2_int_brain_trans.yaml')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed (default: 0)')
parser.add_argument('--gpus', type=str, default='0,1',
                    help='gpus(default: 0)')
parser.add_argument('--dataset-type', type=str,
                    default='scripts', help='gpus(default: 0)')
parser.add_argument('--manual-shuffle', action='store_true',
                    default=False, help='log power spectogram')
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

    this_callbacks = []
    checkpoint_callback = ModelCheckpoint(monitor='Test/EER',
                                          filename='%s-{epoch:02d}-{Test/EER:.2f}' % (
                                              config_args['loss']),
                                          save_top_k=3,
                                          mode='min',
                                          save_last=True)
    this_callbacks.append(checkpoint_callback)

    log_interval = 'step' if config_args['scheduler'] == 'cyclic' else 'epoch'
    this_callbacks.append(LearningRateMonitor(logging_interval=log_interval))

    if args.manual_shuffle:
        this_callbacks.append(ShufTrainset(train_dir=train_dir))

    # profiler = AdvancedProfiler(
    #     filename='profilers')

    # profiler = PyTorchProfiler(
    #     filename='profilers', profile_memory=True, use_cpu=False, use_kineto=True)

    # strategy="ddp_find_unused_parameters_false",
    # precision=16, amp_backend='native',
    val_check_interval = max([math.gcd(len(train_loader)+i, config_args['val_check_interval']+j) for i in range(0, 256) for j in range(-256, 256)])
    
    print('Val interval: {:>7d} (Train: {:>7d} Interval: {:>7d})'.format(val_check_interval, len(train_loader), config_args['val_check_interval']))

    precision = config_args['precision'] if 'precision' in config_args else 32
    trainer = Trainer(max_epochs=config_args['epochs'],
                      accelerator='cuda', devices=args.gpus, strategy="ddp_find_unused_parameters_false",
                      num_sanity_val_steps=0, precision=precision,
                      callbacks=this_callbacks,  # max_steps=100,
                      default_root_dir=config_args['check_path'],
                      # profiler=profiler,
                      check_val_every_n_epoch=None,
                      val_check_interval=val_check_interval,
                      )

    trainer.fit(model=model, train_dataloaders=train_loader,
                val_dataloaders=[train_extract_loader, valid_loader])

    # val_dataloaders=[valid_loader, train_extract_loader])
    # return 0


if __name__ == '__main__':  # pragma: no cover
    main()
