#!/usr/bin/env python
# encoding: utf-8
'''
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: VS Code
@File: callback.py
@Time: 2023/02/24 22:07
@Overview: 
'''

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from typing import Any


class ShufTrainset(Callback):
    def __init__(self, train_dir) -> None:
        super().__init__()
        self.train_dir = train_dir

    def on_train_epoch_end(self, trainer, pl_module: LightningModule, outputs: Any) -> None:
        self.train_dir.__shuffle__()
        pl_module.print('Shuffle Training set!')
        return super().on_train_epoch_end(trainer, pl_module, outputs)
