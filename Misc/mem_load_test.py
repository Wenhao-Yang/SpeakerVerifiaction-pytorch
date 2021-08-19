#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: mem_load_test.py
@Time: 2020/3/29 10:30 PM
@Overview:
"""
import torch
import numpy as np
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock

model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=1000)

