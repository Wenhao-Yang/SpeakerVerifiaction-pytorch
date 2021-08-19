#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: test.py
@Time: 2019/9/13 下午4:54
@Overview:
"""
from Define_Model.model import ResSpeakerModel
import torch
from thop import profile


model = ResSpeakerModel(resnet_size=10, embedding_size=512, num_classes=1211)

a = torch.ones((1, 1, 300, 257))
b = model(a)

print(b)
