#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: cam_2.py
@Time: 2021/4/12 21:47
@Overview:
    Created on 2019/8/4 上午9:37
    @author: mick.yi
"""

import os
import pdb

import numpy as np
import torch
from torch.nn.parallel.distributed import DistributedDataParallel

from Define_Model.ResNet import ThinResNet


class GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = {}
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output
        print("forward feature shape:{}".format(output[0].size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        self.gradient[input_grad[0].device] = output_grad[0]
        print("Device {}, backward feature shape:{}".format(input_grad.device, output_grad[0].size()))

    def _register_hook(self):

        if isinstance(self.net, DistributedDataParallel):
            modules = self.net.module.named_modules()
        else:
            modules = self.net.named_modules()

        for (name, module) in modules:
            if name == self.layer_name:
                self.handlers.append(module.register_full_backward_hook(self._get_features_hook))
                self.handlers.append(module.register_full_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index):
        """
        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        output, _ = self.net(inputs)  # [1,num_classes]
        pdb.set_trace()
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output.gather(1, index).mean()

        # target = output[0][index]
        target.backward()

        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        # weight = np.mean(gradient, axis=(1, 2))  # [C]
        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]

        # cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        # cam = np.sum(cam, axis=0)  # [H,W]
        # cam = np.maximum(cam, 0)  # ReLU
        #
        # # 数值归一化
        # cam -= np.min(cam)
        # cam /= np.max(cam)
        # # resize to 224*224
        # cam = cv2.resize(cam, (224, 224))
        # return cam
        print("gradient: ", gradient.shape)
        print("feature: ", feature.shape)


os.environ['CUDA_VISIBLE_DEVICES'] = "1,3"
torch.distributed.init_process_group(backend="nccl", init_method='tcp://localhost:12556', rank=0,
                                     world_size=1)

model = ThinResNet()
model = model.cuda()
model = DistributedDataParallel(model)
gc = GradCAM(model, 'layer4')

x = torch.randn((20, 1, 224, 224)).cuda()
l = torch.range(0, 19).long().unsqueeze(1).cuda()

y = model(x)

#
cam = gc(x, l)

print(cam.shape)
