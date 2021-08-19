#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: cam.py
@Time: 2021/4/12 21:42
@Overview:
"""
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
from PIL import Image


class Guided_backProp():
    def __init__(self, model):
        self.model = model
        # self.activations = []
        self.input_grad = None
        self.add_hook()

    # def forward_hook(self,module,input,output):
    #     self.activations.append(output)

    def backward_hook(self, module, grad_in, grad_out):
        return (torch.clamp(grad_in[0], 0.),)  # relu(gradient)

    def firstlayer_backward_hook(self, module, grad_in, grad_out):
        # 对卷积层，grad_input = (对输入feature的导数，对权重 W 的导数，对 bias 的导数)
        # 对全连接层，grad_input=(对 bias 的导数，对输入feature 的导数，对 W 的导数)
        self.input_grad = grad_in[0]

    def add_hook(self):
        modules = list(self.model.named_children())

        for name, layer in modules:
            if isinstance(layer, nn.ReLU):
                # layer.register_forward_hook(self.forward_hook)
                layer.register_backward_hook(self.backward_hook)

        first_layer = modules[0][1]
        first_layer.register_backward_hook(self.firstlayer_backward_hook)

    def __call__(self, inputs, index, class_names):
        self.model.eval()
        pred = self.model(Img_trans)
        self.model.zero_grad()

        pred = torch.nn.functional.softmax(pred)
        if index == None:
            val, index = torch.max(pred, dim=1)

        print('预测结果:%s' % class_names[str(index.item())])

        one_hot = torch.zeros(pred.shape)
        one_hot[0][index] = 1

        # backward
        pred.backward(one_hot)

        return self.input_grad[0].permute(1, 2, 0).numpy(), False


class GradCAM():
    def __init__(self, model, module_name, layer_name):
        self.model = model
        self.model.eval()
        self.module_name = module_name
        self.layer_name = layer_name
        self.features = None
        self.grad = None
        self.add_hook()

    def forward_hook(self, layer, input, output):
        self.features = output

    def backward_hook(self, layer, grad_in, grad_out):
        self.grad = grad_in[0]

    def add_hook(self):
        for name_module, module in self.model.named_children():
            if name_module == self.module_name:
                for name_layer, layer in module.named_children():
                    if name_layer == self.layer_name:
                        layer.register_forward_hook(self.forward_hook)
                        layer.register_backward_hook(self.backward_hook)

    def __call__(self, input, index, class_names):
        self.model.eval()
        self.model.zero_grad()

        pred = self.model(input)
        if index == None:
            val, index = torch.max(pred, dim=1)

        print('预测结果:%s' % class_names[str(index.item())])

        one_hot = torch.zeros(pred.shape)
        one_hot[0][index] = 1

        pred.backward(one_hot)

        w_c = torch.mean(self.grad.view(self.grad.shape[0], self.grad.shape[1], -1), dim=2)

        output = w_c[0].view(-1, 1, 1) * self.features[0]

        output = torch.sum(output, dim=0)

        return output.detach().numpy(), True


class GradCAMPlusPlus(GradCAM):
    def __call__(self, input, index, class_names):
        self.model.eval()
        self.model.zero_grad()

        pred = self.model(input)
        if index == None:
            val, index = torch.max(pred, dim=1)

        print('预测结果:%s' % class_names[str(index.item())])

        one_hot = torch.zeros(pred.shape)
        one_hot[0][index] = 1

        pred.backward(one_hot)

        #
        grad = torch.clamp(self.grad, min=0.)
        alpha_num = self.grad.pow(2)
        alpha_denom = 2 * alpha_num + self.features.mul(self.grad.pow(3)).view(self.features.shape[0],
                                                                               self.features.shape[1], -1).sum(
            dim=2).view(self.features.shape[0], self.features.shape[1], 1, 1)
        alpha = alpha_num / (alpha_denom + 1e-8)
        weights = (alpha * grad).view(alpha.shape[0], alpha.shape[1], -1).sum(dim=2)
        saliency_map = (weights.view(weights.shape[0], weights.shape[1], 1, 1) * self.features).sum(dim=1)
        output = torch.clamp(saliency_map, 0.)
        return output[0].detach().numpy(), True


def Show_Res(Img, input_grad, bAdd):
    if bAdd:
        input_grad = (input_grad - input_grad.min()) / input_grad.max()
        image = np.array(Img).astype(np.float) / 255.
        rez_img = cv2.resize(input_grad, (Img.size[0], Img.size[1]))
        rez_img = (rez_img * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(rez_img, cv2.COLORMAP_JET).astype(np.float)
        heatmap /= 255.
        result = heatmap + image
        result = result / result.max()
    else:
        # 归一化input_grad
        input_grad = (input_grad - input_grad.mean()) / input_grad.std()
        input_grad *= 0.1
        input_grad += 0.5
        input_grad = cv2.resize(input_grad, (Img.size[0], Img.size[1]))
        result = input_grad.clip(0, 1)

    return result


if __name__ == "__main__":
    # 读取一张图像，预处理
    Img = Image.open('./DogAndCat.jpg').convert('RGB')  # DogAndCat.jpg

    mean = [0.485, 0.456, 0.406]
    std = (0.229, 0.224, 0.225)
    transfer = transform.Compose([transform.Resize((224, 224)),
                                  transform.ToTensor(),
                                  transform.Normalize(mean, std)])

    Img_trans = transfer(Img).unsqueeze(0).requires_grad_()

    # class-names
    with open('./labels.json') as fp:
        class_names = json.load(fp)

    # 加载一个预训练模型
    model = torchvision.models.resnet34(pretrained=True)

    # gb
    name = 'gb.jpg'
    gb = Guided_backProp(model)
    output, bAdd = gb(Img_trans, None, class_names)
    GBResImg = Show_Res(Img, output, bAdd)
    cv2.imwrite(name, (GBResImg * 255).astype(np.uint8))

    # grad-cam
    name = 'grad-cam.jpg'
    cam = GradCAM(model, 'layer4', '2')
    output, bAdd = cam(Img_trans, None, class_names)
    GradCamResImg = Show_Res(Img, output, bAdd)
    cv2.imwrite(name, (GradCamResImg * 255).astype(np.uint8))

    # grad-cam++
    name = 'grad-cam++.jpg'
    cam = GradCAMPlusPlus(model, 'layer4', '2')
    output, bAdd = cam(Img_trans, None, class_names)
    GradCamPPResImg = Show_Res(Img, output, bAdd)
    cv2.imwrite(name, (GradCamPPResImg * 255).astype(np.uint8))

    # show
    ResImg = np.hstack((GBResImg, GradCamResImg, GradCamPPResImg))
    cv2.imshow('Res', ResImg)
    cv2.waitKey()
