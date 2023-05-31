#!/usr/bin/env python
# encoding: utf-8
'''
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: VS Code
@File: gradient.py
@Time: 2023/05/26 12:19
@Overview: 
'''
import pdb
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import captum


def InputGradeint(data) -> torch.Tensor:
    return data.grad


def Grad_CAM(data, feat_lst, grad_lst,
             zero_padding=False):

    grad_cam = feat_lst[-1] * grad_lst[0]
    if len(grad_cam.shape) == 3:
        grad_cam = grad_cam.unsqueeze(0)

    if zero_padding and grad.shape[-1] < data.shape[-1]:
        grad = F.pad(grad, (1, 1, 1, 1), "constant", 0)

    grad_cam = F.interpolate(grad_cam, size=data.shape[-2:],
                             mode='bilinear', align_corners=True)

    return grad_cam


def Grad_CAM_pp(data, feat_lst, grad_lst, logits, label,
                zero_padding=False):

    grad_cam = feat_lst[-1] * grad_lst[0]
    if len(grad_cam.shape) == 3:
        grad_cam = grad_cam.unsqueeze(0)

    last_grad = grad_lst[0]
    last_feat = feat_lst[-1]

    first_derivative = logits[0][label.long()].exp().cpu() * last_grad

    alpha = last_grad.pow(2) / (
        2 * last_grad.pow(2) + (last_grad.pow(3) * last_feat).sum(dim=(2, 3), keepdim=True))
    weight = alpha * (first_derivative.clamp_min_(0))

    weight = weight.mean(dim=(2, 3), keepdim=True)
    weight /= weight.sum()

    grad = (last_feat * weight).sum(dim=1, keepdim=True).clamp_min(0)

    if zero_padding and grad.shape[-1] < data.shape[-1]:
        grad = F.pad(grad, (1, 1, 1, 1), "constant", 0)

    grad_cam_pp = F.interpolate(grad, size=data.shape[-2:],
                                mode='bilinear', align_corners=True)

    return grad_cam_pp


def FullGrad(data, feat_lst, grad_lst, biases_lst, bias_layers, model,
             zero_padding=False):

    input_gradient = (data.grad * data)
    full_grad = input_gradient.abs()  # .cpu()
    full_grad -= full_grad.min()
    full_grad /= full_grad.max() + 1e-8

    for i, l in enumerate(bias_layers):
        bias = biases_lst[L - i - 1]
        if len(bias.shape) == 1:
            bias = bias.reshape(1, -1, 1, 1)

        grads_shape = feat_lst[i].shape
        if len(grads_shape) == 3:
            # pdb.set_trace()
            if bias.shape[1] == grads_shape[-1]:
                if model.avgpool != None and bias.shape[1] % model.avgpool.output_size[1] == 0:
                    bias = bias.reshape(
                        1, -1, 1, model.avgpool.output_size[1])
                    grad_lst[i] = grad_lst[i].reshape(1, -1, grads_shape[1],
                                                      model.avgpool.output_size[1])
                else:
                    grad_lst[i] = grad_lst[i].reshape(1, bias.shape[1],
                                                      grads_shape[1], -1)

        try:
            bias = bias.expand_as(grad_lst[i])
        except Exception as e:
            pdb.set_trace()

        bias_grad = (grad_lst[i] * bias).mean(dim=1, keepdim=True).abs()
        bias_grad -= bias_grad.min()
        bias_grad /= bias_grad.max() + 1e-8

        if zero_padding and bias_grad.shape[-1] < input_gradient.shape[-1]:
            bias_grad = F.pad(bias_grad, (1, 1, 1, 1), "constant", 0)

        full_grad += F.interpolate(bias_grad, size=data.shape[-2:],
                                   mode='bilinear', align_corners=True)

    return full_grad


def Feat_Grad(data, feat_lst, grad_lst,
              zero_padding=False):

    input_gradient = (data.grad * data)
    grad = input_gradient.clamp_min(0)
    grad -= grad.min()

    for i in range(len(feat_lst)):

        this_grad = grad_lst[i] * feat_lst[-1 - i]
        if zero_padding and this_grad.shape[-1] < input_gradient.shape[-1]:
            this_grad = F.pad(this_grad, (1, 1, 1, 1), "constant", 0)

        this_grad = F.interpolate(this_grad, size=data.shape[-2:],
                                  mode='bilinear', align_corners=True).mean(dim=1, keepdim=True)

        this_grad = this_grad.clamp_min(0)
        this_grad /= this_grad.max()

        grad += this_grad

    feat_grad = grad / grad.sum()

    return feat_grad


def Feats(data, feat_lst, grad_lst,
          zero_padding=False, layer_weight=False):

    grad = torch.zeros_like(data)
    # acc_grad /= acc_grad.max()

    for i in range(len(feat_lst)):
        this_feat = feat_lst[-1 - i]

        if len(this_feat.shape) == 3:
            this_feat = this_feat.unsqueeze(0)

        if zero_padding and this_feat.shape[-1] < grad.shape[-1]:
            this_feat = F.pad(this_feat, (1, 1, 1, 1), "constant", 0)

        this_grad = F.interpolate(this_feat, size=data.shape[-2:],
                                  mode='bilinear', align_corners=True).mean(dim=1, keepdim=True)

        this_grad = this_grad.clamp_min(0)
        this_grad /= this_grad.max() + 1e-6

        if np.isnan(this_grad.detach().cpu().numpy()).any():
            pdb.set_trace()

        grad += this_grad if layer_weight else (len(feat_lst) - i) / len(
            feat_lst) * this_grad

    grad = grad / grad.sum()

    return grad


def calculate_outputs_and_gradients(inputs, model, target_label_idx):

    gradients = []
    for s in inputs:
        s = Variable(s.cuda(), requires_grad=True)

        output, _ = model(s)
        output = F.softmax(output, dim=1)

        if target_label_idx is None:
            target_label_idx = torch.argmax(output, 1).item()

        index = torch.ones((output.size()[0], 1)) * target_label_idx
        index = torch.tensor(index, dtype=torch.int64)
        if s.is_cuda:
            index = index.cuda()

        output = output.gather(1, index)
        model.zero_grad()
        output.backward()

        gradient = s.grad.detach()
        gradients.append(gradient)

    gradients = torch.cat(gradients)

    return gradients, target_label_idx


def InteGrad(data, baseline, steps, model, label):
    if baseline is None:
        baseline = 0. * data

    scaled_inputs = [baseline + (float(i) / steps) * (data - baseline) for i in
                     range(1, steps + 1)]

    grads, _ = calculate_outputs_and_gradients(
        scaled_inputs, model, label)

    with torch.no_grad():
        grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = grads.mean(dim=0)
        grad = (data - baseline).cuda() * avg_grads

    return grad


def ExpectGrad(data, steps, model, label, samples):

    scaled_inputs = []
    baselines = []
    for i in range(steps):
        alpha = np.random.uniform(0, 1)
        b = np.random.choice(samples)

        while b.shape[-2] < data.shape[-2]:
            b = torch.cat([b, b], dim=-2)

        b = b[:, :, :data.shape[-2], :]

        scaled_inputs.append(data + alpha * (data-b))
        baselines.append(data - b)

    baselines = torch.cat(baselines, dim=0)

    grads, _ = calculate_outputs_and_gradients(
        scaled_inputs, model, label)

    with torch.no_grad():
        grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = grads.mean(dim=0)
        grad = baselines.cuda() * avg_grads

    return grad


def SmoothGrad(data,  model, label,
               steps=25,
               stdev_spread=0.15, magnitude=True):
    # baseline,
    # if baseline is None:
    #     baseline = 0. * data

    # original smooth use a std normal distribution here
    #  stdev = stdev_spread * (np.max(x) - np.min(x))

    stdev = stdev_spread * \
        (data.max(dim=-2, keepdim=True).values -
         data.min(dim=-2, keepdim=True).values)
    scaled_inputs = [data + torch.normal(torch.zeros_like(
        data), stdev.repeat(1, 1, data.shape[-2], 1)) for i in range(steps)]

    grad, _ = calculate_outputs_and_gradients(
        scaled_inputs, model, label)

    with torch.no_grad():
        if magnitude:
            grad = grad.pow(2)

        grad = grad.mean(dim=0)

    return grad


def GradientShap(data,  model, label, baselines):

    gradshap = captum.attr.GradientShap(model)
    grad = gradshap.attribute(data, baselines, target=label)

    return grad
