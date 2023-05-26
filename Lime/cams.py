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
import torch
import torch.nn.functional as F
 
def InputGradeint(data) -> torch.Tensor:
    return data.grad
    

def Grad_CAM(data, feat_lst, grad_lst):
    grad_cam = feat_lst[-1] * grad_lst[0]
    if len(grad_cam.shape) == 3:
        grad_cam = grad_cam.unsqueeze(0)
        
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
        grad = F.pad(grad, (1,1,1,1), "constant", 0)

    grad_cam_pp = F.interpolate(grad, size=data.shape[-2:],
                              mode='bilinear', align_corners=True)
    
    return grad_cam_pp
    

def FullGrad(data, feat_lst, grad_lst, biases_lst, bias_layers, model,
             zero_padding=False):
    input_gradient = (data.grad * data)
    full_grad = input_gradient.abs()#.cpu()
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
            bias_grad = F.pad(bias_grad, (1,1,1,1), "constant", 0)

        full_grad += F.interpolate(bias_grad, size=data.shape[-2:],
                              mode='bilinear', align_corners=True)

    return full_grad



def Grad_CAM(data, feat_lst, grad_lst):
    grad_cam = feat_lst[-1] * grad_lst[0]
    if len(grad_cam.shape) == 3:
        grad_cam = grad_cam.unsqueeze(0)
        
    grad_cam = F.interpolate(grad_cam, size=data.shape[-2:],
                              mode='bilinear', align_corners=True)
    
    return grad_cam

def Grad_CAM(data, feat_lst, grad_lst):
    grad_cam = feat_lst[-1] * grad_lst[0]
    if len(grad_cam.shape) == 3:
        grad_cam = grad_cam.unsqueeze(0)
        
    grad_cam = F.interpolate(grad_cam, size=data.shape[-2:],
                              mode='bilinear', align_corners=True)
    
    return grad_cam

def Grad_CAM(data, feat_lst, grad_lst):
    grad_cam = feat_lst[-1] * grad_lst[0]
    if len(grad_cam.shape) == 3:
        grad_cam = grad_cam.unsqueeze(0)
        
    grad_cam = F.interpolate(grad_cam, size=data.shape[-2:],
                              mode='bilinear', align_corners=True)
    
    return grad_cam