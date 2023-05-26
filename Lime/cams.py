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
        grad = zeros(grad)

  
    grad_cam_pp = F.interpolate(grad, size=data.shape[-2:],
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