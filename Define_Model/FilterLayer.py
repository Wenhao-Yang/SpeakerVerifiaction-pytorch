#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: FilterLayer.py
@Time: 2020/8/19 20:30
@Overview:
"""

import numpy as np
import torch
from python_speech_features import hz2mel, mel2hz
from torch import nn
from torch.nn.parallel import DistributedDataParallel


class fDLR(nn.Module):
    def __init__(self, input_dim, sr, num_filter, exp=False, filter_fix=False):
        super(fDLR, self).__init__()
        self.input_dim = input_dim
        self.num_filter = num_filter
        self.sr = sr
        self.exp = exp
        self.filter_fix = filter_fix

        requires_grad = not filter_fix
        input_freq = np.linspace(0, self.sr / 2, input_dim)
        self.input_freq = nn.Parameter(torch.from_numpy(input_freq).expand(num_filter, input_dim).float(),
                                       requires_grad=False)

        centers = np.linspace(0, hz2mel(sr / 2), num_filter + 2)
        centers = mel2hz(centers)
        self.frequency_center = nn.Parameter(torch.from_numpy(centers[1:-1]).float().reshape(num_filter, 1),
                                             requires_grad=requires_grad)

        bandwidth = []
        for i in range(2, len(centers)):
            bandwidth.append(centers[i] - centers[i - 1])
        self.bandwidth = nn.Parameter(torch.tensor(bandwidth).reshape(num_filter, 1).float(),
                                      requires_grad=requires_grad)
        # self.gain = nn.Parameter(torch.ones(num_filter, dtype=torch.float32).reshape(num_filter, 1))

    def forward(self, input):
        if self.exp:
            input = torch.exp(input)

        # frequency_center = self.frequency_center.sort(dim=0).values
        new_centers = self.frequency_center.expand(self.num_filter, self.input_dim).clamp(min=0, max=self.sr / 2)
        new_bandwidth = self.bandwidth.clamp(min=1e-12, max=self.sr / 2)
        # if input.is_cuda:
        #     new_centers = new_centers.cuda()
        # pdb.set_trace()
        # power = -1. * torch.pow(self.input_freq - new_centers, 2)
        # power = torch.div(power, 0.5 * self.bandwidth.pow(2))
        dist_center = torch.abs(self.input_freq - new_centers) / new_bandwidth
        dist_center = dist_center.clamp_max(1)
        weights = 1.0 - dist_center

        # weights = torch.exp(power)
        # weights = weights / weights.max(dim=1, keepdim=True).values
        # weights = weights.mul(self.gain).transpose(0, 1)
        weights = weights.transpose(0, 1)

        return torch.log(torch.matmul(input, weights).clamp_min(1e-12))

    def __repr__(self):
        return "fDLR(input_dim=%d, filter_fix=%f, num_filter=%d)" % (self.input_dim, self.filter_fix, self.num_filter)


class grl_func(torch.autograd.Function):
    def __init__(self):
        super(grl_func, self).__init__()

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None


class GRL(nn.Module):
    def __init__(self, lambda_=0.):
        super(GRL, self).__init__()
        self.lambda_ = torch.tensor(lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = torch.tensor(lambda_)

    def forward(self, x):
        return grl_func.apply(x, self.lambda_)


class Inst_Norm(nn.Module):

    def __init__(self, dim):
        super(Inst_Norm, self).__init__()
        self.dim = dim
        self.norm_layer = nn.InstanceNorm1d(self.dim)

    def forward(self, input):
        # alpha = log(p * ( class -2) / (1-p))
        output = input.squeeze().transpose(-1, -2)
        output = self.norm_layer(output)
        output = output.unsqueeze(1).transpose(-1, -2)

        return output

    def __repr__(self):
        return "Inst_Norm(dim=%f)" % self.dim


class Mean_Norm(nn.Module):
    def __init__(self, dim=-2):
        super(Mean_Norm, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x - torch.mean(x, dim=self.dim, keepdim=True)

    def __repr__(self):
        return "Mean_Norm(dim=%d)" % self.dim


class MeanStd_Norm(nn.Module):
    def __init__(self, dim=-2):
        super(MeanStd_Norm, self).__init__()
        self.dim = dim

    def forward(self, x):
        return (x - torch.mean(x, dim=self.dim, keepdim=True)) / x.std()

    def __repr__(self):
        return "MeanStd_Norm(mean_dim=%d, std_dim=all)" % self.dim


class L2_Norm(nn.Module):

    def __init__(self, alpha=1.):
        super(L2_Norm, self).__init__()
        self.alpha = alpha

    def forward(self, input):
        # alpha = log(p * ( class -2) / (1-p))

        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output * self.alpha

    def __repr__(self):
        return "L2_Norm(alpha=%f)" % self.alpha


class TimeMaskLayer(nn.Module):
    def __init__(self, mask_len=25, normalized=False):
        super(TimeMaskLayer, self).__init__()
        self.mask_len = mask_len
        self.normalized = normalized

    def forward(self, x):
        if not self.training:
            return x

        assert self.mask_len < x.shape[-2]

        this_len = np.random.randint(low=0, high=self.mask_len)
        start = np.random.randint(0, x.shape[-2] - this_len)

        if self.normalized:
            one_mask = torch.ones(x.shape)
            one_mask[:, :, start:(start + this_len), :] = 0
            x = x * one_mask
        else:
            this_mean = x.mean(dim=-1, keepdim=True)
            x[:, :, :, start:(start + this_len)] = this_mean[:, :, start:(start + this_len), :].expand(
                (x.shape[0], x.shape[1], this_len, x.shape[3]))

            # x[:, :, start:(start + this_len), :] = this_mean

        return x


class FreqMaskLayer(nn.Module):
    def __init__(self, mask_len=25, normalized=False):
        super(FreqMaskLayer, self).__init__()
        self.mask_len = mask_len
        self.normalized = normalized

    def forward(self, x):
        if not self.training:
            return x

        assert self.mask_len < x.shape[-1]

        this_len = np.random.randint(low=0, high=self.mask_len)
        start = np.random.randint(0, x.shape[-1] - this_len)

        if self.normalized:
            one_mask = torch.ones(x.shape)
            one_mask[:, :, :, start:(start + this_len)] = 0
            x = x * one_mask
        else:
            this_mean = x.mean(dim=-2, keepdim=True)
            x[:, :, :, start:(start + this_len)] = this_mean[:, :, :, start:(start + this_len)].expand(
                (x.shape[0], x.shape[1], x.shape[2], this_len))

        return x


class CBAM(nn.Module):
    # input should be like [Batch, channel, time, frequency]
    def __init__(self, inplanes, planes, time_freq='both'):
        super(CBAM, self).__init__()
        self.time_freq = time_freq

        self.cov_t = nn.Conv2d(inplanes, planes, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.avg_t = nn.AdaptiveAvgPool2d((None, 1))

        self.cov_f = nn.Conv2d(inplanes, planes, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.avg_f = nn.AdaptiveAvgPool2d((1, None))

        self.activation = nn.Sigmoid()

    def forward(self, input):
        t_output = self.avg_t(input)
        t_output = self.cov_t(t_output)
        t_output = self.activation(t_output)
        t_output = input * t_output

        f_output = self.avg_f(input)
        f_output = self.cov_f(f_output)
        f_output = self.activation(f_output)
        f_output = input * f_output

        output = (t_output + f_output) / 2

        return output


class SqueezeExcitation(nn.Module):
    # input should be like [Batch, channel, time, frequency]
    def __init__(self, inplanes, reduction_ratio=4):
        super(SqueezeExcitation, self).__init__()
        self.reduction_ratio = reduction_ratio

        self.glob_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(inplanes, max(int(inplanes / self.reduction_ratio), 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(max(int(inplanes / self.reduction_ratio), 1), inplanes)
        self.activation = nn.Sigmoid()

    def forward(self, input):
        scale = self.glob_avg(input).squeeze(dim=2).squeeze(dim=2)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.activation(scale).unsqueeze(2).unsqueeze(2)

        output = input * scale

        return output


class GAIN(nn.Module):
    def __init__(self, time, freq, scale=2., theta=1.):
        super(GAIN, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.theta = nn.Parameter(torch.tensor(theta))

        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.UpsamplingBilinear2d(size=(time, freq))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, f, f_grad):
        weight = f_grad.mean(dim=(2, 3), keepdim=True)

        T = (f * weight).sum(dim=1, keepdim=True)
        T = self.relu(T)
        T = self.upsample(T)
        T = self.scale * (T - self.theta)
        T = T.clamp(0, 1.0)
        # T = self.sigmoid(T)

        return x - T * x


class Back_GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name):
        super(Back_GradCAM, self).__init__()
        self.net = net
        self.layer_name = layer_name
        self.feature = {}
        self.gradient = {}
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        if isinstance(self.net, DistributedDataParallel):
            self.feature[input[0].device] = output[0]
        else:
            self.feature = output[0]

    #         print("Device {}, forward out feature shape:{}".format(input[0].device, output[0].size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        if isinstance(self.net, DistributedDataParallel):
            if input_grad[0].device not in self.gradient:
                self.gradient[input_grad[0].device] = output_grad[0]
            else:
                self.gradient[input_grad[0].device] += output_grad[0]
        else:
            self.gradient += output_grad[0]

    #         print(output_grad[0])
    #         print("Device {}, backward out gradient shape:{}".format(input_grad[0].device, output_grad[0].size()))

    def _register_hook(self):

        if isinstance(self.net, DistributedDataParallel):
            modules = self.net.module.named_modules()
        else:
            modules = self.net.named_modules()

        for (name, module) in modules:
            if name == self.layer_name:
                self.handlers.append(module.register_backward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self):
        if isinstance(self.net, DistributedDataParallel):
            feature = []
            gradient = []
            for d in self.gradient:
                feature.append(self.feature[d])
                gradient.append(self.gradient[d])

            feature = torch.cat(feature, dim=0)
            gradient = torch.cat(gradient, dim=0)
        else:
            feature = self.feature
            gradient = self.gradient

        return feature, gradient

    # def __call__(self, inputs, index):
    #     """
    #     :param inputs: [1,3,H,W]
    #     :param index: class id
    #     :return:
    #     """
    #     #         self.net.zero_grad()
    #
    #     output, _ = self.net(inputs)  # [1,num_classes]
    #
    #     if index is None:
    #         index = torch.argmax(output)
    #     target = output.gather(1, index)  # .mean()
    #     # target = output[0][index]
    #     for i in target:
    #         i.backward(retain_graph=True)
    #
    #     if isinstance(self.net, DistributedDataParallel):
    #         feature = []
    #         gradient = []
    #         for d in self.gradient:
    #             feature.append(self.feature[d])
    #             gradient.append(self.gradient[d])
    #
    #         feature = torch.cat(feature, dim=0)
    #         gradient = torch.cat(gradient, dim=0)
    #     else:
    #         feature = self.feature
    #         gradient = self.gradient
    #
    #     return feature, gradient
