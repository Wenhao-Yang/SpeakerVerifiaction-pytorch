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

import math

import numpy as np
import torch
import torch.nn.functional as F
from python_speech_features import hz2mel, mel2hz
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from scipy import interpolate

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
        return "fDLR(input_dim=%d, filter_fix=%s, num_filter=%d)" % (self.input_dim, self.filter_fix, self.num_filter)


class fBLayer(nn.Module):
    def __init__(self, input_dim, sr, num_filter, exp=False, filter_fix=False):
        super(fBLayer, self).__init__()
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
        bandwidth = np.diff(centers)
        self.frequency_center = nn.Parameter(torch.from_numpy(centers[1:-1]).float().reshape(num_filter, 1),
                                             requires_grad=requires_grad)

        self.bandwidth_left = nn.Parameter(torch.from_numpy(bandwidth[:-1]).float().reshape(num_filter, 1),
                                           requires_grad=requires_grad)
        self.bandwidth_right = nn.Parameter(torch.from_numpy(bandwidth[1:]).float().reshape(num_filter, 1),
                                            requires_grad=requires_grad)

    def forward(self, input):
        if self.exp:
            input = torch.exp(input)

        frequency_center = self.frequency_center.clamp(min=0, max=self.sr / 2)
        bandwidth_left = self.bandwidth_left.clamp(min=1e-12, max=self.sr / 2)
        bandwidth_right = self.bandwidth_right.clamp(min=1e-12, max=self.sr / 2)
        new_centers = frequency_center.expand(self.num_filter, self.input_dim)

        dist_center_a = (new_centers - self.input_freq) / bandwidth_left
        dist_center_a = 1.0 - dist_center_a.clamp(min=0., max=1.)

        dist_center_b = (self.input_freq - new_centers) / bandwidth_right
        dist_center_b = 1.0 - dist_center_b.clamp(min=0., max=1.)
        weights = dist_center_a + dist_center_b
        weights = weights.transpose(0, 1).clamp(min=0., max=1.)

        return torch.log(torch.matmul(input, weights).clamp_min(1e-12))

    def __repr__(self):
        return "fBLayer(input_dim=%d, filter_fix=%s, num_filter=%d)" % (
            self.input_dim, self.filter_fix, self.num_filter)


class fBPLayer(nn.Module):

    def __init__(self, input_dim, sr, num_filter, exp=False, filter_fix=False):
        super(fBPLayer, self).__init__()
        self.input_dim = input_dim
        self.num_filter = num_filter
        self.sr = sr
        self.exp = exp
        self.filter_fix = filter_fix

        requires_grad = not filter_fix
        input_freq = np.linspace(0, self.sr / 2, input_dim)
        self.input_freq = nn.Parameter(torch.from_numpy(input_freq).expand(num_filter, input_dim).float(),
                                       requires_grad=False)

        borders = np.linspace(0, hz2mel(sr / 2), num_filter + 2)
        borders = mel2hz(borders)

        self.bandwidth_low = nn.Parameter(torch.from_numpy(borders[:-2]).float().reshape(num_filter, 1),
                                          requires_grad=requires_grad)

        self.bandwidth = nn.Parameter(torch.from_numpy(borders[2:] - borders[:-2]).float().reshape(num_filter, 1),
                                      requires_grad=requires_grad)

    def forward(self, input):
        if self.exp:
            input = torch.exp(input)

        bandwidth_low = self.bandwidth_low.clamp(min=1e-12, max=self.sr / 2)
        bandwidth_high = (bandwidth_low + self.bandwidth.clamp(min=1e-12, max=self.sr / 2)).clamp_max(self.sr / 2)

        bandwidth_low = bandwidth_low.expand(self.num_filter, self.input_dim)
        bandwidth_high = bandwidth_high.expand(self.num_filter, self.input_dim)

        low_mask = torch.where(self.input_freq >= bandwidth_low, torch.ones_like(bandwidth_low),
                               torch.zeros_like(bandwidth_low))
        high_mask = torch.where(self.input_freq <= bandwidth_high, torch.ones_like(bandwidth_high),
                                torch.zeros_like(bandwidth_high))

        mask = low_mask * high_mask
        weights = mask.transpose(0, 1)

        return torch.log(torch.matmul(input, weights).clamp_min(1e-12))

    def __repr__(self):
        return "fBPLayer(input_dim=%d, filter_fix=%s, num_filter=%d)" % (
            self.input_dim, self.filter_fix, self.num_filter)


class fLLayer(nn.Module):
    def __init__(self, input_dim, num_filter, exp=False):
        super(fLLayer, self).__init__()
        self.input_dim = input_dim
        self.num_filter = num_filter
        self.exp = exp

        self.linear = nn.Linear(input_dim, num_filter, bias=False)
        # nn.init.normal_(self.linear.weight, mean=0.5, std=0.25)
        self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm2d(num_filter)
        # nn.init.constant_(self.linear.weight, 1.0)

    def forward(self, input):
        if self.exp:
            input = torch.exp(input)

        input = self.linear(input)
        input = self.relu(input)
        # input = self.bn(input)

        return torch.log(input.clamp_min(1e-12))

    def __repr__(self):
        return "fLLayer(input_dim=%d, num_filter=%d) without batchnorm2d " % (
            self.input_dim, self.num_filter)

# https://github.com/mravanelli/SincNet
class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        # 反变换到线性频率域
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast, self).__init__()

        if in_channels != 1:
            # msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        # the out_channels is 80 in the paper
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        # kernel_size will be 251 in the paper
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        # 计算mel尺度下的滤波器参数，反变换回线性域
        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        # 滤波器起始频率
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        # 滤波器宽度
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1,
                               steps=int((self.kernel_size / 2)))  # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size);

        # (kernel_size, 1)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1,
                                                         -1) / self.sample_rate  # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (
                self.n_ / 2)) * self.window_  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])

        # 时域滤波器
        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)


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

class RevGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output
        return grad_input

class RevGradLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        A gradient reversal layer.

        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

    def forward(self, input_):
        return RevGrad.apply(input_)

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

        # assert self.mask_len < x.shape[-2]

        this_len = np.random.randint(low=0, high=self.mask_len)
        start = np.random.randint(0, x.shape[-2] - this_len)
        x_shape = len(x.shape)

        this_mean = x.mean(dim=-2, keepdim=True)
        if x_shape == 4:
            x[:, :, start:(start + this_len), :] = this_mean
        elif x_shape == 3:
            x[:, start:(start + this_len), :] = this_mean

        return x

    def __repr__(self):
        return "TimeMaskLayer(mask_len=%f)" % self.mask_len


class FreqMaskLayer(nn.Module):
    def __init__(self, mask_len=25, normalized=False):
        super(FreqMaskLayer, self).__init__()
        self.mask_len = mask_len
        self.normalized = normalized

    def forward(self, x):
        if not self.training:
            return x

       # assert self.mask_len < x.shape[-1]
        this_len = np.random.randint(low=0, high=self.mask_len)
        start = np.random.randint(0, x.shape[-1] - this_len)
        x_shape = len(x.shape)

        this_mean = x.mean(dim=-1, keepdim=True)  # .add(1e-6)
        if x_shape == 4:
            x[:, :, :, start:(start + this_len)] = this_mean
        elif x_shape == 3:
            x[:, :, start:(start + this_len)] = this_mean

        return x

    def __repr__(self):
        return "FreqMaskLayer(mask_len=%f)" % self.mask_len


class DropweightLayer(nn.Module):
    def __init__(self, dropout_p=0.1):
        super(DropweightLayer, self).__init__()

        m = np.arange(0, 2840.0230467083188)
        m = 700 * (10 ** (m / 2595.0) - 1)
        n = np.array([m[i] - m[i - 1] for i in range(1, len(m))])
        n = 1 / n
        x = np.arange(161) * 8000 / (161 - 1)  # [0-8000]

        f = interpolate.interp1d(m[1:], n)
        xnew = np.arange(np.min(m[1:]), np.max(m[1:]), (np.max(m[1:]) - np.min(m[1:])) / 161)
        ynew = f(xnew)
        ynew = 1 / ynew  # .max()
        ynew /= ynew.max()

        self.drop_p = ynew * dropout_p

    def forward(self, x):
        if not self.training:
            return x
        else:
            assert len(self.drop_p) == x.shape[-1]
            drop_weight = []
            for i in self.drop_p:
                drop_weight.append((torch.Tensor(1).uniform_(0, 1) > i).float())

            drop_weight = torch.tensor(drop_weight).reshape(1, 1, 1, -1)

            if x.is_cuda:
                drop_weight = drop_weight.cuda()

            return x * drop_weight


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

    def __repr__(self):
        return "SqueezeExcitation(reduction_ratio=%f)" % self.reduction_ratio


class GAIN(nn.Module):
    def __init__(self, time, freq, scale=1.1, theta=0.5):
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

        # Put on the same device
        if x.is_cuda:
            T = T.cuda()
            scale = self.scale.cuda()  # .clamp_min(1e-6)
            theta = self.theta.cuda()

        # pdb.set_trace()
        T = scale * (T - theta)
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
        # print(type(self.net))

        if isinstance(self.net, DistributedDataParallel):
            if input_grad[0].device not in self.gradient:
                self.gradient[input_grad[0].device] = output_grad[0]
            else:
                self.gradient[input_grad[0].device] += output_grad[0]
        else:
            self.gradient = output_grad[0]

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

            feature = torch.cat(feature, dim=0).cuda()
            gradient = torch.cat(gradient, dim=0).cuda()
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

# https://github.com/mravanelli/SincNet
class Sinc2Conv(nn.Module):
    def __init__(self, input_dim, out_dim=60, fs=16000):
        super(Sinc2Conv, self).__init__()
        self.fs = fs
        self.current_input = input_dim
        self.out_dim = out_dim

        # conv_layers = [(80, 251, 1), (60, 5, 1), (out_dim, 5, 1)]
        self.conv_layers = nn.ModuleList()
        self.sinc_conv = nn.Sequential(
            SincConv_fast(80, 251, self.fs, stride=6),
            nn.MaxPool1d(kernel_size=3),  # nn.AvgPool1d(kernel_size=3),
            nn.InstanceNorm1d(80),  # nn.LayerNorm([80, int((self.current_input - 251 + 1) / 6 / 3)]),
            nn.LeakyReLU(),
        )

        self.current_input = int((self.current_input - 251 + 1) / 6 / 3)
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=80, out_channels=60, kernel_size=5, stride=1),
            nn.MaxPool1d(kernel_size=3),  # nn.AvgPool1d(kernel_size=3),
            nn.InstanceNorm1d(60),  # nn.LayerNorm([60, int((self.current_input - 5 + 1) / 3)]),
            nn.LeakyReLU(),
        )

        self.current_input = int((self.current_input - 5 + 1) / 3)
        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=self.out_dim, kernel_size=5, stride=1),
            nn.MaxPool1d(kernel_size=3),
            nn.InstanceNorm1d(self.out_dim),  # nn.LayerNorm([self.out_dim, int((self.current_input - 5 + 1) / 3)]),
            nn.LeakyReLU(),
        )

        # self.conv_layer4 = nn.Sequential(
        #     nn.Conv1d(in_channels=128, out_channels=self.out_dim, kernel_size=5, stride=2),
        #     nn.AvgPool1d(kernel_size=3),  # nn.MaxPool1d(kernel_size=3),
        #     nn.InstanceNorm1d(self.out_dim),  # nn.LayerNorm([self.out_dim, int((self.current_input - 5 + 1) / 3)]),
        #     nn.LeakyReLU(),
        # )

        self.current_output = int((self.current_input - 5 + 1) / 3)

    def forward(self, x):
        # BxT -> BxCxT
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 4:
            x = x.squeeze(1)

        x = self.sinc_conv(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        # x = self.conv_layer4(x)

        return x.transpose(1, 2)

    # 20210604 maxpooling :
    # Epoch 40: Train Accuracy: 99.828235%, Avg loss: 0.139713.
    #           Valid Accuracy: 88.391376%, Avg loss: 0.597342.
    #           Train EER: 21.7448%, Threshold: 0.0681, mindcf-0.01: 0.8932, mindcf-0.001: 0.8932.
    #           Test  ERR: 30.0954%, Threshold: 0.0494, mindcf-0.01: 0.9014, mindcf-0.001: 0.9014.

    # # 20210604 avgpooling :
    # Epoch 29: Train Accuracy: 82.360265%, Avg loss: 0.826197.
    #           Valid Accuracy: 74.087894%, Avg loss: 1.180882.
    #           Train EER: 27.2819%, Threshold: 0.0729, mindcf-0.01: 0.9020, mindcf-0.001: 0.9020.
    #           Test  ERR: 31.4687%, Threshold: 0.0716, mindcf-0.01: 0.8942, mindcf-0.001: 0.9099.

    # 20210605 con4 80 64 128
    # Epoch 40: Train Accuracy: 99.947745%, Avg loss: 0.096998.
    #           Valid Accuracy: 94.361526%, Avg loss: 0.361226.
    #           Train EER: 18.6485%, Threshold: 0.0850, mindcf-0.01: 0.8663, mindcf-0.001: 0.9555.
    #           Test  ERR: 28.2980%, Threshold: 0.0558, mindcf-0.01: 0.8640, mindcf-0.001: 0.8996.

    # 20210605 sinc ==> lr 0.01 5e4
    # Epoch 40: Train Accuracy: 99.999839%, Avg loss: 0.038800.
    #           Valid Accuracy: 99.170813%, Avg loss: 0.109573.
    #           Train EER: 2.0071%, Threshold: 0.2855, mindcf-0.01: 0.2954, mindcf-0.001: 0.5138.
    #           Test  ERR: 7.1262%, Threshold: 0.2042, mindcf-0.01: 0.6215, mindcf-0.001: 0.8115.

# https://github.com/pytorch/fairseq/blob/c47a9b2eef0f41b0564c8daf52cb82ea97fc6548/fairseq/models/wav2vec/wav2vec.py#L367
class Wav2Conv(nn.Module):
    def __init__(self, out_dim=512, log_compression=True):
        super(Wav2Conv, self).__init__()

        in_d = 1
        conv_layers = [(40, 10, 5), (200, 5, 4), (300, 3, 2), (512, 3, 2), (out_dim, 3, 2)]
        self.conv_layers = nn.ModuleList()
        for dim, k, stride in conv_layers:
            self.conv_layers.append(self.block(in_d, dim, k, stride))
            in_d = dim
        self.tmp_gate = nn.Sequential(
            nn.Linear(out_dim, 1),
            nn.Sigmoid()
        )
        self.log_compression = log_compression
        # self.skip_connections = skip_connections
        # self.residual_scale = math.sqrt(residual_scale)

    def block(self, n_in, n_out, k, stride):
        return nn.Sequential(
            nn.Conv1d(n_in, n_out, k, stride=stride, bias=False),
            nn.InstanceNorm1d(n_out),  # nn.GroupNorm(1, n_out), in wav2spk replace group by instance normalization
            nn.ReLU(),
        )

    def forward(self, x):
        # BxT -> BxCxT
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 4:
            x = x.squeeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        if self.log_compression:
            x = x.abs()
            x = x + 1
            x = x.log()

        tmp_gate = self.tmp_gate(x.transpose(1, 2)).transpose(1, 2)
        x = x * tmp_gate
        return x.transpose(1, 2)

    # 20210604
    # Epoch 40: Train Accuracy: 99.999839%, Avg loss: 0.036217.
    #           Valid Accuracy: 99.502488%, Avg loss: 0.078200.
    #           Train EER: 2.4206%, Threshold: 0.2692, mindcf-0.01: 0.3075, mindcf-0.001: 0.5515.
    #           Test  ERR: 7.2534%, Threshold: 0.2015, mindcf-0.01: 0.6058, mindcf-0.001: 0.7008.
