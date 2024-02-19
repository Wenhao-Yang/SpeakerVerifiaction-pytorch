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
import pdb

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from python_speech_features import hz2mel, mel2hz
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from scipy import interpolate
from Define_Model.ParallelBlocks import gumbel_softmax
from Define_Model.Pooling import SelfAttentionPooling_v2
from Process_Data.audio_processing import MelSpectrogram
from torchaudio.transforms import Spectrogram
import torchvision
import Process_Data.constants as c


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
        new_centers = self.frequency_center.expand(
            self.num_filter, self.input_dim).clamp(min=0, max=self.sr / 2)
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
        bandwidth_right = self.bandwidth_right.clamp(
            min=1e-12, max=self.sr / 2)
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
        bandwidth_high = (bandwidth_low + self.bandwidth.clamp(min=1e-12,
                          max=self.sr / 2)).clamp_max(self.sr / 2)

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


class MelFbankLayer(nn.Module):
    def __init__(self, sr, num_filter, stretch_ratio=[1.0], log_scale=True):
        super(MelFbankLayer, self).__init__()
        self.num_filter = num_filter
        self.sr = sr
        self.stretch_ratio = stretch_ratio
        self.t = MelSpectrogram(n_fft=512, stretch_ratio=stretch_ratio,
                                win_length=int(0.025 * sr), hop_length=int(0.01 * sr),
                                window_fn=torch.hamming_window, n_mels=num_filter)

    def forward(self, input):
        output = self.t(input.squeeze(1))

        output = torch.transpose(output, -2, -1)
        return torch.log10(output + 1e-6)

    def __repr__(self):
        return "MelFbankLayer(sr={}, num_filter={}, stretch_ratio={})".format(self.sr, self.num_filter, '/'.join(['{:4>.2f}'.format(i) for i in self.stretch_ratio]))


class SpectrogramLayer(nn.Module):
    def __init__(self, sr, n_fft=512, stretch_ratio=[1.0], log_scale=True,
                 win_length=None, hop_length=None, f_max=None, f_min: float = 0.0,
                 pad=0, n_mels=80, init_weight='mel',
                 window_fn=torch.hann_window, power: float = 2.0, normalized: bool = False,
                 center: bool = True, pad_mode: str = "reflect",
                 onesided: bool = True, norm=None, wkwargs=None,):

        super(SpectrogramLayer, self).__init__()
        self.sr = sr
        self.stretch_ratio = stretch_ratio

        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else int(
            0.025 * sr)
        self.hop_length = hop_length if hop_length is not None else int(
            0.01 * sr)
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.init_weight = init_weight
        self.n_mels = n_mels  # number of mel frequency bins
        self.f_max = f_max
        self.f_min = f_min

        self.spectrogram = Spectrogram(n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length,
                                       pad=self.pad, window_fn=window_fn, power=power, normalized=self.normalized,
                                       wkwargs=wkwargs, center=center, pad_mode=pad_mode,
                                       onesided=onesided,)

    def forward(self, input):

        specgram = self.spectrogram(input.squeeze(1))

        stretch_ratio = np.random.choice(self.stretch_ratio)
        if stretch_ratio != 1.0 and self.training:
            specgram = self.stretch(specgram, stretch_ratio)

        return torch.log10(specgram + 1e-6).transpose(-1, -2)

    def __repr__(self):
        return "SpectrogramLayer(sr={}, stretch_ratio={})".format(self.sr, '/'.join(['{:4>.2f}'.format(i) for i in self.stretch_ratio]))


class SparseFbankLayer(nn.Module):
    def __init__(self, sr, num_filter, n_fft=512, stretch_ratio=[1.0], log_scale=True,
                 win_length=None, hop_length=None, f_max=None, f_min: float = 0.0,
                 pad=0, n_mels=80, init_weight='mel',
                 window_fn=torch.hann_window, power: float = 2.0, normalized: bool = False,
                 center: bool = True, pad_mode: str = "reflect",
                 onesided: bool = True, norm=None, wkwargs=None,):

        super(SparseFbankLayer, self).__init__()
        self.num_filter = num_filter
        self.sr = sr
        self.stretch_ratio = stretch_ratio

        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else int(0.025 * sr)
        self.hop_length = hop_length if hop_length is not None else int(0.01 * sr)
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.init_weight = init_weight
        self.n_mels = n_mels  # number of mel frequency bins
        self.f_max = f_max
        self.f_min = f_min

        self.spectrogram = Spectrogram(n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length,
                                       pad=self.pad, window_fn=window_fn, power=power, normalized=self.normalized,
                                       wkwargs=wkwargs, center=center, pad_mode=pad_mode,
                                       onesided=onesided,)
        # init mel
        if init_weight == 'mel':
            all_freqs = torch.linspace(f_min, int(self.sr/2), n_fft // 2 + 1)
            m_pts = torch.linspace(hz2mel(f_min), hz2mel(int(self.sr/2)), num_filter + 2)
            f_pts = mel2hz(m_pts)

            f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
            slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)
            # create overlapping triangles
            zero = torch.zeros(1)
            down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
            up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
            sparsebank = torch.max(zero, torch.min(down_slopes, up_slopes))

        elif init_weight == 'linear':
            all_freqs = torch.linspace(f_min, int(self.sr/2), n_fft // 2 + 1)
            m_pts = torch.linspace(f_min, int(self.sr/2), num_filter+2)

            f_diff = m_pts[1:] - m_pts[:-1]  # (n_filter + 1)
            # (n_freqs, n_filter + 2)
            slopes = m_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
            # create overlapping triangles
            zero = torch.zeros(1)
            down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
            up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
            sparsebank = torch.min(down_slopes, up_slopes).clamp_min(0)
            sparsebank = torch.where(sparsebank > 0, 1, 0)

        elif init_weight == 'rand':
            sparsebank = torch.randn(n_fft // 2 + 1, num_filter)

        self.SpareFbank = nn.Parameter(sparsebank)

    def forward(self, input):

        specgram = self.spectrogram(input.squeeze(1))

        stretch_ratio = np.random.choice(self.stretch_ratio)
        if stretch_ratio != 1.0 and self.training:
            specgram = self.stretch(specgram, stretch_ratio)

        # print(specgram.shape)
        # specgram = specgram.pow(2).sum(-1)
        # normalize
        weight = self.SpareFbank.data
        self.SpareFbank.data = (weight / weight.norm(p=2, dim=0).reshape(1,-1)).abs()

        output = torch.transpose(specgram.squeeze(1), 1, 2)
        # print(output.shape, self.SpareFbank.shape)
        output = torch.matmul(output, self.SpareFbank)

        return torch.log10(output.unsqueeze(1) + 1e-6)

    def __repr__(self):
        return "SparseFbankLayer(sr={}, num_filter={}, stretch_ratio={}, init_weight={},)".format(self.sr, self.num_filter, '/'.join(['{:4>.2f}'.format(i) for i in self.stretch_ratio]), self.init_weight)


class Identity(nn.Module):
    def __init__(self, **kwargs):
        super(Identity, self).__init__()
        self.identity = nn.Identity()
    def forward(self, x, length):
        return self.identity(x)
    
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
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (
                in_channels)
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
        self.window_ = 0.54 - 0.46 * \
            torch.cos(2 * math.pi * n_lin / self.kernel_size)

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
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),
                           self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (
            self.n_ / 2)) * self.window_  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])

        # 时域滤波器
        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1).abs()


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
        output = input.squeeze(1).transpose(-1, -2)
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


class SlideMean_Norm(nn.Module):
    def __init__(self, dim=-2, win_len=300):
        super(SlideMean_Norm, self).__init__()
        self.dim = dim
        self.win_len = win_len

    def forward(self, x):
        indexs = torch.arange(0, x.shape[self.dim])
        start = (indexs - int(self.win_len / 2)).clamp_min(0)
        end = (indexs + int(self.win_len / 2)).clamp_max(x.shape[self.dim] - 1)

        x_mean = []
        for i in range(x.shape[self.dim]):
            s = start[i]
            e = end[i]
            x_mean.append(torch.mean(
                x[:, :, s:e, :], dim=self.dim, keepdim=True))

        return x - torch.cat(x_mean, dim=self.dim)

    def __repr__(self):
        return "SlideMean_Norm(dim=%d)" % self.dim


class MeanStd_Norm(nn.Module):
    def __init__(self, dim=-2):
        super(MeanStd_Norm, self).__init__()
        self.dim = dim

    def forward(self, x):
        return (x - torch.mean(x, dim=self.dim, keepdim=True)) / torch.std(x, dim=self.dim, keepdim=True)

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
        if not self.training or torch.Tensor(1).uniform_(0, 1) < 0.5:
            return x

        # assert self.mask_len < x.shape[-2]
        this_len = np.random.randint(low=0, high=self.mask_len)
        start = np.random.randint(0, x.shape[-2] - this_len)
        x_shape = len(x.shape)

        this_mean = x.mean(dim=-1, keepdim=True).repeat(1,1,1,x_shape[3])
        
        if x_shape == 4:
            x[:, :, start:(start + this_len), :] = this_mean[:, :, start:(start + this_len), :]
            
        elif x_shape == 3:
            x[:, start:(start + this_len), :] = this_mean[:, start:(start + this_len), :]

        return x

    def __repr__(self):
        return "TimeMaskLayer(mask_len=%f)" % self.mask_len


class FreqMaskLayer(nn.Module):
    def __init__(self, mask_len=5, normalized=False):
        super(FreqMaskLayer, self).__init__()
        self.mask_len = mask_len
        self.normalized = normalized

    def forward(self, x):
        if not self.training or torch.Tensor(1).uniform_(0, 1) < 0.5:
            return x

       # assert self.mask_len < x.shape[-1]
        this_len = np.random.randint(low=1, high=self.mask_len+1)
        start    = np.random.randint(0, x.shape[-1] - this_len)
        x_shape = x.shape

        weight = torch.ones(x_shape[-1])
        weight[start:(start + this_len)] = 0 
        weight /= weight.mean()

        if x.is_cuda:
            weight = weight.cuda()

        return x * weight

        # this_mean = x.mean(dim=-2, keepdim=True).repeat(1,1,x_shape[2],1)  # .add(1e-6)
        # if len(x_shape) == 4:
        #     x[:, :, :, start:(start + this_len)] = this_mean[:, :, :, start:(start + this_len)]
        # elif len(x_shape) == 3:
        #     x[:, :, start:(start + this_len)] = this_mean[:, :, start:(start + this_len)]
        # return x

    def __repr__(self):
        return "FreqMaskLayer(mask_len=%f)" % self.mask_len


class FreqMaskIndexLayer(nn.Module):
    def __init__(self, start=0, mask_len=2, normalized=False,
                 mask_type='specaug', mask_value=None):
        super(FreqMaskIndexLayer, self).__init__()
        self.start = start
        self.mask_len = mask_len
        self.normalized = normalized
        self.mask_type = mask_type
        
        self.mask_value = mask_value
        if mask_type == 'const':
            assert mask_value != None
        elif mask_type == 'blur':
            self.mask_value = torchvision.transforms.GaussianBlur(kernel_size=(1,5), sigma=3)
            
    def forward(self, x):
        # x [batch, time, frequency]
        x_shape = len(x.shape)
        # print(x.shape)

        if self.mask_type == 'specaug':
            this_mean = x.mean(dim=-2, keepdim=True)  # .add(1e-6)
        elif self.mask_type == 'zero':
            this_mean = x.mean(dim=-2, keepdim=True) * 0  # .add(1e-6)
        elif self.mask_type == 'const':
            this_mean = self.mask_value 
        
        start = self.start
        end = start + self.mask_len

        if self.mask_type != 'blur':
            if x_shape == 4:
                this_mean = this_mean.repeat(1,1,x.shape[2],1)
                x[:, :, :, start:end] = this_mean[:, :, start:end]
            elif x_shape == 3:
                this_mean = this_mean.repeat(1,x.shape[1],1)
                x[:, :, start:end] = this_mean[:, :, start:end]
        else:
            if x_shape == 4:
                x[:, :, :, start:end] = self.mask_value(x[:, :, start:end])
            elif x_shape == 3:
                x[:, :, start:end] = self.mask_value(x[:, :, start:end])

        return x

    def __repr__(self):
        return "FreqMaskIndexLayer(start=%d, mask_len=%d, mask_type=%s)" % (self.start,
                                                                            self.mask_len,
                                                                            self.mask_type)


class TimeFreqMaskLayer(nn.Module):
    def __init__(self, mask_len=[5, 10], normalized=True):
        super(TimeFreqMaskLayer, self).__init__()
        self.mask_len = mask_len
        self.normalized = normalized

    def forward(self, x):
        if not self.training or torch.Tensor(1).uniform_(0, 1) < 0.5:
            return x

        # assert self.mask_len < x.shape[-2]

        this_len = np.random.randint(low=0, high=self.mask_len[0])
        start = np.random.randint(0, x.shape[-2] - this_len)
        x_shape = len(x.shape)

        time_mean = x.mean(dim=-2, keepdim=True)
        freq_mean = x.mean(dim=-1, keepdim=True)

        if x_shape == 4:
            x[:, :, start:(start + this_len), :] = time_mean
        elif x_shape == 3:
            x[:, start:(start + this_len), :] = time_mean

        this_len = np.random.randint(low=0, high=self.mask_len[1])
        start = np.random.randint(0, x.shape[-1] - this_len)

        if x_shape == 4:
            x[:, :, :, start:(start + this_len)] = freq_mean
        elif x_shape == 3:
            x[:, :, start:(start + this_len)] = freq_mean

        return x

    def __repr__(self):
        return "TimeFreqMaskLayer(mask_len=%s)" % str(self.mask_len)


class SpecAugmentLayer(nn.Module):
    """Implement specaugment for acoustics features' augmentation but without time wraping.
    Reference: Park, D. S., Chan, W., Zhang, Y., Chiu, C.-C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019). 
               Specaugment: A simple data augmentation method for automatic speech recognition. arXiv 
               preprint arXiv:1904.08779.

    Likes in Compute Vision: 
           [1] DeVries, T., & Taylor, G. W. (2017). Improved regularization of convolutional neural networks 
               with cutout. arXiv preprint arXiv:1708.04552.

           [2] Zhong, Z., Zheng, L., Kang, G., Li, S., & Yang, Y. (2017). Random erasing data augmentation. 
               arXiv preprint arXiv:1708.04896. 
    """

    def __init__(self, frequency=0.1, frame=0.1, rows=1, cols=1, 
                 random_rows=False, random_cols=False):
        super(SpecAugmentLayer, self).__init__()
        assert 0. <= frequency < 1.
        assert 0. <= frame < 1.  # a.k.a time axis.

        self.p_f = frequency
        self.p_t = frame

        # Multi-mask.
        self.rows = rows  # Mask rows times for frequency.
        self.cols = cols  # Mask cols times for frame.

        self.random_rows = random_rows
        self.random_cols = random_cols

        self.init = False

    def forward(self, inputs):
        # def __call__(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a matrix), including [batch, time, frenquency]
        """

        input_squeeze = False
        if len(inputs.shape) == 4:
            inputs = inputs.squeeze(1)
            input_squeeze = True

        ones_prob = (torch.ones(inputs.shape[0],1,1).uniform_(0, 1) < 0.8).float()
        ones_prob = ones_prob.to(inputs.device)

        mask = torch.ones_like(inputs)
        mask = mask.to(inputs.device)

        if self.p_f > 0. and self.p_t > 0.:

            if not self.init:
                input_size = inputs.shape
                # assert len(input_size) == 2
                if self.p_f > 0.:
                    self.num_f = input_size[-1]  # Total channels.
                    # Max channels to drop.
                    self.F = int(self.num_f * self.p_f)
                if self.p_t > 0.:
                    # Total frames. It requires all egs with the same frames.
                    self.num_t = input_size[-2]
                    self.T = int(self.num_t * self.p_t)  # Max frames to drop.
                self.init = True

            if self.p_f > 0.:
                if self.random_rows:
                    multi = np.random.randint(1, self.rows+1)
                else:
                    multi = self.rows

                for i in range(multi):
                    f = np.random.randint(0, self.F + 1)
                    f_0 = np.random.randint(0, self.num_f - f + 1)

                    inverted_factor = self.num_f / (self.num_f - f)
                    # print('freq', f_0, f_0+f)
                    
                    mask[:, :, f_0:f_0+f].fill_(0.)
                    mask.mul_(inverted_factor)

            if self.p_t > 0.:
                if self.random_cols:
                    multi = np.random.randint(1, self.cols+1)
                else:
                    multi = self.cols

                for i in range(multi):
                    t = np.random.randint(0, self.T + 1)
                    t_0 = np.random.randint(0, self.num_t - t + 1)
                    
                    # print('time', t_0, t_0+t)

                    mask[:, t_0:t_0+t, :].fill_(0.)
        
        mask = (1-mask) * ones_prob + 1
        inputs = inputs * mask

        if input_squeeze:
            inputs = inputs.unsqueeze(1)

        return inputs
    
    def __repr__(self):
        return "SpecAugmentLayer(p_freq={}, p_time={}, rows={}, cols={})".format(self.p_f, self.p_t,
                                                                             self.rows, self.cols)


def get_weight(weight: str, input_dim: int, power_weight: str):

    m = np.arange(0, 2840.0230467083188)
    m = 700 * (10 ** (m / 2595.0) - 1)
    n = np.array([1/(m[i] - m[i - 1]) for i in range(1, len(m))])
    # x = np.arange(input_dim) * 8000 / (input_dim - 1)  # [0-8000]
    f = interpolate.interp1d(m[1:], n)
    xnew = np.arange(np.min(m[1:]), np.max(
        m[1:]), (np.max(m[1:]) - np.min(m[1:])) / input_dim)
    mel = f(xnew)
    amel = 1/f(xnew)

    weights = {
        "mel": mel,
        "amel": amel,
        "rand": np.random.uniform(size=input_dim),
        "one": np.ones(input_dim),
        "clean": c.VOX1_CLEAN,
        "rclean": c.VOX1_RCLEAN,
        "rclean_max": c.VOX1_RCLEAN_MAX,
        "vox2_rclean": c.VOX2_RCLEAN,
        "aug": c.VOX1_AUG,
        "vox2": c.VOX2_CLEAN,
        "vox1_cf": c.VOX1_CFB40,
        "vox2_cf": c.VOX2_CFB40,
        "vox1_rcf": c.VOX1_RCFB40,
        "vox2_rcf": c.VOX2_RCFB40,
        "v2_rclean_gean": c.VOX2_RCLEAN_GRAD_MEAN,
        "v2_rclean_iean": c.VOX2_RCLEAN_INPT_MEAN,
        "v2_rclean_igean": c.VOX2_RCLEAN_INGR_MEAN,
        "v2_rclean_gax": c.VOX2_RCLEAN_GRAD_MAX,
        "v2_rclean_imax": c.VOX2_RCLEAN_INPT_MAX,
        "v2_rclean_igmax": c.VOX2_RCLEAN_INGR_MAX,
        "v2_fratio": c.VOX2_FRATIO,
        "v2_eer": c.V2_EER,
        "v2_frl": c.FRL_V2,
        "v2_intems": c.INTE_MEAN_STD_v2
    }

    assert weight in weights.keys()
    ynew = weights[weight]

    if len(ynew) != input_dim:
        x = np.arange(0, 8000+8000/(len(ynew)-1), 8000/(len(ynew)-1))
        f = interpolate.interp1d(x, ynew)

        xnew = np.arange(0, 8000+8000/(input_dim-1), 8000/(input_dim-1))
        ynew = f(xnew)

    ynew = np.array(ynew)
    if 'power' in power_weight:
        ynew = np.power(ynew, 2)

    if 'mean' in power_weight:
        ynew /= ynew.mean()
    elif 'norm' in power_weight:
        ynew = (ynew - ynew.min()) / (ynew.max() - ynew.min())
    else:
        ynew /= ynew.max()

    return ynew


class DropweightLayer(nn.Module):
    def __init__(self, dropout_p=0.1, weight='mel', input_dim=161, scale=0.2,
                 power_weight='none'):
        super(DropweightLayer, self).__init__()
        self.input_dim = input_dim
        self.weight = weight
        self.dropout_p = dropout_p
        self.scale = scale

        ynew = get_weight(weight, input_dim, power_weight)
        self.drop_p = ynew * self.scale + 1-self.scale - dropout_p

    def forward(self, x):
        if not self.training or torch.Tensor(1).uniform_(0, 1) < 0.5:
            return x
        else:
            assert len(
                self.drop_p) == x.shape[-1], print(len(self.drop_p), x.shape)
            drop_weight = []
            for i in self.drop_p:
                drop_weight.append(
                    (torch.Tensor(1).uniform_(0, 1) < i).float())

            if len(x.shape) == 4:
                drop_weight = torch.tensor(drop_weight).reshape(1, 1, 1, -1)
            else:
                drop_weight = torch.tensor(drop_weight).reshape(1, 1, -1)

            if x.is_cuda:
                drop_weight = drop_weight.cuda()

            return x * drop_weight

    def __repr__(self):
        return "DropweightLayer(input_dim=%d, weight=%s, dropout_p=%s, scale=%f)" % (self.input_dim, self.weight,
                                                                                     self.dropout_p, self.scale)


class DropweightLayer_v2(nn.Module):
    def __init__(self, dropout_p=0.1, weight='mel', input_dim=161,
                 scale=0.2, power_weight='mean'):
        super(DropweightLayer_v2, self).__init__()
        self.input_dim = input_dim
        self.weight = weight
        self.dropout_p = dropout_p
        self.scale = scale
        ynew = get_weight(weight, input_dim, power_weight)

        self.drop_p = ynew * self.scale + 1-self.scale - dropout_p

    def forward(self, x):
        if not self.training:
            return x
        else:
            assert len(
                self.drop_p) == x.shape[-1], print(len(self.drop_p), x.shape)
            drop_weight = []
            for i in self.drop_p:
                drop_weight.append(
                    (torch.ones(x.shape[-2]).uniform_(0, 1) < i).float())
            if len(x.shape) == 4:
                drop_weight = torch.stack(drop_weight, dim=0).reshape(
                    1, 1, x.shape[-2], x.shape[-1])
            else:
                drop_weight = torch.stack(drop_weight, dim=0).reshape(
                    1, x.shape[-2], x.shape[-1])

            if x.is_cuda:
                drop_weight = drop_weight.cuda()

            return x * drop_weight

    def __repr__(self):

        return "DropweightLayer_v2(input_dim=%d, weight=%s, dropout_p==%s, scale=%f)" % (self.input_dim, self.weight,
                                                                                         self.dropout_p, self.scale)


class AttentionweightLayer(nn.Module):
    def __init__(self, input_dim=161, weight='mel', weight_norm='none'):
        super(AttentionweightLayer, self).__init__()
        self.input_dim = input_dim
        self.weight = weight
        self.weight_norm = weight_norm

        self.w = nn.Parameter(torch.tensor(2.0))
        self.b = nn.Parameter(torch.tensor(-1.0))
        self.drop_p = get_weight(weight, input_dim, weight_norm)
        # self.activation = nn.Tanh()
        # self.activation = nn.Softmax(dim=-1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # assert len(self.drop_p) == x.shape[-1], print(len(self.drop_p), x.shape)
        if len(x.shape) == 4:
            drop_weight = torch.tensor(
                self.drop_p).reshape(1, 1, 1, -1).float()
        else:
            drop_weight = torch.tensor(self.drop_p).reshape(1, 1, -1).float()

        if x.is_cuda:
            drop_weight = drop_weight.cuda()

        drop_weight = self.w * drop_weight + self.b
        drop_weight = self.activation(drop_weight)

        return x * drop_weight

    def __repr__(self):
        return "AttentionweightLayer_v0(input_dim=%d, weight=%s, weight_norm=%s)" % (
            self.input_dim, self.weight, self.weight_norm)


class ReweightLayer(nn.Module):
    def __init__(self, input_dim=161, weight='v1_f2m'):
        super(ReweightLayer, self).__init__()
        self.input_dim = input_dim
        self.weight = weight

        if weight == 'v1_f2m':
            ynew = np.array(c.VOX1_F2M)
        elif weight == 'v2_stu1':
            ynew = np.array(c.INTE_STUDENT1)
        else:
            raise ValueError(weight)

        self.weight = nn.Parameter(torch.tensor(ynew), requires_grad=False)
        # self.drop_p = ynew  # * dropout_p

        # self.activation = nn.Tanh()
        # self.activation = nn.Softmax(dim=-1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        assert len(
            self.weight) == x.shape[-1], print(len(self.weight), x.shape)

        if len(x.shape) == 4:
            weight = self.weight.reshape(1, 1, 1, -1).float()
        else:
            weight = self.weight.reshape(1, 1, -1).float()

        return x * weight

    def __repr__(self):
        return "ReweightLayer(input_dim=%d, weight=%s)" % (self.input_dim, self.weight)


class RedropLayer(nn.Module):
    def __init__(self, input_dim=80, mask_len=5, weight='v2_v12base1'):
        super(RedropLayer, self).__init__()
        self.input_dim = input_dim
        self.weight = weight

        if weight == 'v2_stbase1':
            ynew = np.array(c.INTE_STBASE1)
        elif weight == 'v2_v12base1':
            ynew = np.array(c.INTE_V12BASE)
        else:
            raise ValueError(weight)

        # self.weight = ynew
        # idx_number = np.ceil(np.clip(ynew, a_max=None, a_min=1) * 10)

        idx_number = np.ceil(ynew/ynew.min() * 2)
        choice_size = int(np.ceil(idx_number.sum() * mask_len / input_dim))

        all_idxs = []
        for i in range(len(idx_number)):
            all_idxs.extend([i]*int(idx_number[i]))

        self.all_idxs = all_idxs
        self.choice_size = choice_size
        self.mask_len = mask_len
    
    def forward(self, x):
        # assert len(
            # self.weight) == x.shape[-1], print(len(self.weight), x.shape)
        if not self.training or torch.Tensor(1).uniform_(0, 1) < 0.5:
            return x
        
        this_len = np.random.randint(low=1, high=self.mask_len+1)
        start    = np.random.choice(self.all_idxs)

        while start + this_len > self.input_dim:
            start    = np.random.choice(self.all_idxs)

        # normal speechaug
        # x_norm = torch.normal(mean=x.mean(dim=[2], keepdim=True).repeat(1,1,x.shape[-2],1),
        #                       std=x.std(dim=[2], keepdim=True)).detach()
        # x[:, :, :, start:(start+this_len)] = x_norm[:, :, :, start:(start+this_len)]

        # return x
        
        # zero speechaug
        weight = torch.ones(self.input_dim)
        weight[start:(start+this_len)] = 0 
        weight /= weight.mean()

        if x.is_cuda:
            weight = weight.cuda()

        return x * weight

    def __repr__(self):
        return "RedropLayer(input_dim=%d, weight=%s)" % (self.input_dim, self.weight)


# On The Importance Of Different Frequency Bins For speaker verification
class FrequencyReweightLayer(nn.Module):
    def __init__(self, input_dim=161):
        super(FrequencyReweightLayer, self).__init__()
        self.input_dim = input_dim

        self.weight = nn.Parameter(torch.ones(1, 1, 1, input_dim))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """sumary_line

        Keyword arguments:
        X -- batch, channel, time, frequency
        Return: x + U
        """
        # assert self.weight.shape[-1] == x.shape[-1], print(self.weight.shape, x.shape)
        f = 1 + self.activation(self.weight)

        return x * f

    def __repr__(self):
        return "FrequencyReweightLayer(input_dim=%d)" % (self.input_dim)


class FrequencyDecayReweightLayer(nn.Module):
    def __init__(self, input_dim=161):
        super(FrequencyDecayReweightLayer, self).__init__()
        self.input_dim = input_dim
        
        self.decay_std = nn.Parameter(torch.ones(1, 1, 1, input_dim))
        # self.decay_classifier = nn.Sequential(
        #     nn.Linear(input_dim, input_dim),
        # )
        self.theta = 1.0
        # self.theta = nn.Parameter(torch.ones(1, 1, 1, input_dim))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """sumary_line
        
        Keyword arguments:
        X -- batch, channel, time, frequency
        Return: x + U
        """
        # assert self.weight.shape[-1] == x.shape[-1], print(self.weight.shape, x.shape)
        freq_std   = x.std(dim=-2, keepdim=True)
        freq_score = (self.decay_std - freq_std) / self.theta
        # .clamp_min(0.25)
        # freq_score   = self.decay_classifier(freq_score)
        # freq_score = torch.nn.functional.tanhshrink(freq_score).clamp_max(1)
        # f = freq_score.exp()
        f = 2 * self.activation(freq_score)

        return x * f

    def __repr__(self):
        return "FrequencyDecayReweightLayer(input_dim=%d)" % (self.input_dim)
    

class FrequencyDecayReweightLayer2(nn.Module):
    def __init__(self, input_dim=161):
        super(FrequencyDecayReweightLayer2, self).__init__()
        self.input_dim = input_dim
        
        self.decay_classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
        )

        self.decay_std = nn.Parameter(torch.ones(1, 1, 1, input_dim))
        self.theta = 1.0
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """sumary_line
        
        Keyword arguments:
        X -- batch, channel, time, frequency
        Return: x + U
        """
        # assert self.weight.shape[-1] == x.shape[-1], print(self.weight.shape, x.shape)
        freq_std   = x.std(dim=-2, keepdim=True)
        freq_std   = self.decay_classifier(freq_std)
        freq_score = (self.decay_std - freq_std) / self.theta

        f = 0.5 + self.activation(freq_score)
        # f = f 
        return x * f

    def __repr__(self):
        return "FrequencyDecayReweightLayer2(input_dim=%d)" % (self.input_dim)
    

class FrequencyDecayReweightLayer3(nn.Module):
    def __init__(self, input_dim=161):
        super(FrequencyDecayReweightLayer3, self).__init__()
        self.input_dim = input_dim
        
        self.decay_std = nn.Parameter(torch.ones(1, 1, 1, input_dim))
        self.theta = 0.5
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """sumary_line
        
        Keyword arguments:
        X -- batch, channel, time, frequency
        Return: x + U
        """
        # assert self.weight.shape[-1] == x.shape[-1], print(self.weight.shape, x.shape)
        freq_std   = x.std(dim=-2, keepdim=True)
        freq_score = (self.decay_std - freq_std) / self.theta
        freq_score = torch.nn.functional.tanhshrink(freq_score)
        # torch.clamp_max(freq_score, )

        f = 0.5 + self.activation(freq_score)

        return x * f

    def __repr__(self):
        return "FrequencyDecayReweightLayer3(input_dim=%d)" % (self.input_dim)
    


class FrequencyReweightLayer2(nn.Module):
    def __init__(self, input_dim=161):
        super(FrequencyReweightLayer2, self).__init__()
        self.input_dim = input_dim

        self.weight = nn.Parameter(torch.ones(1, 1, 1, input_dim))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """sumary_line
        
        Keyword arguments:
        X -- batch, channel, time, frequency
        Return: x + U
        """
        # assert self.weight.shape[-1] == x.shape[-1], print(self.weight.shape, x.shape)
        f = 0.5 + self.activation(self.weight)
        
        return x * f

    def __repr__(self):
        return "FrequencyReweightLayer2(input_dim=%d)" % (self.input_dim)
    

class FrequencyGenderReweightLayer2(nn.Module):
    def __init__(self, input_dim=161):
        super(FrequencyGenderReweightLayer2, self).__init__()
        self.input_dim = input_dim

        self.weight = nn.Parameter(torch.ones(1, 1, 2, input_dim))
        self.gender_classifier = nn.Linear(input_dim, 2)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """sumary_line
        
        Keyword arguments:
        X -- batch, channel, time, frequency
        Return: x + U
        """
        # assert self.weight.shape[-1] == x.shape[-1], print(self.weight.shape, x.shape)
        freq_std = x.std(dim=-2)
        if freq_std.shape[1] != 1:
            freq_std = freq_std.mean(dim=1, keepdim=True)

        gender_score = self.gender_classifier(freq_std.squeeze(1))
        soft_gender_score = F.softmax(gender_score, dim=1).unsqueeze(1).unsqueeze(3)
        
        f = self.weight * soft_gender_score
        f = f.sum(dim=2, keepdim=True)
        
        f = 0.5 + self.activation(f)

        if self.return_logits:
            return x * f, gender_score
        else:
            return x * f

    def __repr__(self):
        return "FrequencyGenderReweightLayer2(input_dim=%d)" % (self.input_dim)
    
class FrequencyGenderReweightLayer3(nn.Module):
    def __init__(self, input_dim=161):
        super(FrequencyGenderReweightLayer3, self).__init__()
        self.input_dim = input_dim
        # {'f': 0, 'm': 1}
        female_w = torch.FloatTensor(c.INTE_FEMALE)
        male_w = torch.FloatTensor(c.INTE_MALE)
        
        weight = torch.stack([female_w, male_w]).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(weight, requires_grad=False)
        
        self.gender_classifier = nn.Linear(input_dim, 2)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """sumary_line
        
        Keyword arguments:
        X -- batch, channel, time, frequency
        Return: x + U
        """
        # assert self.weight.shape[-1] == x.shape[-1], print(self.weight.shape, x.shape)
        freq_std     = x.std(dim=-2)
        gender_score = self.gender_classifier(freq_std.squeeze(1))
        gender_score = F.softmax(gender_score, dim=1).unsqueeze(1).unsqueeze(3)
        
        f = 0.25 + self.activation(self.weight)
        # linear inteplolation
        f = gender_score * f
        f = f.mean(dim=2, keepdim=True)
        
        # max selection
        # gender_index = torch.max(gender_score, dim=1)[1]
        # f = f[:,:,gender_index].transpose(0,2)
        
        return x * f

    def __repr__(self):
        return "FrequencyGenderReweightLayer3(input_dim=%d)" % (self.input_dim)
    

class FrequencyGenderReweightLayer4(nn.Module):
    def __init__(self, input_dim=161, bias=0.25):
        super(FrequencyGenderReweightLayer4, self).__init__()
        self.input_dim = input_dim
        self.bias = bias
        # {'f': 0, 'm': 1}
        female_w = torch.FloatTensor(c.INTE_FEMALE)
        male_w = torch.FloatTensor(c.INTE_MALE)
        
        weight = torch.stack([female_w, male_w]).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(weight, requires_grad=False)
        
        self.gender_classifier = nn.Linear(input_dim, 2)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """sumary_line
        
        Keyword arguments:
        X -- batch, channel, time, frequency
        Return: x + U
        """
        # assert self.weight.shape[-1] == x.shape[-1], print(self.weight.shape, x.shape)
        freq_std     = x.std(dim=-2)
        if freq_std.shape[1] == 1:
            freq_std = freq_std.squeeze(1)
        else:
            freq_std = freq_std.mean(dim=1)

        gender_score = self.gender_classifier(freq_std)
        gender_score = F.softmax(gender_score, dim=1)
        
        f = self.bias + self.activation(self.weight)
        # f = self.activation(self.weight)
        # semi-hard inteplolation
        gender_index = torch.max(gender_score, dim=1)[1]
        gender_index = torch.nn.functional.one_hot(gender_index, num_classes=2).float()
        gender_score = (gender_score + gender_index).unsqueeze(1).unsqueeze(3)
        
        f = gender_score/2 * f
        f = f.mean(dim=2, keepdim=True)
        # f =  f / self.activation(freq_std.unsqueeze(1).unsqueeze(1))
        # f = f / f.mean(dim=3, keepdim=True)
        
        return x * f

    def __repr__(self):
        return "FrequencyGenderReweightLayer4(input_dim=%d, bias=%f)" % (self.input_dim, self.bias)


class FrequencyGenderReweightLayer6(nn.Module):
    def __init__(self, input_dim=161):
        super(FrequencyGenderReweightLayer6, self).__init__()
        self.input_dim = input_dim
        # {'f': 0, 'm': 1}
        female_w = torch.FloatTensor(c.INTE_FEMALE)
        male_w   = torch.FloatTensor(c.INTE_MALE)
        
        weight = torch.stack([female_w, male_w]).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(weight, requires_grad=False)
        
        self.encoder = SelfAttentionPooling_v2(input_dim=input_dim,
                                                   hidden_dim=int(input_dim/2))
        
        self.gender_classifier = nn.Linear(input_dim, 2)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """sumary_line
        
        Keyword arguments:
        X -- batch, channel, time, frequency
        Return: x + U
        """
        freq_std     = self.encoder(x)
        gender_score = self.gender_classifier(freq_std)
        gender_score = F.softmax(gender_score, dim=1)
        
        f = 0.25 + self.activation(self.weight)
        # semi-hard inteplolation
        gender_index = torch.max(gender_score, dim=1)[1]
        gender_index = torch.nn.functional.one_hot(gender_index, num_classes=2).float()
        gender_score = (gender_score + gender_index).unsqueeze(1).unsqueeze(3)
        
        f = gender_score/2 * f
        f = f.mean(dim=2, keepdim=True)
        
        return x * f

    def __repr__(self):
        return "FrequencyGenderReweightLayer6(input_dim=%d)" % (self.input_dim)
    

class FrequencyGenderReweightLayer62(nn.Module):
    def __init__(self, input_dim=80):
        super(FrequencyGenderReweightLayer62, self).__init__()
        self.input_dim = input_dim
        # {'f': 0, 'm': 1}
        female_w = torch.FloatTensor(c.INTE_FEMALE)
        male_w   = torch.FloatTensor(c.INTE_MALE)
        
        weight = torch.stack([female_w, male_w]).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(weight)
        
        self.encoder = SelfAttentionPooling_v2(input_dim=input_dim,
                                                   hidden_dim=int(input_dim/2))
        
        self.gender_classifier = nn.Linear(input_dim, 2)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """sumary_line
        
        Keyword arguments:
        X -- batch, channel, time, frequency
        Return: x + U
        """
        freq_std     = self.encoder(x)
        gender_score = self.gender_classifier(freq_std)
        gender_score = gumbel_softmax(gender_score).contiguous().float()
        
        f = 1.0 + self.activation(self.weight)
        # semi-hard inteplolation
        f = gender_score.unsqueeze(1).unsqueeze(3) * f
        f = f.sum(dim=2, keepdim=True)
        
        return x * f

    def __repr__(self):
        return "FrequencyGenderReweightLayer62(input_dim=%d)" % (self.input_dim)


class FrequencyGenderReweightLayer5(nn.Module):
    def __init__(self, input_dim=161):
        super(FrequencyGenderReweightLayer5, self).__init__()
        self.input_dim = input_dim
        # {'f': 0, 'm': 1}
        # female_w = torch.FloatTensor(c.INTE_FEMALE)
        # male_w = torch.FloatTensor(c.INTE_MALE)
        # weight = torch.stack([female_w, male_w]).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(torch.ones(1, 1, 2, input_dim))
        
        self.gender_classifier = nn.Linear(input_dim, 2)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """sumary_line
        
        Keyword arguments:
        X -- batch, channel, time, frequency
        Return: x + U
        """
        # assert self.weight.shape[-1] == x.shape[-1], print(self.weight.shape, x.shape)
        freq_std     = x.std(dim=-2)
        if freq_std.shape[1] == 1:
            freq_std = freq_std.squeeze(1)
        else:
            freq_std = freq_std.mean(dim=1)

        gender_score = self.gender_classifier(freq_std)
        gender_score = F.softmax(gender_score, dim=1)
        
        f = 0.25 + self.activation(self.weight)
        # semi-hard inteplolation
        gender_index = torch.max(gender_score, dim=1)[1]
        gender_index = torch.nn.functional.one_hot(gender_index, num_classes=2).float()
        gender_score = (gender_score + gender_index).unsqueeze(1).unsqueeze(3)
        
        f = gender_score/2 * f
        f = f.mean(dim=2, keepdim=True)
        
        return x * f

    def __repr__(self):
        return "FrequencyGenderReweightLayer5(input_dim=%d)" % (self.input_dim)


class FrequencyGenderReweightLayer7(nn.Module):
    def __init__(self, ckp_path, input_dim=80, fix_param=True):
        super(FrequencyGenderReweightLayer7, self).__init__()
        self.input_dim = input_dim
        # {'f': 0, 'm': 1}
        female_w = torch.FloatTensor(c.INTE_FEMALE)
        male_w   = torch.FloatTensor(c.INTE_MALE)
        weight = torch.stack([female_w, male_w]).unsqueeze(0).unsqueeze(0)

        self.weight = nn.Parameter(weight, requires_grad=False)
        
        self.gender_classifier = torch.load(ckp_path)
        self.activation = nn.Sigmoid()
        
        if fix_param:
            self.fix_params()

    def forward(self, x):
        """sumary_line
        
        Keyword arguments:
        X -- batch, channel, time, frequency
        Return: x + U
        """
        gender_score = self.gender_classifier(x)
        gender_score = F.softmax(gender_score, dim=1).unsqueeze(1).unsqueeze(3)
        
        # semi-hard inteplolation
        # gender_index = torch.max(gender_score, dim=1)[1]
        # gender_index = torch.nn.functional.one_hot(gender_index, num_classes=2).float()
        # gender_score = (gender_score + gender_index).unsqueeze(1).unsqueeze(3)
        
        f = 0.25 + self.activation(self.weight)
        f = gender_score * f
        f = f.mean(dim=2, keepdim=True)
        
        # f = gender_score/2 * self.weight
        # f = 0.5 + self.activation(f)
        
        return x * f

    def fix_params(self):
        for p in self.parameters():
            p.requires_grad = False

    def __repr__(self):
        return "FrequencyGenderReweightLayer7(input_dim=%d)" % (self.input_dim)
    

class FrequencyGenderReweightLayer8(nn.Module):
    def __init__(self, ckp_path, input_dim=80, fix_param=True):
        super(FrequencyGenderReweightLayer8, self).__init__()
        self.input_dim = input_dim
        # {'f': 0, 'm': 1}
        # female_w = torch.FloatTensor(c.INTE_FEMALE)
        # male_w   = torch.FloatTensor(c.INTE_MALE)
        weight = torch.ones(2, input_dim).unsqueeze(0).unsqueeze(0)
        
        self.weight = nn.Parameter(weight, requires_grad=True)
        
        self.gender_classifier = torch.load(ckp_path)
        self.activation = nn.Sigmoid()
        
        if fix_param:
            self.fix_params()

    def forward(self, x):
        """sumary_line
        
        Keyword arguments:
        X -- batch, channel, time, frequency
        Return: x + U
        """
        gender_score = self.gender_classifier(x)
        gender_score = F.softmax(gender_score, dim=1).unsqueeze(1).unsqueeze(3)
        
        # semi-hard inteplolation
        # gender_index = torch.max(gender_score, dim=1)[1]
        # gender_index = torch.nn.functional.one_hot(gender_index, num_classes=2).float()
        # gender_score = (gender_score + gender_index).unsqueeze(1).unsqueeze(3)
        
        f = 0.5 + self.activation(self.weight)
        f = gender_score * f
        f = f.sum(dim=2, keepdim=True)
        
        # f = gender_score/2 * self.weight
        # f = 0.5 + self.activation(f)
        
        return x * f

    def fix_params(self):
        for p in self.parameters():
            p.requires_grad = False

    def __repr__(self):
        return "FrequencyGenderReweightLayer8(input_dim=%d)" % (self.input_dim)
    

class FrequencyGenderReweightLayer82(nn.Module):
    def __init__(self, ckp_path, input_dim=80, fix_param=True):
        super(FrequencyGenderReweightLayer82, self).__init__()
        self.input_dim = input_dim
        # {'f': 0, 'm': 1}
        # female_w = torch.FloatTensor(c.INTE_FEMALE)
        # male_w   = torch.FloatTensor(c.INTE_MALE)
        weight = torch.ones(2, input_dim).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(weight, requires_grad=True)
        
        self.gender_classifier = torch.load(ckp_path)
        self.activation = nn.Sigmoid()
        
        if fix_param:
            self.fix_params()

    def forward(self, x):
        """sumary_line
        
        Keyword arguments:
        X -- batch, channel, time, frequency
        Return: x + U
        """
        gender_score = self.gender_classifier(x)
        gender_score = gumbel_softmax(gender_score).contiguous().float()
        gender_score = gender_score.unsqueeze(1).unsqueeze(3)
        
        # semi-hard inteplolation
        # gender_index = torch.max(gender_score, dim=1)[1]
        # gender_index = torch.nn.functional.one_hot(gender_index, num_classes=2).float()
        # gender_score = (gender_score + gender_index).unsqueeze(1).unsqueeze(3)
        
        f = 1.0 + self.activation(self.weight)
        f = gender_score * f
        f = f.sum(dim=2, keepdim=True)
        
        # f = gender_score/2 * self.weight
        # f = 0.5 + self.activation(f)
        
        return x * f

    def fix_params(self):
        for p in self.parameters():
            p.requires_grad = False

    def __repr__(self):
        return "FrequencyGenderReweightLayer82(input_dim=%d)" % (self.input_dim)


class FrequencyGenderReweightLayer9(nn.Module):
    def __init__(self, ckp_path, input_dim=80, fix_param=True):
        super(FrequencyGenderReweightLayer9, self).__init__()
        self.input_dim = input_dim
        # {'f': 0, 'm': 1}
        female_w = torch.FloatTensor(c.INTE_FEMALE)
        male_w   = torch.FloatTensor(c.INTE_MALE)
        weight = torch.stack([female_w, male_w]).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(weight, requires_grad=False)
        
        self.vad = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.ReLU6()
        )

        self.gender_classifier = torch.load(ckp_path)
        self.activation = nn.Sigmoid()
        
        if fix_param:
            self.fix_params()

    def forward(self, x):
        """sumary_line
        
        Keyword arguments:
        X -- batch, channel, time, frequency
        Return: x + U
        """
        gender_score = self.gender_classifier(x)
        gender_score = F.softmax(gender_score, dim=1).unsqueeze(1).unsqueeze(3)
        # semi-hard inteplolation
    
        f = gender_score * self.weight
        f_vad = self.vad(x)
        f = 0.5 + self.activation(f_vad * f.sum(dim=2, keepdim=True))
        
        return x * f

    def fix_params(self):
        for p in self.parameters():
            p.requires_grad = False

    def __repr__(self):
        return "FrequencyGenderReweightLayer9(input_dim=%d)" % (self.input_dim)


class FrequencyNormReweightLayer(nn.Module):
    def __init__(self, input_dim=161):
        super(FrequencyNormReweightLayer, self).__init__()
        self.input_dim = input_dim

        self.weight = nn.Parameter(torch.ones(1, 1, 1, input_dim))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """sumary_line
        
        Keyword arguments:
        X -- batch, channel, time, frequency
        Return: x + U
        """
        # assert self.weight.shape[-1] == x.shape[-1], print(self.weight.shape, x.shape)
        F = 1 + self.activation(self.weight)
        F = F / F.mean()
        
        return x * F

    def __repr__(self):
        return "FrequencyNormReweightLayer(input_dim=%d)" % (self.input_dim)


class FrequencyGenderReweightLayer22(nn.Module):
    def __init__(self, input_dim=80):
        super(FrequencyGenderReweightLayer22, self).__init__()
        self.input_dim = input_dim

        self.key   = nn.Linear(input_dim, 2)
        self.value = nn.Linear(input_dim, 2)
        
        self.avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """sumary_line
        
        Keyword arguments:
        X -- batch, channel, time, frequency
        Return: x + U
        """
        # assert self.weight.shape[-1] == x.shape[-1], print(self.weight.shape, x.shape)
        # freq_std = x.std(dim=-2)
        # if freq_std.shape[1] != 1:
        #     freq_std = freq_std.mean(dim=1, keepdim=True)
        k = self.key(x).squeeze(1)
        v = self.value(x).squeeze(1)
        q = x.squeeze(1)
        
        e = torch.bmm(k, v.permute(0,2,1))
        s = torch.nn.functional.softmax(e, dim=2) / math.sqrt(2)
        s = torch.bmm(s, q) 
        s = self.avg(s) - s.mean(dim=1, keepdim=True)
        f = 0.5 + self.activation(s) #.mean(dim=1, keepdim=True)
        
        return x*f.unsqueeze(1)

        # gender_score = self.gender_classifier(x).mean(dim=2, keepdim=True)
        # gender_score = gender_score.transpose(1,2)

        # soft_gender_score = F.softmax(gender_score, dim=2)#.unsqueeze(1).unsqueeze(3)
        
        # f = self.weight * soft_gender_score#.mean(dim=3, keepdim=True)
        # f = self.avg(f)
        # f = self.activation(f)
        # f = f.sum(dim=2, keepdim=True)
        
        # f = f / f.mean()
        # return x * f #, f
    
        # return xf * x_std / xf_std

    def __repr__(self):
        return "FrequencyGenderReweightLayer22(input_dim=%d)" % (self.input_dim)
    

class FrequencyNormReweightLayer(nn.Module):
    def __init__(self, input_dim=161):
        super(FrequencyNormReweightLayer, self).__init__()
        self.input_dim = input_dim

        self.weight = nn.Parameter(torch.ones(1, 1, 1, input_dim))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """sumary_line
        
        Keyword arguments:
        X -- batch, channel, time, frequency
        Return: x + U
        """
        # assert self.weight.shape[-1] == x.shape[-1], print(self.weight.shape, x.shape)
        F = 1 + self.activation(self.weight)
        F = F / F.mean()
        
        return x * F

    def __repr__(self):
        return "FrequencyNormReweightLayer(input_dim=%d)" % (self.input_dim)

class TimeReweightLayer(nn.Module):
    def __init__(self, input_dim=161):
        super(TimeReweightLayer, self).__init__()
        self.input_dim = input_dim

        self.weight = nn.Parameter(torch.ones(1, 1, 1, input_dim))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """sumary_line
        
        Keyword arguments:
        X -- batch, channel, time, frequency
        Return: x + U
        """
        # assert self.weight.shape[-1] == x.shape[-1], print(self.weight.shape, x.shape)
        F = x * self.activation(self.weight)
        T = self.activation(F.sum(dim=-1, keepdim=True))
        
        return x * ( 1 + T)

    def __repr__(self):
        return "TimeReweightLayer(input_dim=%d)" % (self.input_dim)  

class FreqTimeReweightLayer(nn.Module):
    def __init__(self, input_dim=161):
        super(FreqTimeReweightLayer, self).__init__()
        self.input_dim = input_dim

        self.weight = nn.Parameter(torch.ones(1, 1, 1, input_dim))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """sumary_line
        
        Keyword arguments:
        X -- batch, channel, time, frequency
        Return: x + U
        """
        # assert self.weight.shape[-1] == x.shape[-1], print(self.weight.shape, x.shape)
        F = x * self.activation(self.weight)
        T = x * self.activation(F.sum(dim=-1, keepdim=True))
        
        return x + (F + T) / 2

    def __repr__(self):
        return "FreqTimeReweightLayer(input_dim=%d)" % (self.input_dim)


class AttentionweightLayer_v2(nn.Module):
    def __init__(self, input_dim=161, weight='mel', weight_norm='none'):
        super(AttentionweightLayer_v2, self).__init__()
        self.input_dim = input_dim
        self.weight = weight
        self.weight_norm = weight_norm

        self.w = nn.Parameter(torch.tensor(2.0))
        self.b = nn.Parameter(torch.tensor(-1.0))

        self.drop_p = nn.Parameter(torch.tensor(
            get_weight(weight, input_dim, weight_norm)).float())
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # assert len(self.drop_p) == x.shape[-1], print(len(self.drop_p), x.shape)
        if len(x.shape) == 4:
            drop_weight = self.drop_p.reshape(1, 1, 1, -1).float()
        else:
            drop_weight = self.drop_p.reshape(1, 1, -1).float()
        # if x.is_cuda:
        #     drop_weight = drop_weight.cuda()
        drop_weight = self.w * drop_weight + self.b
        drop_weight = self.activation(drop_weight)

        return x * drop_weight

    def __repr__(self):
        return "AttentionweightLayer_Trainable(input_dim=%d, weight=%s, weight_norm=%s)" % (
            self.input_dim, self.weight, self.weight_norm)


class AttentionweightLayer_v3(nn.Module):
    def __init__(self, weight, input_dim=161,
                 power_weight='norm'):
        super(AttentionweightLayer_v3, self).__init__()
        self.input_dim = input_dim
        self.weight = weight
        self.power_weight = power_weight

        self.w = 4.0 #nn.Parameter(torch.tensor(2.0))
        self.b = -2 #nn.Parameter(torch.tensor(-1.0))

        self.drop_p = get_weight(
            weight, input_dim, power_weight)  # * dropout_p

        self.activation = nn.Sigmoid()

    def forward(self, x):
        drop_weight = torch.tensor(self.drop_p).float().unsqueeze(0).unsqueeze(0)

        if len(x.shape) == 4:
            drop_weight = drop_weight.unsqueeze(0)
        
        if x.is_cuda:
            drop_weight = drop_weight.cuda()

        drop_weight = self.w * drop_weight + self.b
        drop_weight = self.activation(drop_weight)

        # maxfix_mean
        # beta = drop_weight.mean()
        # drop_weight = drop_weight.clamp_max(beta)/beta
        
        # maxfix
        drop_weight /= drop_weight.max()

        return x * drop_weight

    def __repr__(self):

        return "AttentionweightLayer_v3(input_dim=%d, weight=%s, w=%.4f, b=%.4f)" % (
            self.input_dim, self.weight, self.w, self.b)


class DropweightLayer_v3(nn.Module):
    def __init__(self, weight, dropout_p=0.1, input_dim=161,
                 scale=0.2, power_weight='norm'):
        super(DropweightLayer_v3, self).__init__()
        self.input_dim = input_dim
        self.weight = weight
        self.dropout_p = dropout_p
        self.scale = scale
        self.power_weight = power_weight

        drop_p = get_weight(
            weight, input_dim, power_weight)

        # self.drop_p = drop_p * self.scale + 1 - self.scale - dropout_p
        numofchance = 1 / np.clip(drop_p, a_min=0.05, a_max=None)
        numofchance = np.ceil(numofchance)
        all_bins = []
        for i,b in enumerate(numofchance):
            all_bins.extend([i]*int(b))
            
        self.drop_p = all_bins

    def forward(self, x):
        if (not self.training) or torch.Tensor(1).uniform_(0, 1) < 0.5:
            return x
        else:
            # assert len(self.drop_p) == x.shape[-1], print(len(self.drop_p), x.shape)
            x_mean = torch.mean(x, dim=-2, keepdim=True).repeat(1,1,x.shape[2],1)
            x_std = torch.std(x, dim=-2, keepdim=True).repeat(1,1,x.shape[2],1)
            
            mask_x = torch.normal(x_mean, std=x_std) # mask with normal distributions
            # all drop
            # for i,p in enumerate(self.drop_p):
            #     if torch.Tensor(1).uniform_(0, 1) > p:
            #         x[:,:,:,i] = mask_x[:,:,:,i]
            
            # specaug-like mask 
            mask_xs = np.random.choice(self.drop_p, size=5)
            for i in set(mask_xs):
                x[:,:,:,i] = mask_x[:,:,:,i]
                
            # specaug mask 
            # mask_xs = np.random.choice(self.drop_p)
            # for i in range(mask_xs, mask_xs+5):
            #     if i < x.shape[-1]:
            #         x[:,:,:,i] = mask_x[:,:,:,i]
                    
            return x 

    def __repr__(self):
        return "DropweightLayer_v3(input_dim=%d, weight=%s, dropout_p==%s, scale=%f)" % (self.input_dim, self.weight,
                                                                                         self.dropout_p, self.scale)


class AttentionweightLayer_v0(nn.Module):
    def __init__(self, input_dim=161, weight='mel', power_weight=False,
                 weight_norm='max'):
        super(AttentionweightLayer_v0, self).__init__()
        self.input_dim = input_dim
        self.power_weight = power_weight
        self.weight_norm = weight_norm
        self.weight = weight
        # self.s = nn.Parameter(torch.tensor(0.5))
        # self.b = nn.Parameter(torch.tensor(0.75))
        self.drop_p = get_weight(
            weight, input_dim, power_weight)  # * dropout_p
        # self.activation = nn.Sigmoid()

    def forward(self, x):
        # assert len(self.drop_p) == x.shape[-1], print(len(self.drop_p), x.shape)
        if len(x.shape) == 4:
            drop_weight = torch.tensor(
                self.drop_p).reshape(1, 1, 1, -1).float()
        else:
            drop_weight = torch.tensor(self.drop_p).reshape(1, 1, -1).float()

        if x.is_cuda:
            drop_weight = drop_weight.cuda()

        return x * drop_weight

    def __repr__(self):
        return "AttentionweightLayer_v00(input_dim=%d, weight=%s, power_weight=%s)" % (
            self.input_dim, self.weight, str(self.power_weight))


class GaussianNoiseLayer(nn.Module):
    def __init__(self, dropout_p=0.01, input_dim=161):
        super(GaussianNoiseLayer, self).__init__()
        self.input_dim = input_dim
        m = np.arange(0, 2840.0230467083188)
        m = 700 * (10 ** (m / 2595.0) - 1)
        n = np.array([m[i] - m[i - 1] for i in range(1, len(m))])
        n = 1 / n
        # x = np.arange(input_dim) * 8000 / (input_dim - 1)  # [0-8000]

        f = interpolate.interp1d(m[1:], n)
        xnew = np.arange(np.min(m[1:]), np.max(
            m[1:]), (np.max(m[1:]) - np.min(m[1:])) / input_dim)
        ynew = f(xnew)
        ynew = 1 / ynew  # .max()
        ynew /= ynew.max()

        self.gaussion_weight = torch.tensor(
            ynew * dropout_p).reshape(1, 1, 1, -1).float()

    def forward(self, x):
        if not self.training:
            return x
        else:
            assert self.gaussion_weight.shape[-1] == x.shape[-1], print(
                len(self.gaussion_weight), x.shape)

            x_mean = torch.mean(x, dim=2, keepdim=True)
            x_std = torch.std(x, dim=2, keepdim=True)

            gaussian_noise = torch.normal(mean=x_mean, std=x_std)
            drop_weight = self.gaussion_weight.cuda() if x.is_cuda else self.gaussian_noise
            gaussian_noise *= drop_weight

            return x + gaussian_noise


class MusanNoiseLayer(nn.Module):
    def __init__(self, snr=15, input_dim=161):
        super(MusanNoiseLayer, self).__init__()
        self.input_dim = input_dim
        self.mean = torch.FloatTensor(c.MUSAN_MEAN)
        self.std = torch.FloatTensor(c.MUSAN_STD)

        self.weight = 1 / np.power(10, snr / 10)
        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        if not self.training:
            return x
        else:
            gaussian_noise = torch.normal(
                mean=self.mean, std=self.std).reshape(1, 1, 1, x.shape[3])
            gaussian_noise = gaussian_noise.repeat(1, 1, x.shape[2], 1)

            weight = torch.ones(size=(1, 1, x.shape[2], 1))
            torch.nn.init.uniform_(weight, self.weight, 0.4)
            weight = torch.nn.functional.dropout(weight, p=0.5, training=True)
            weight = torch.where(weight > 1.0, torch.tensor(
                (self.weight+1.0)/2), weight)

            gaussian_noise *= weight
            noise_weight = gaussian_noise.cuda() if x.is_cuda else gaussian_noise

            return x + noise_weight


class CBAM(nn.Module):
    # input should be like [Batch, channel, time, frequency]
    def __init__(self, inplanes, planes, time_freq='both'):
        super(CBAM, self).__init__()
        self.time_freq = time_freq

        self.cov_t = nn.Conv2d(inplanes, planes, kernel_size=(
            7, 1), stride=1, padding=(3, 0))
        # self.avg_t = nn.AdaptiveAvgPool2d((None, 1))

        self.cov_f = nn.Conv2d(inplanes, planes, kernel_size=(
            1, 7), stride=1, padding=(0, 3))
        # self.avg_f = nn.AdaptiveAvgPool2d((1, None))

        self.activation = nn.Sigmoid()

    def forward(self, input):
        # t_output = self.avg_t(input)
        t_output = input.mean(dim=2, keepdim=True)
        t_output = self.cov_t(t_output)
        t_output = self.activation(t_output)
        # t_output = input * t_output
        # f_output = self.avg_f(input)
        f_output = input.mean(dim=3, keepdim=True)
        f_output = self.cov_f(f_output)
        f_output = self.activation(f_output)
        # f_output = input * f_output
        output = (t_output/2 + f_output/2) * input

        return output


class SqueezeExcitation(nn.Module):
    # input should be like [Batch, channel, time, frequency]
    def __init__(self, inplanes, reduction_ratio=4):
        super(SqueezeExcitation, self).__init__()
        self.reduction_ratio = reduction_ratio

        self.glob_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(inplanes, max(
            int(inplanes / self.reduction_ratio), 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(
            max(int(inplanes / self.reduction_ratio), 1), inplanes)
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
                self.handlers.append(
                    module.register_backward_hook(self._get_features_hook))
                self.handlers.append(
                    module.register_backward_hook(self._get_grads_hook))

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
            # nn.LayerNorm([80, int((self.current_input - 251 + 1) / 6 / 3)]),
            nn.InstanceNorm1d(80),
            nn.LeakyReLU(),
        )

        self.current_input = int((self.current_input - 251 + 1) / 6 / 3)
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=80, out_channels=60,
                      kernel_size=5, stride=1),
            nn.MaxPool1d(kernel_size=3),  # nn.AvgPool1d(kernel_size=3),
            # nn.LayerNorm([60, int((self.current_input - 5 + 1) / 3)]),
            nn.InstanceNorm1d(60),
            nn.LeakyReLU(),
        )

        self.current_input = int((self.current_input - 5 + 1) / 3)
        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=self.out_dim,
                      kernel_size=5, stride=1),
            nn.MaxPool1d(kernel_size=3),
            # nn.LayerNorm([self.out_dim, int((self.current_input - 5 + 1) / 3)]),
            nn.InstanceNorm1d(self.out_dim),
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


class Sinc2Down(nn.Module):
    def __init__(self, input_dim, out_dim=60, fs=16000):
        super(Sinc2Down, self).__init__()
        self.fs = fs
        self.input_dim = input_dim
        self.current_input = input_dim
        self.out_dim = out_dim

        # conv_layers = [(80, 251, 1), (60, 5, 1), (out_dim, 5, 1)]
        self.conv_layers = nn.ModuleList()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=int(input_dim / 2), out_channels=80,
                      kernel_size=(1, 31), stride=(1, 4), bias=False),
            # SincConv_fast(80, 251, self.fs, stride=6),
            # nn.MaxPool1d(kernel_size=3),  # nn.AvgPool1d(kernel_size=3),
            # nn.LayerNorm([80, int((self.current_input - 251 + 1) / 6 / 3)]),
            nn.InstanceNorm2d(80),
            nn.LeakyReLU(),
            nn.Dropout2d(0.5)
        )

        # self.current_input = int((self.current_input - 251 + 1) / 6 / 3)
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=80, out_channels=60,
                      kernel_size=(2, 5), stride=(1, 2)),
            # nn.MaxPool1d(kernel_size=3),  # nn.AvgPool1d(kernel_size=3),
            # nn.LayerNorm([60, int((self.current_input - 5 + 1) / 3)]),
            nn.InstanceNorm2d(60),
            nn.LeakyReLU(),
            nn.Dropout2d(0.5)
        )
        #
        # self.current_input = int((self.current_input - 5 + 1) / 3)
        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=60,
                      kernel_size=5, stride=2),
            # nn.MaxPool1d(kernel_size=3),
            # nn.LayerNorm([self.out_dim, int((self.current_input - 5 + 1) / 3)]),
            nn.InstanceNorm1d(60),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        #
        self.conv_layer4 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=self.out_dim,
                      kernel_size=5, stride=2),
            # nn.AvgPool1d(kernel_size=3),  # nn.MaxPool1d(kernel_size=3),
            # nn.LayerNorm([self.out_dim, int((self.current_input - 5 + 1) / 3)]),
            nn.InstanceNorm1d(self.out_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )

        self.current_output = int((self.current_input - 5 + 1) / 3)

    def forward(self, x):
        # BxT -> BxCxT
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 4:
            x = x.squeeze(1)

        # print(x.shape)
        if x.shape[2] == self.input_dim:
            x = x.transpose(1, 2)

        x_shape = x.shape
        x = x.reshape(x_shape[0], 20, 2, -1)

        x = self.conv_layer1(x)
        x = self.conv_layer2(x)

        x = x.squeeze(2)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)

        return x.transpose(1, 2)


# https://github.com/pytorch/fairseq/blob/c47a9b2eef0f41b0564c8daf52cb82ea97fc6548/fairseq/models/wav2vec/wav2vec.py#L367
class Wav2Conv(nn.Module):
    def __init__(self, out_dim=512, log_compression=True):
        super(Wav2Conv, self).__init__()

        in_d = 1
        conv_layers = [(40, 10, 5), (200, 5, 4), (300, 3, 2),
                       (512, 3, 2), (out_dim, 3, 2)]
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
            # nn.GroupNorm(1, n_out), in wav2spk replace group by instance normalization
            nn.InstanceNorm1d(n_out),
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


class Wav2Down(nn.Module):
    def __init__(self, input_dim=1, out_dim=512, log_compression=False):
        super(Wav2Down, self).__init__()

        self.input_dim = input_dim
        in_d = input_dim
        # conv_layers = [(40, 10, 5), (200, 5, 4), (300, 3, 2), (512, 3, 2), (out_dim, 3, 2)]
        conv_layers = [(40, 10, 4), (200, 5, 2), (300, 3, 2),
                       (512, 3, 2), (out_dim, 3, 1)]
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
            # nn.GroupNorm(1, n_out), in wav2spk replace group by instance normalization
            nn.InstanceNorm1d(n_out),
            nn.ReLU(),
        )

    def forward(self, x):
        # BxT -> BxCxT
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 4:
            x = x.squeeze(1)

        if x.shape[2] == self.input_dim:
            x = x.transpose(1, 2)

        for conv in self.conv_layers:
            x = conv(x)

        if self.log_compression:
            x = x.abs()
            x = x + 1
            x = x.log()

        tmp_gate = self.tmp_gate(x.transpose(1, 2)).transpose(1, 2)
        x = x * tmp_gate
        return x.transpose(1, 2)
