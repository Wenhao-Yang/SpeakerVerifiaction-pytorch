#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: train_egs.py
@Time: 2020/8/21 20:30
@Overview:
from speechbrain

A popular speaker recognition and diarization model.

Authors
 * Hwidong Na 2020

"""

# import os
from scipy import stats
import torch  # noqa: F401
import numpy as np
from scipy.stats import burr12
import torch.nn as nn
import torch.nn.functional as F
from Define_Model.NoiseInjection import AttentionDrop1d, MagCauchyNoiseInject, MagnitudeDropout1d, NoiseInject, DropBlock1d, Dropout1d, DropAttention1d, AttentionNoiseInject, PartialAttentionNoiseInject, RadioNoiseInject
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.nnet.CNN import Conv1d as _Conv1d
from speechbrain.nnet.normalization import BatchNorm1d as _BatchNorm1d
from speechbrain.nnet.linear import Linear

from Define_Model.FilterLayer import get_filter_layer, get_input_norm, get_mask_layer


# Skip transpose as much as possible for efficiency
class Conv1d(_Conv1d):
    """1D convolution. Skip transpose is used to improve efficiency."""

    def __init__(self, *args, **kwargs):
        super().__init__(skip_transpose=True, *args, **kwargs)


class BatchNorm1d(_BatchNorm1d):
    """1D batch normalization. Skip transpose is used to improve efficiency."""

    def __init__(self, *args, **kwargs):
        super().__init__(skip_transpose=True, *args, **kwargs)


class InstBatchNorm1d(nn.Module):
    """An implementation of InstBatchNorm1d.

    Arguments
    ----------
    in_channels : int
        Number of input channels.
    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> layer = TDNNBlock(64, 64, kernel_size=3, dilation=1)
    >>> out_tensor = layer(inp_tensor).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(self, in_channels, shuffle=False, bath_ratio=0.5, affine=False):
        super(InstBatchNorm1d, self).__init__()

        self.bath_ratio = bath_ratio
        self.bath_size  = int(in_channels * bath_ratio)
        self.inst_size = in_channels - self.bath_size
        self.batch_norm = BatchNorm1d(input_size=self.bath_size)
        self.inst_norm = nn.InstanceNorm1d(self.inst_size, affine=affine)
        self.shuffle = shuffle

    def forward(self, x):
        """ Processes the input tensor x and returns an output tensor."""
        x_shape = x.shape
        if self.shuffle:
            # x = torch.nn.functional.channel_shuffle(x, 2)
            x = x.reshape(x_shape[0], 2, -1, x_shape[2]).transpose(1,2).reshape(x_shape[0], x_shape[1], x_shape[2])
        x1 = x[:, :self.bath_size]
        x2 = x[:, -self.inst_size:]

        x1 = self.batch_norm(x1)
        x2 = self.inst_norm(x2)

        x = torch.cat([x1, x2], dim=1)

        if self.shuffle:
            # x = torch.nn.functional.channel_shuffle(x, x_shape[1]//2)
            x = x.reshape(x_shape[0], -1, 2, x_shape[2]).transpose(1,2).reshape(x_shape[0], x_shape[1], x_shape[2])

        return x


class TDNNBlock(nn.Module):
    """An implementation of TDNN.

    Arguments
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        The number of output channels.
    kernel_size : int
        The kernel size of the TDNN blocks.
    dilation : int
        The dilation of the TDNN block.
    activation : torch class
        A class for constructing the activation layers.
    groups: int
        The groups size of the TDNN blocks.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> layer = TDNNBlock(64, 64, kernel_size=3, dilation=1)
    >>> out_tensor = layer(inp_tensor).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        activation=nn.ReLU,
        groups=1,
        norm='batch', shuffle=False, bath_ratio=0.5, affine=False
    ):
        super(TDNNBlock, self).__init__()
        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
        )
        self.activation = activation()
        if norm == 'inst':
            self.norm = nn.InstanceNorm1d(out_channels)
        elif norm == 'inbn':
            self.norm = InstBatchNorm1d(out_channels, shuffle, bath_ratio=bath_ratio, affine=affine)
        elif norm == 'group':
            self.norm = nn.GroupNorm(int(bath_ratio), out_channels)
        else:
            self.norm = BatchNorm1d(input_size=out_channels)

    def forward(self, x):
        """ Processes the input tensor x and returns an output tensor."""
        return self.norm(self.activation(self.conv(x)))


class TDNNBottleBlock(nn.Module):
    """An implementation of TDNN.

    Arguments
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        The number of output channels.
    kernel_size : int
        The kernel size of the TDNN blocks.
    dilation : int
        The dilation of the TDNN block.
    activation : torch class
        A class for constructing the activation layers.
    groups: int
        The groups size of the TDNN blocks.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> layer = TDNNBlock(64, 64, kernel_size=3, dilation=1)
    >>> out_tensor = layer(inp_tensor).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(
        self,
        in_channels,
        bottle_scale,
        out_channels,
        kernel_size,
        dilation,
        activation=nn.ReLU,
        groups=1,
    ):
        super(TDNNBottleBlock, self).__init__()
        self.conv1 = Conv1d(
            in_channels=in_channels,
            out_channels=bottle_scale,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
        )
        self.activation = activation()
        self.norm1 = BatchNorm1d(input_size=bottle_scale)
        
        self.conv2 = Conv1d(
            in_channels=bottle_scale,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
        )
        self.norm2 = BatchNorm1d(input_size=out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal(m.weight, mean=0., std=.02)
                if m.bias != None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """ Processes the input tensor x and returns an output tensor."""
        # print(x.shape)
        x = self.norm1(self.activation(self.conv1(x)))
        x = self.norm2(self.activation(self.conv2(x)))
        return x


class Res2NetBlock(torch.nn.Module):
    """An implementation of Res2NetBlock w/ dilation.

    Arguments
    ---------
    in_channels : int
        The number of channels expected in the input.
    out_channels : int
        The number of output channels.
    scale : int
        The scale of the Res2Net block.
    kernel_size: int
        The kernel size of the Res2Net block.
    dilation : int
        The dilation of the Res2Net block.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> layer = Res2NetBlock(64, 64, scale=4, dilation=3)
    >>> out_tensor = layer(inp_tensor).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(
        self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1
    ):
        super(Res2NetBlock, self).__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.blocks = nn.ModuleList(
            [
                TDNNBlock(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for i in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, x):
        """ Processes the input tensor x and returns an output tensor."""
        y = []
        for i, x_i in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            y.append(y_i)
        y = torch.cat(y, dim=1)
        return y


class SEBlock(nn.Module):
    """An implementation of squeeze-and-excitation block.

    Arguments
    ---------
    in_channels : int
        The number of input channels.
    se_channels : int
        The number of output channels after squeeze.
    out_channels : int
        The number of output channels.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> se_layer = SEBlock(64, 16, 64)
    >>> lengths = torch.rand((8,))
    >>> out_tensor = se_layer(inp_tensor, lengths).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(self, in_channels, se_channels, out_channels,
                 dropout_type='vanilla', dropout_p=0,
                 linear_step=0):
        super(SEBlock, self).__init__()

        self.conv1 = Conv1d(
            in_channels=in_channels, out_channels=se_channels, kernel_size=1
        )
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = Conv1d(
            in_channels=se_channels, out_channels=out_channels, kernel_size=1
        )
        self.sigmoid = torch.nn.Sigmoid()
        self.this_step = 0
        self.linear_step = linear_step
        self.dropout_p = dropout_p
        self.drop_after = False if 'before' in dropout_type else True

        if 'vanilla' in dropout_type:
            self.drop = nn.Dropout1d(dropout_p)
        elif 'magnitude' in dropout_type:
            self.drop = MagnitudeDropout1d(dropout_p)
        elif 'attention' in dropout_type:
            self.drop = DropAttention1d(dropout_p)
        elif 'dropblock' in dropout_type:
            self.drop = DropBlock1d(dropout_p, linear_step=linear_step)
        elif 'attendrop' in dropout_type:
            self.drop = AttentionDrop1d(dropout_p)
        elif 'attenoise' in dropout_type and isinstance(dropout_p, list):
            self.drop = AttentionNoiseInject(drop_prob=dropout_p)
        elif 'pattenoise' in dropout_type and isinstance(dropout_p, list):
            self.drop = PartialAttentionNoiseInject(drop_prob=dropout_p)
        else:
            self.drop = None

    def forward(self, x, lengths=None):
        """ Processes the input tensor x and returns an output tensor."""
        L = x.shape[-1]

        # linear create dropout
        if self.drop != None:
            if self.linear_step > 0:
                if self.this_step <= self.linear_step:
                    self.drop.p = self.dropout_p * self.this_step / self.linear_step
                    self.this_step += 1

        if not self.drop_after and self.drop != None:
            if isinstance(self.drop,  DropBlock1d) or isinstance(self.drop, AttentionNoiseInject):   
                x = self.drop(x)

        if lengths is not None:
            mask = length_to_mask(lengths * L, max_len=L, device=x.device)
            mask = mask.unsqueeze(1)
            total = mask.sum(dim=2, keepdim=True)
            s = (x * mask).sum(dim=2, keepdim=True) / total
        else:
            s = x.mean(dim=2, keepdim=True)

        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))

        if self.drop_after and self.drop != None:   
            if isinstance(self.drop,  DropBlock1d) or isinstance(self.drop, AttentionNoiseInject):   
                x = self.drop(x)
            else:    
                s = self.drop(s)

        return s * x


class AttentiveStatisticsPooling(nn.Module):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.

    Arguments
    ---------
    channels: int
        The number of input channels.
    attention_channels: int
        The number of attention channels.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> asp_layer = AttentiveStatisticsPooling(64)
    >>> lengths = torch.rand((8,))
    >>> out_tensor = asp_layer(inp_tensor, lengths).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 1, 128])
    """

    def __init__(self, channels, attention_channels=128, global_context=True):
        super().__init__()

        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = Conv1d(
            in_channels=attention_channels, out_channels=channels, kernel_size=1
        )

    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape [N, C, L].
        """
        L = x.shape[-1]

        def _compute_statistics(x, m, dim=2, eps=self.eps):
            mean = (m * x).sum(dim)
            std = torch.sqrt(
                (m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps)
            )
            return mean, std

        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device)

        # Make binary mask of shape [N, 1, L]
        mask = length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        if self.global_context == True:
            # torch.std is unstable for backward computation
            # https://github.com/pytorch/pytorch/issues/4320
            total = mask.sum(dim=2, keepdim=True).float()
            mean, std = _compute_statistics(x, mask / total)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        elif self.global_context == 'window':
            total = mask.sum(dim=2, keepdim=True).float()
            means, stds = [], []
            for i in range(int(np.ceil(x.shape[-1]/25))):
                start = i*25
                end = min((i+1)*25, x.shape[-1])
                mean, std = _compute_statistics(x[:, :, start:end], mask[:, :, start:end] / total)
                mean = mean.unsqueeze(2).repeat(1, 1, end-start)
                std = std.unsqueeze(2).repeat(1, 1, end-start)
                means.append(mean)
                stds.append(std)

            mean = torch.cat(means, dim=-1)
            std = torch.cat(stds, dim=-1)

            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x

        # Apply layers
        attn = self.conv(self.tanh(self.tdnn(attn)))

        # Filter out zero-paddings
        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=2)
        mean, std = _compute_statistics(x, attn)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        # pooled_stats = pooled_stats  # .unsqueeze(2)

        return pooled_stats


class SequentialASTPooling(nn.Module):

    def __init__(self, channels, attention_channels=128, context_frames=100):
        super().__init__()
        self.pooling = AttentiveStatisticsPooling(channels, attention_channels)
        self.context_frames = context_frames
    
    def forward(self, x, lengths=None):
        embeddings = []
        for i in range(torch.round(x.shape[-1]/self.context_frames)):
            
            end = min((i+1)*self.context_frames, x.shape[-1])
            start = max(0, end-self.context_frames)

            embeddings.append(self.pooling(x[:, :, start:end], lengths))
        
        return embeddings



class AttentiveMultiStatisticsPooling(nn.Module):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.

    Arguments
    ---------
    channels: int
        The number of input channels.
    attention_channels: int
        The number of attention channels.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> asp_layer = AttentiveStatisticsPooling(64)
    >>> lengths = torch.rand((8,))
    >>> out_tensor = asp_layer(inp_tensor, lengths).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 1, 128])
    """

    def __init__(self, channels, attention_channels=128, global_context=True,
                 stddev=True,):
        super().__init__()

        self.eps = 1e-12
        self.stddev = stddev
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
        self.tanh = nn.Sigmoid()
        self.conv = Conv1d(
            in_channels=attention_channels, out_channels=channels, kernel_size=1
        )

    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape [N, C, L].
        """
        L = x.shape[-1]

        def _compute_statistics(x, m, dim=2, eps=self.eps):
            mean = (m * x).sum(dim)
            std = torch.sqrt(
                (m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps)
            )
            return mean, std

        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device)

        # Make binary mask of shape [N, 1, L]
        mask = length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        if self.global_context == True:
            # torch.std is unstable for backward computation
            # https://github.com/pytorch/pytorch/issues/4320
            total = mask.sum(dim=2, keepdim=True).float()
            mean, std = _compute_statistics(x, mask / total)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        elif self.global_context == 'window':
            total = mask.sum(dim=2, keepdim=True).float()
            means, stds = [], []
            for i in range(int(np.ceil(x.shape[-1]/25))):
                start = i*25
                end = min((i+1)*25, x.shape[-1])
                mean, std = _compute_statistics(x[:, :, start:end],
                                                mask[:, :, start:end] / total)
                mean = mean.unsqueeze(2).repeat(1, 1, end-start)
                std = std.unsqueeze(2).repeat(1, 1, end-start)
                means.append(mean)
                stds.append(std)

            mean = torch.cat(means, dim=-1)
            std = torch.cat(stds, dim=-1)

            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x

        # Apply layers
        attn = self.tanh(self.tdnn(attn))
        attn1 = self.conv(attn)
        attn2 = self.conv(1-attn)

        # Filter out zero-paddings
        attn1 = attn1.masked_fill(mask == 0, float("-inf"))
        attn1 = F.softmax(attn1, dim=2)
        mean1, std1 = _compute_statistics(x, attn1)

        attn2 = attn2.masked_fill(mask == 0, float("-inf"))
        attn2 = F.softmax(attn2, dim=2)
        mean2, std2 = _compute_statistics(x, attn2)
        # Append mean and std of the batch
        if self.stddev == True :
            pooled_stats = torch.cat((mean1, std1), dim=1) + torch.cat((mean2, std2), dim=1)
        elif self.stddev == 'half' :
            pooled_stats = torch.cat((mean1, std1), dim=1) + 0.5 * torch.cat((mean2, std2), dim=1)
        else:
            pooled_stats = torch.cat((mean1, mean2), dim=1)
        # pooled_stats = pooled_stats  # .unsqueeze(2)

        return pooled_stats


class SERes2NetBlock(nn.Module):
    """An implementation of building block in ECAPA-TDNN, i.e.,
    TDNN-Res2Net-TDNN-SEBlock.

    Arguments
    ----------
    out_channels: int
        The number of output channels.
    res2net_scale: int
        The scale of the Res2Net block.
    kernel_size: int
        The kernel size of the TDNN blocks.
    dilation: int
        The dilation of the Res2Net block.
    activation : torch class
        A class for constructing the activation layers.
    groups: int
    Number of blocked connections from input channels to output channels.

    Example
    -------
    >>> x = torch.rand(8, 120, 64).transpose(1, 2)
    >>> conv = SERes2NetBlock(64, 64, res2net_scale=4)
    >>> out = conv(x).transpose(1, 2)
    >>> out.shape
    torch.Size([8, 120, 64])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        res2net_scale=8,
        se_channels=128,
        kernel_size=1,
        dilation=1,
        activation=torch.nn.ReLU,
        groups=1, norm='batch', shuffle=False, bath_ratio=0.5, affine=False,
        dropout_type='vanilla', dropout_p=0.0, linear_step=0
    ):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TDNNBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
            groups=groups,
            norm=norm, shuffle=shuffle, bath_ratio=bath_ratio, affine=affine
        )
        self.res2net_block = Res2NetBlock(
            out_channels, out_channels, res2net_scale, kernel_size, dilation
        )
        self.tdnn2 = TDNNBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
            groups=groups,
        )
        self.se_block = SEBlock(out_channels, se_channels, out_channels,
                                dropout_type=dropout_type, dropout_p=dropout_p,
                                linear_step=linear_step)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x, lengths=None):
        """ Processes the input tensor x and returns an output tensor."""
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x, lengths)

        return x + residual
    
    
class SERes2NetBottleblock(nn.Module):
    """An implementation of building block in ECAPA-TDNN, i.e.,
    TDNN-Res2Net-TDNN-SEBlock.

    Arguments
    ----------
    out_channels: int
        The number of output channels.
    res2net_scale: int
        The scale of the Res2Net block.
    kernel_size: int
        The kernel size of the TDNN blocks.
    dilation: int
        The dilation of the Res2Net block.
    activation : torch class
        A class for constructing the activation layers.
    groups: int
    Number of blocked connections from input channels to output channels.

    Example
    -------
    >>> x = torch.rand(8, 120, 64).transpose(1, 2)
    >>> conv = SERes2NetBlock(64, 64, res2net_scale=4)
    >>> out = conv(x).transpose(1, 2)
    >>> out.shape
    torch.Size([8, 120, 64])
    """

    def __init__(
        self,
        in_channels,
        bottle_scale,
        out_channels,
        res2net_scale=4,
        se_channels=64,
        kernel_size=1,
        dilation=1,
        activation=torch.nn.ReLU,
        groups=1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TDNNBlock(
            in_channels,
            bottle_scale,
            kernel_size=1,
            dilation=1,
            activation=activation,
            groups=groups,
        )
        self.res2net_block = Res2NetBlock(
            bottle_scale, bottle_scale, res2net_scale, kernel_size, dilation
        )
        self.tdnn2 = TDNNBlock(
            bottle_scale,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
            groups=groups,
        )
        self.se_block = SEBlock(out_channels, bottle_scale, out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal(m.weight, mean=0., std=.02)
                if m.bias != None:
                    nn.init.zeros_(m.bias)


    def forward(self, x, lengths=None):
        """ Processes the input tensor x and returns an output tensor."""
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x, lengths)

        return x + residual


class ECAPA_TDNN(torch.nn.Module):
    """An implementation of the speaker embedding model in a paper.
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143).

    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    activation : torch class
        A class for constructing the activation layers.
    channels : list of ints
        Output channels for TDNN/SERes2Net layer.
    kernel_sizes : list of ints
        List of kernel sizes for each layer.
    dilations : list of ints
        List of dilations for kernels in each layer.
    lin_neurons : int
        Number of neurons in linear layers.
    groups : list of ints
        List of groups for kernels in each layer.

    Example
    -------
    >>> input_feats = torch.rand([5, 120, 80])
    >>> compute_embedding = ECAPA_TDNN(80, lin_neurons=192)
    >>> outputs = compute_embedding(input_feats)
    >>> outputs.shape
    torch.Size([5, 1, 192])
    """

    def __init__(
            self, input_dim, num_classes, embedding_size=192, activation=torch.nn.ReLU,
            input_norm='', filter=None, sr=16000, feat_dim=80, exp=False, filter_fix=False,
            win_length=int(0.025*16000), nfft=512, stretch_ratio=[1.0],
            init_weight='mel', scale=0.2, weight_p=0.1, weight_norm='max',
            mask='None', mask_len=[5, 20], mask_ckp='',
            channels=[512, 512, 512, 512, 1536],
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            encoder_type='SASP2', encoder_bias=False, stddev=True,
            norm='batch', shuffle=False, bath_ratio=0.5, affine=False,
            dropouts=[0, 0, 0], dropout_type='vanilla', linear_step=0,
            noise_norm='none', noise_type='none',
            domain_feat='embeddings',
            domain_mix=False,
            attention_channels=128,
            res2net_scale=8,
            se_channels=128,
            global_context=True,
            mix='mixup',
            groups=[1, 1, 1, 1, 1], **kwargs):

        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.domain_feat = domain_feat
        self.mix_type = mix
        self.encoder_type = encoder_type
        mix_types = {
            "mixup": self.mixup,
            "manifold": self.mixup,
            # "addup": self.addup,
            "style": self.mixstyle,
            # "align": self.alignmix,
            # "style_time": self.mixstyle_time,
            # "style_base": self.mixbase,
            "cutmix": self.cutmix,
            "cutmixup": self.cutmixup,
            # "cutmixstyle": self.cutmixstylebase,
        }
        self.mix = mix_types[mix]

        if len(dropouts) == 3:
            self.dropouts = [0]
            self.dropouts.extend(dropouts)
        else:
            self.dropouts = dropouts

        input_mask = []
        filter_layer = get_filter_layer(filter=filter, input_dim=input_dim, sr=sr, feat_dim=feat_dim,
                                        exp=exp, filter_fix=filter_fix,
                                        stretch_ratio=stretch_ratio, win_length=win_length, nfft=nfft)
        if filter_layer != None:
            input_mask.append(filter_layer)
        norm_layer = get_input_norm(input_norm, input_dim=input_dim)
        if norm_layer != None:
            input_mask.append(norm_layer)
        mask_layer = get_mask_layer(mask=mask, mask_len=mask_len, input_dim=input_dim,
                                    init_weight=init_weight, weight_p=weight_p,
                                    scale=scale, weight_norm=weight_norm, mask_ckp=mask_ckp)
        if mask_layer != None:
            input_mask.append(mask_layer)
        self.input_mask = nn.Sequential(*input_mask)

        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.channels = channels
        self.domain_mix = domain_mix
        
        self.blocks = nn.ModuleList()
        if isinstance(bath_ratio, float):
            bath_ratio = [bath_ratio] * (len(channels)+1)
        elif len(bath_ratio) < len(channels) + 1:
            while len(bath_ratio) < len(channels) + 1:
                bath_ratio.append(0.5)
        self.bath_ratio = bath_ratio

        if isinstance(norm, str):
            norm = [norm] * (len(channels)+1)
        elif len(norm) < len(channels) + 1:
            while len(norm) < len(channels) + 1:
                norm.append('batch')
        self.norm = norm

        # The initial TDNN layer
        tdnn_layer1 = [
            TDNNBlock(
                    input_dim,
                    channels[0],
                    kernel_sizes[0],
                    dilations[0],
                    activation,
                    groups[0],
                    norm=norm[0], shuffle=shuffle, bath_ratio=bath_ratio[0], affine=affine) 
            ]
        
        if 'attenoise' in dropout_type:
            if isinstance(self.dropouts[0], float):
                tdnn_layer1.append(AttentionNoiseInject(drop_prob=self.dropouts[0:2]))
            else:
                tdnn_layer1.append(AttentionNoiseInject(drop_prob=self.dropouts[0]))
        elif 'noiseinject' in dropout_type:
            tdnn_layer1.append(NoiseInject(drop_prob=self.dropouts[0], input_dim=channels[0],
                                           noise_norm=noise_norm,
                                           noise_type=noise_type))
        elif 'magcauchy' in dropout_type:
            tdnn_layer1.append(MagCauchyNoiseInject(drop_prob=self.dropouts[0]))
        elif 'radionoise' in dropout_type:
            tdnn_layer1.append(RadioNoiseInject(drop_prob=self.dropouts[0]))
        elif 'dropblock' in dropout_type and self.dropouts[0] > 0:
            tdnn_layer1.append(DropBlock1d(drop_prob=self.dropouts[0], linear_step=linear_step))
        elif 'vanilla' in dropout_type and self.dropouts[0] > 0:
            tdnn_layer1.append(Dropout1d(drop_prob=self.dropouts[0], linear_step=linear_step))

        if len(tdnn_layer1) > 1:
            self.blocks.append(nn.Sequential(*tdnn_layer1))
        else:
            self.blocks.append(tdnn_layer1[0])

        # SE-Res2Net layers
        for i in range(1, len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    activation=activation,
                    groups=groups[i],
                    norm=norm[i], shuffle=shuffle, bath_ratio=bath_ratio[i], affine=affine,
                    dropout_type=dropout_type, dropout_p=self.dropouts[i],
                    linear_step=linear_step)
            )

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(
            channels[-1],
            channels[-1],
            kernel_sizes[-1],
            dilations[-1],
            activation,
            groups=groups[-1],
            norm=norm[-2], shuffle=shuffle, bath_ratio=bath_ratio[-2], affine=affine
        )

        # Attentive Statistical Pooling
        if self.encoder_type == 'SASTP':
            self.asp = SequentialASTPooling(
                channels[-1],
                attention_channels=attention_channels,
                global_context=global_context,
            )

        else:
            self.asp = AttentiveStatisticsPooling(
                channels[-1],
                attention_channels=attention_channels,
                global_context=global_context,
            )

        if self.norm[-1] == 'batch':
            self.asp_bn = BatchNorm1d(input_size=channels[-1] * 2)
        elif self.norm[-1] == 'inst':
            self.asp_bn = nn.InstanceNorm1d(channels[-1] * 2)
        elif self.norm[-1] == 'group':
            self.asp_bn = nn.GroupNorm(int(bath_ratio[-1]), channels[-1] * 2)
        elif self.norm[-1] == 'inbn':
            self.asp_bn = InstBatchNorm1d(channels[-1] * 2, shuffle,
                                          bath_ratio=bath_ratio[-1], affine=affine)

        # Final linear transformation
        # self.fc = Conv1d(
        #     in_channels=channels[-1] * 2,
        #     out_channels=embedding_size,
        #     kernel_size=1,)
        self.fc = nn.Linear(channels[-1] * 2, embedding_size)

        self.classifier = Classifier(
            input_size=embedding_size, lin_neurons=embedding_size, out_neurons=num_classes)

    def forward(self, x, lengths=None, last=False,
                freeze=False, feature_map='',
                lamda_beta=0.2, mixup_alpha=-1, proser=None,):
        """Returns the embedding vector.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        """
        if isinstance(mixup_alpha, float) or isinstance(mixup_alpha, int):
            layer_mix = mixup_alpha
        elif isinstance(mixup_alpha, list):
            layer_mix = np.random.choice(mixup_alpha)

        if freeze:
            with torch.no_grad():
                x = self.input_mask(x)
                if len(x.shape) == 4:
                    x = x.squeeze(1).float()
                x = x.transpose(1, 2)

                xl = []
                for layer in self.blocks:
                    try:
                        x = layer(x, lengths=lengths)
                    except TypeError:
                        x = layer(x)
                    xl.append(x)
                    
                x = torch.cat(xl[1:], dim=1)
                x = self.mfa(x)
                x = self.asp(x, lengths=lengths)
                x = self.asp_bn(x)
                embeddings = self.fc(x)
        else:
            if proser != None and layer_mix == 0:
                x = self.mix(x, proser, lamda_beta)
            x = self.input_mask(x)
            if len(x.shape) == 4:
                x = x.squeeze(1).float()
            x = x.transpose(1, 2)

            if proser != None and layer_mix == 1:
                x = self.mix(x, proser, lamda_beta)

            xl = []
            for j, layer in enumerate(self.blocks):
                try:
                    x = layer(x, lengths=lengths)
                except TypeError:
                    x = layer(x)
                
                if proser != None and layer_mix == j+2:
                    x = self.mix(x, proser, lamda_beta)

                xl.append(x)

            # Multi-layer feature aggregation
            x_cat = torch.cat(xl[1:], dim=1)
            x_mfa = self.mfa(x_cat)
            if proser != None and layer_mix == 6:
                x = self.mix(x, proser, lamda_beta)

            # Attentive Statistical Pooling
            x = self.asp(x_mfa, lengths=lengths)
            if isinstance(x, list):
                logits, embeddings = self.sequential_classify(x)
                
                return logits, embeddings
            
            x = self.asp_bn(x)

            if proser != None and layer_mix == 7:
                x = self.mix(x, proser, lamda_beta)

            if self.domain_mix and self.training:
                x = self.seperate(x)

            # Final linear transformation
            embeddings = self.fc(x)

            if proser != None and layer_mix == 8:
                embeddings = self.mix(embeddings, proser, lamda_beta)

        logits = self.classifier(embeddings)

        if hasattr(self, 'domain_classifier') and self.training:
            if self.domain_feat == 'embeddings':
                domain_embeddings = embeddings
            elif self.domain_feat == 'concat':
                domain_embeddings = x_cat
            elif self.domain_feat == 'mfa':
                domain_embeddings = x_mfa
            else:
                raise ValueError('domain_feat must be either "embeddings" or "logits"')
            
            dlogits = self.domain_classifier((domain_embeddings, logits))
            logits  = (logits, dlogits)
        
        if feature_map == 'attention':
            embeddings = xl

        return logits, embeddings
    
    def sequential_classify(self, statistics, embedding_sum=True):

        logits = []
        embeddings = []

        for x in statistics:
            x = self.asp_bn(x)
            emb = self.fc(x)
            embeddings.append(emb)
            logits.append(self.classifier(emb))
        
        logits = torch.stack(logits, dim=-1).sum(dim=-1)
        embeddings = torch.stack(embeddings, dim=-1)

        if embedding_sum:
            embeddings = embeddings.sum(dim=-1)

        return logits, embeddings


    def get_embedding_dim(self):
        return self.embedding_size
    
    def seperate(self, x):
        
        x_shape = x.shape
        clean_xs, domain_xs = x.reshape(2, -1, x_shape[1])#.clone()
        
        # beta distributions interpolation
        lambda1 = torch.tensor(stats.beta.rvs(2, 2, size=clean_xs.shape[0])).float().unsqueeze(-1).to(x.device)
        mix_xs = clean_xs * lambda1 + domain_xs * (1-lambda1)
        
        return torch.cat([x, mix_xs], dim=0)


    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for pp in list(self.parameters()):
            if pp.requires_grad: # only using the parameter that require the gradient
                grads.append(pp.grad.view(-1))
        return torch.cat(grads)
    
    def mixup(self, x, idx_tensor, lamda_beta):
        mix_size = idx_tensor.shape[0]
        half_feats = x[-mix_size:]
        x = torch.cat(
            [x[:-mix_size], lamda_beta * half_feats +
                (1 - lamda_beta) * half_feats[idx_tensor]],
            dim=0)

        return x
    
    def cutmix(self, x, idx_tensor, lamda_beta):
        x_shape = x.shape
        if len(x_shape) == 2:
            x = x.unsqueeze(1)
        elif len(x_shape) == 4:
            x = x.squeeze(1)

        mix_size = idx_tensor.shape[0]
        half_feats = x[-mix_size:]

        lam_t = int(half_feats.shape[2] * (1.0 - lamda_beta))
        if lam_t > 0:
            if lam_t < half_feats.shape[2]:
                t_start = np.random.randint(0, half_feats.shape[2]-lam_t)
            else:
                t_start = 0

            half_feats_shuf = half_feats[idx_tensor].clone().detach()
            if lam_t > 0:
                end_t = t_start + lam_t
                half_feats[:, :, t_start:end_t] = half_feats_shuf[:, :, t_start:end_t]

            x = torch.cat(
                [x[:-mix_size], half_feats],
                dim=0)
            
        if len(x_shape) == 2:
            x = x.squeeze(1)
        elif len(x_shape) == 4:
            x = x.unsqueeze(1)

        return x

    def cutmixup(self, x, idx_tensor, lamda_beta):
        x_shape = x.shape
        if len(x_shape) == 2:
            x = x.unsqueeze(1)
        elif len(x_shape) == 4:
            x = x.squeeze(1)

        mix_size = idx_tensor.shape[0]
        half_feats = x[-mix_size:]

        mix_lamda_beta = torch.ones(1,1,x.shape[2]).float().to(x.device) * lamda_beta
        cut_lamda_beta = torch.ones(1,1,x.shape[2]).float().to(x.device)

        lam_t = int(half_feats.shape[2] * (1.0 - lamda_beta))
        if lam_t > 0:
            if lam_t < half_feats.shape[2]:
                t_start = np.random.randint(0, half_feats.shape[2]-lam_t)
            else:
                t_start = 0

            half_feats_shuf = half_feats[idx_tensor].clone().detach()
            if lam_t > 0:
                end_t = t_start + lam_t
                cut_lamda_beta[:, :, t_start:end_t] *= 0 

            tensor_lamda_beta = mix_lamda_beta + cut_lamda_beta
            tensor_lamda_beta /= 2
            # t_lamda_beta = np.random.uniform(0, 1)
            # tensor_lamda_beta = t_lamda_beta * mix_lamda_beta + (1-t_lamda_beta)*cut_lamda_beta

            half_feats = half_feats * tensor_lamda_beta + half_feats_shuf*(1-tensor_lamda_beta)
            x = torch.cat(
                [x[:-mix_size], half_feats],
                dim=0)
        
        if len(x_shape) == 2:
            x = x.squeeze(1)
        elif len(x_shape) == 4:
            x = x.unsqueeze(1)

        return x

    def mixstyle(self, x, shuf_half_idx_ten, lamda_beta):
        x_shape = x.shape
        if len(x_shape) == 2:
            x = x.unsqueeze(1)
        elif len(x_shape) == 4:
            x = x.squeeze(1)

        mix_size = shuf_half_idx_ten.shape[0]
        half_feats = x[-mix_size:]

        mu  = half_feats.mean(dim=2, keepdim=True)
        var = half_feats.var(dim=2, keepdim=True)
        sig = (var + 1e-6).sqrt()
        # mu, sig = mu.detach(), sig.detach()
        x_normed = (half_feats - mu) / sig

        mu2, sig2 = mu[shuf_half_idx_ten], sig[shuf_half_idx_ten]
        mu_mix  = mu  * lamda_beta + mu2  * (1-lamda_beta)
        sig_mix = sig * lamda_beta + sig2 * (1-lamda_beta)

        x = torch.cat(
            [x[:-mix_size], x_normed*sig_mix + mu_mix],
            dim=0)
        
        if len(x_shape) == 2:
            x = x.squeeze(1)
        elif len(x_shape) == 4:
            x = x.unsqueeze(1)

        return x
    
class ECAPA_DBTDNN(torch.nn.Module):
    """An implementation of the speaker embedding model in a paper.
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143).

    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    activation : torch class
        A class for constructing the activation layers.
    channels : list of ints
        Output channels for TDNN/SERes2Net layer.
    kernel_sizes : list of ints
        List of kernel sizes for each layer.
    dilations : list of ints
        List of dilations for kernels in each layer.
    lin_neurons : int
        Number of neurons in linear layers.
    groups : list of ints
        List of groups for kernels in each layer.

    Example
    -------
    >>> input_feats = torch.rand([5, 120, 80])
    >>> compute_embedding = ECAPA_TDNN(80, lin_neurons=192)
    >>> outputs = compute_embedding(input_feats)
    >>> outputs.shape
    torch.Size([5, 1, 192])
    """

    def __init__(
            self, input_dim, num_classes, embedding_size=192, activation=torch.nn.ReLU,
            input_norm='', filter=None, sr=16000, feat_dim=80, exp=False, filter_fix=False,
            win_length=int(0.025*16000), nfft=512, stretch_ratio=[1.0],
            init_weight='mel', scale=0.2, weight_p=0.1, weight_norm='max',
            mask='None', mask_len=[5, 20],
            channels=[512, 512, 512, 512, 1536],
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            dropouts=[0, 0, 0], block_size=5, linear_step=0,
            attention_channels=128,
            res2net_scale=8,
            se_channels=128,
            global_context=True,
            groups=[1, 1, 1, 1, 1], **kwargs):

        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        
        input_mask = []
        filter_layer = get_filter_layer(filter=filter, input_dim=input_dim, sr=sr, feat_dim=feat_dim,
                                        exp=exp, filter_fix=filter_fix,
                                        stretch_ratio=stretch_ratio, win_length=win_length, nfft=nfft)
        if filter_layer != None:
            input_mask.append(filter_layer)
        norm_layer = get_input_norm(input_norm, input_dim=input_dim)
        if norm_layer != None:
            input_mask.append(norm_layer)
        mask_layer = get_mask_layer(mask=mask, mask_len=mask_len, input_dim=input_dim,
                                    init_weight=init_weight, weight_p=weight_p,
                                    scale=scale, weight_norm=weight_norm)
        if mask_layer != None:
            input_mask.append(mask_layer)
        self.input_mask = nn.Sequential(*input_mask)

        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.channels = channels
        self.blocks = nn.ModuleList()

        # The initial TDNN layer
        self.blocks.append(
            TDNNBlock(
                input_dim,
                channels[0],
                kernel_sizes[0],
                dilations[0],
                activation,
                groups[0],
            )
        )

        # SE-Res2Net layers
        for i in range(1, len(channels) - 1):
            self.blocks.append(
                nn.Sequential(
                    SERes2NetBlock(
                        channels[i - 1],
                        channels[i],
                        res2net_scale=res2net_scale,
                        se_channels=se_channels,
                        kernel_size=kernel_sizes[i],
                        dilation=dilations[i],
                        activation=activation,
                        groups=groups[i],
                    ),
                    DropBlock1d(drop_prob=dropouts[i-1], block_size=block_size, linear_step=linear_step),
                )
            )

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(
            channels[-1],
            channels[-1],
            kernel_sizes[-1],
            dilations[-1],
            activation,
            groups=groups[-1],
        )

        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            channels[-1],
            attention_channels=attention_channels,
            global_context=global_context,
        )
        self.asp_bn = BatchNorm1d(input_size=channels[-1] * 2)

        # Final linear transformation
        # self.fc = Conv1d(
        #     in_channels=channels[-1] * 2,
        #     out_channels=embedding_size,
        #     kernel_size=1,)

        self.fc = nn.Linear(channels[-1] * 2, embedding_size)

        self.classifier = Classifier(
            input_size=embedding_size, lin_neurons=embedding_size, out_neurons=num_classes)

    def forward(self, x, lengths=None, last=False,
                freeze=False):
        """Returns the embedding vector.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        """
        if freeze:
            with torch.no_grad():
                x = self.input_mask(x)
                if len(x.shape) == 4:
                    x = x.squeeze(1).float()
                x = x.transpose(1, 2)

                xl = []
                for layer in self.blocks:
                    try:
                        x = layer(x, lengths=lengths)
                    except TypeError:
                        x = layer(x)
                    xl.append(x)
                    
                x = torch.cat(xl[1:], dim=1)
                x = self.mfa(x)
                x = self.asp(x, lengths=lengths)
                x = self.asp_bn(x)
                embeddings = self.fc(x)
        else:
            # Minimize transpose for efficiency
            x = self.input_mask(x)
            # if proser != None and layer_mix == 1:
            #     x = self.mixup(x, proser, lamda_beta)
            if len(x.shape) == 4:
                x = x.squeeze(1).float()
            x = x.transpose(1, 2)

            xl = []
            for layer in self.blocks:
                try:
                    x = layer(x, lengths=lengths)
                except TypeError:
                    x = layer(x)
                xl.append(x)

            # Multi-layer feature aggregation
            x = torch.cat(xl[1:], dim=1)
            x = self.mfa(x)

            # Attentive Statistical Pooling
            x = self.asp(x, lengths=lengths)
            x = self.asp_bn(x)

            # Final linear transformation
            embeddings = self.fc(x)
            # embeddings = x.transpose(1, 2).contiguous()

        logits = self.classifier(embeddings)

        return logits, embeddings

    def get_embedding_dim(self):
        return self.embedding_size

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for pp in list(self.parameters()):
            if pp.requires_grad: # only using the parameter that require the gradient
                grads.append(pp.grad.view(-1))
        return torch.cat(grads)


class Classifier(torch.nn.Module):
    """This class implements the cosine similarity on the top of features.

    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of classes.

    Example
    -------
    >>> classify = Classifier(input_size=2, lin_neurons=2, out_neurons=2)
    >>> outputs = torch.tensor([ [1., -1.], [-9., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> outupts = outputs.unsqueeze(1)
    >>> cos = classify(outputs)
    >>> (cos < -1.0).long().sum()
    tensor(0)
    >>> (cos > 1.0).long().sum()
    tensor(0)
    """

    def __init__(self, input_size, lin_blocks=0, lin_neurons=192, out_neurons=1211):

        super().__init__()
        self.blocks = nn.ModuleList()

        for block_index in range(lin_blocks):
            self.blocks.extend(
                [
                    _BatchNorm1d(input_size=input_size),
                    Linear(input_size=input_size, n_neurons=lin_neurons),
                ]
            )
            input_size = lin_neurons

        # Final Layer
        self.weight = nn.Parameter(
            torch.FloatTensor(out_neurons, input_size)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        """Returns the output probabilities over speakers.

        Arguments
        ---------
        x : torch.Tensor
            Torch tensor.
        """
        for layer in self.blocks:
            x = layer(x)

        # Need to be normalized
        x = F.linear(F.normalize(x), F.normalize(self.weight))

        return x  # .unsqueeze(1)
