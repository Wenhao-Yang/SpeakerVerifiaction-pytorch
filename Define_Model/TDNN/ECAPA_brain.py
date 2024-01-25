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
import torch  # noqa: F401
import numpy as np
from scipy.stats import burr12
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.nnet.CNN import Conv1d as _Conv1d
from speechbrain.nnet.normalization import BatchNorm1d as _BatchNorm1d
from speechbrain.nnet.linear import Linear

from Define_Model.model import get_filter_layer, get_input_norm, get_mask_layer


# Skip transpose as much as possible for efficiency
class Conv1d(_Conv1d):
    """1D convolution. Skip transpose is used to improve efficiency."""

    def __init__(self, *args, **kwargs):
        super().__init__(skip_transpose=True, *args, **kwargs)


class BatchNorm1d(_BatchNorm1d):
    """1D batch normalization. Skip transpose is used to improve efficiency."""

    def __init__(self, *args, **kwargs):
        super().__init__(skip_transpose=True, *args, **kwargs)


class DropBlock1d(nn.Module):
    """DropBlock layer.

    Arguments
    ----------
    drop_prob : float
        The drop probability.
    block_size : int
        The size of the block.
    """

    def __init__(self, drop_prob, block_size, linear_step=0):
        super(DropBlock1d, self).__init__()
        self.drop_prob = 1 - drop_prob
        self.drop_prob = block_size
        self.linear_step = linear_step
        self.this_step = 0
    
    def forward(self, x):
        if self.training and self.drop_prob > 0:
            if self.block_size <= 0:
                raise ValueError("Block size should be greater than 0")
            
            # batch_size, channels, time = x.size()            
            gamma = self._compute_gamma(x)

            # sample mask
            mask = torch.bernoulli(torch.ones_like(x) * gamma)

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out
        else:
            return x

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool1d(mask,
                                  kernel_size=self.block_size,
                                  stride=1,
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        # linear drop of dropout probability
        drop_prob = self.drop_prob
        if self.linear_step > 0:
            
            if self.this_step <= self.linear_step:
                drop_prob = 1 - (1 - self.drop_prob) * self.this_step / self.linear_step
                self.this_step += 1

        invalid = (1 - drop_prob) / self.block_size
        valid = (x.shape[-1]) / (x.shape[-1] - self.block_size + 1)
        
        return invalid * valid
    
    def __repr__(self):
        return "DropBlock1d(drop_prob={}, drop_prob={}, linear_step={})".format(self.drop_prob, self.drop_prob,
                                                                self.linear_step)


class DropAttention1d(nn.Module):
    """DropBlock layer.

    Arguments
    ----------
    drop_prob : float
        The drop probability.
    block_size : int
        The size of the block.
    """

    def __init__(self, drop_prob):
        super(DropAttention1d, self).__init__()
        self.p = drop_prob
        
    def forward(self, s):
        if self.training and self.p > 0:
            # T = torch.zeros_like(s)
            # T.scatter_(dim=1, index=torch.topk(s, k=int(s.shape[1]*self.drop_prob), dim=1)[1], src=torch.ones_like(s))
            T = s * 0.2 + 0.9 - self.drop_prob
            # sample mask
            mask = torch.bernoulli(T)

            # place mask on input device
            mask = mask.to(s.device)
            # scale output
            out = mask * mask.numel() / mask.sum()

            return s * out
        else:
            return s


class AttentionDrop1d(nn.Module):
    """AttentionDrop channel layer.

    Arguments
    ----------
    drop_prob : float
        The drop probability.
    block_size : int
        The size of the block.
    """

    def __init__(self, drop_prob):
        super(AttentionDrop1d, self).__init__()
        self.p = drop_prob
        
    def forward(self, s):
        if self.training and self.p > 0:
            # T = torch.zeros_like(s)
            # T.scatter_(dim=1, index=torch.topk(s, k=int(s.shape[1]*self.drop_prob), dim=1)[1], src=torch.ones_like(s))
            T = 1 - (s - s.min()) / (s.max() - s.min())
            mask = torch.bernoulli(torch.ones_like(s) * self.p).to(s.device)
            mask_zero = 1 - mask
            
            mask = s * mask + mask_zero
            # scale output
            out = mask * mask.numel() / mask.sum()

            return s * out
        else:
            return s


class NoiseInject(nn.Module):
    """DropBlock layer.

    Arguments
    ----------
    drop_prob : float
        The drop probability.
    block_size : int
        The size of the block.
    """

    def __init__(self, drop_prob=[0.2, 0.4], linear_step=0):
        super(NoiseInject, self).__init__()
        self.drop_prob = drop_prob[0]
        self.add_prob = drop_prob[1]
        self.linear_step = linear_step
        self.this_step = 0
    
    def forward(self, x):
        if self.training:
            if self.drop_prob > 0:
                # torch.randn(x.shape)
                mul_noise = 2 * torch.cuda.FloatTensor(x.shape).uniform_() - 1
                # mul_noise = mul_noise.to(x.device)
                x = (1 + self.drop_prob * np.random.beta(2, 5) * mul_noise ) * x

            if self.add_prob > 0:
                add_noise = torch.cuda.FloatTensor(x.shape).normal_() 
                # torch.randn(x.shape)
                # add_noise = add_noise.to(x.device)
                x = x + self.add_prob * np.random.beta(2, 5) * add_noise
            
            return x
        else:
            return x
        
    def __repr__(self):
        return "NoiseInject(mul_prob={}, add_prob={})".format(self.drop_prob, self.add_prob)


class RadioNoiseInject(nn.Module):
    """DropBlock layer.

    Arguments
    ----------
    drop_prob : float
        The drop probability.
    block_size : int
        The size of the block.
    """

    def __init__(self, drop_prob=[0.2, 0.4], linear_step=0):
        super(RadioNoiseInject, self).__init__()
        self.drop_prob = drop_prob[0]
        self.add_prob = drop_prob[1]
        self.linear_step = linear_step
        self.this_step = 0
    
    def forward(self, x):
        if self.training:
            time_attention = x.abs().mean(dim=1, keepdim=True)
            time_attention = time_attention / time_attention.max()
            
            if self.drop_prob > 0:
                # torch.randn(x.shape)
                r = burr12.rvs(2, 1, size=(x.shape[0], x.shape[1]))
                ones = torch.bernoulli(torch.ones(r.shape) * 0.3).numpy()
                r = np.where(ones == 1, 1, r)
                mul_noise = torch.tensor(r).float().unsqueeze(2)
                mul_noise = mul_noise.to(x.device)

                x = (1 + np.random.beta(2, 5) * (mul_noise-1)) * x

            if self.add_prob > 0:
                add_noise = torch.cuda.FloatTensor(x.shape).normal_() 
                # torch.randn(x.shape)
                # add_noise = add_noise.to(x.device)
                x = x + self.add_prob * np.random.beta(2, 5) * add_noise * time_attention
            
            return x
        else:
            return x
        
    def __repr__(self):
        return "RadioNoiseInject(mul_prob={}, add_prob={})".format(self.drop_prob, self.add_prob)



class ImpulseNoiseInject(nn.Module):
    """DropBlock layer.

    Arguments
    ----------
    drop_prob : float
        The drop probability.
    block_size : int
        The size of the block.
    """

    def __init__(self, drop_prob=[0.2, 0.4, 0.5, 0.5], linear_step=0):
        super(ImpulseNoiseInject, self).__init__()
        self.drop_prob = drop_prob[0]
        self.mul_impulse = drop_prob[2]

        self.add_prob = drop_prob[1]
        self.add_impulse = drop_prob[3]
        self.linear_step = linear_step
        self.this_step = 0
    
    def forward(self, x):
        if self.training:
            if self.drop_prob > 0:
                # torch.randn(x.shape)
                mul_noise = 2 * torch.cuda.FloatTensor(x.shape).uniform_() - 1
                mask = torch.bernoulli(torch.ones(x.shape) * self.mul_impulse)
                mask = mask.to(x.device)

                mul_noise = mask * mul_noise
                # mul_noise = mul_noise.to(x.device)
                x = (1 + self.drop_prob * np.random.beta(2, 5) * mul_noise ) * x

            if self.add_prob > 0:
                add_noise = torch.cuda.FloatTensor(x.shape).normal_() 
                mask = torch.bernoulli(torch.ones(x.shape) * self.add_impulse)
                mask = mask.to(x.device)

                add_noise = mask * add_noise
                # add_noise = add_noise.to(x.device)
                x = x + self.add_prob * np.random.beta(2, 5) * add_noise
            
            return x
        else:
            return x
        

class AttentionNoiseInject(nn.Module):
    """AttentionNoiseInject layer.

    Arguments
    ----------
    drop_prob : float
        The drop probability.
    block_size : int
        The size of the block.
    """

    def __init__(self, drop_prob=[0.2, 0.4], linear_step=0):
        super(AttentionNoiseInject, self).__init__()
        self.drop_prob = drop_prob[0]
        self.add_prob = drop_prob[1]
        self.linear_step = linear_step
        self.this_step = 0
    
    def forward(self, x):
        if self.training:
            time_attention = x.abs().mean(dim=1, keepdim=True)
            time_attention = time_attention / time_attention.max()

            if self.drop_prob > 0:
                # torch.randn(x.shape)
                mul_noise = 2 * torch.cuda.FloatTensor(x.shape).uniform_() - 1
                # mul_noise = mul_noise.to(x.device)
                x = (1 + self.drop_prob * np.random.beta(2, 5) * mul_noise * time_attention ) * x

            if self.add_prob > 0:
                add_noise = torch.cuda.FloatTensor(x.shape).normal_() 
                # torch.randn(x.shape)
                # add_noise = add_noise.to(x.device)
                x = x + self.add_prob * np.random.beta(2, 5) * add_noise * time_attention
            
            return x
        else:
            return x
        
    def __repr__(self):
        return "AttentionNoiseInject(mul_prob={}, add_prob={})".format(self.drop_prob, self.add_prob)


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
        self.norm = BatchNorm1d(input_size=out_channels)

    def forward(self, x):
        """ Processes the input tensor x and returns an output tensor."""
        return self.norm(self.activation(self.conv(x)))


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

        if 'vanilla' in dropout_type:
            self.drop = torch.nn.Dropout1d(dropout_p)
        elif 'attention' in dropout_type:
            self.drop = DropAttention1d(dropout_p)
        elif 'attendrop' in dropout_type:
            self.drop = AttentionDrop1d(dropout_p)
        else:
            self.drop = None

    def forward(self, x, lengths=None):
        """ Processes the input tensor x and returns an output tensor."""
        L = x.shape[-1]
        if lengths is not None:
            mask = length_to_mask(lengths * L, max_len=L, device=x.device)
            mask = mask.unsqueeze(1)
            total = mask.sum(dim=2, keepdim=True)
            s = (x * mask).sum(dim=2, keepdim=True) / total
        else:
            s = x.mean(dim=2, keepdim=True)

        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))
        
        # linear create dropout
        if self.drop != None:
            if self.linear_step > 0:
                if self.this_step <= self.linear_step:
                    self.drop.p = self.dropout_p * self.this_step / self.linear_step
                    self.this_step += 1         
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
        if self.global_context:
            # torch.std is unstable for backward computation
            # https://github.com/pytorch/pytorch/issues/4320
            total = mask.sum(dim=2, keepdim=True).float()
            mean, std = _compute_statistics(x, mask / total)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
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
        pooled_stats = pooled_stats  # .unsqueeze(2)

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
        groups=1,
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
            dropouts=[0, 0, 0], dropout_type='vanilla', linear_step=0,
            attention_channels=128,
            res2net_scale=8,
            se_channels=128,
            global_context=True,
            groups=[1, 1, 1, 1, 1], **kwargs):

        super().__init__()
        self.embedding_size = embedding_size
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
        self.blocks = nn.ModuleList()

        # The initial TDNN layer
        tdnn_layer1 = [
            TDNNBlock(
                    input_dim,
                    channels[0],
                    kernel_sizes[0],
                    dilations[0],
                    activation,
                    groups[0],) 
            ]
        
        if 'attenoise' in dropout_type:
            if isinstance(self.dropouts[0], float):
                tdnn_layer1.append(AttentionNoiseInject(drop_prob=self.dropouts[0:2]))
            else:
                tdnn_layer1.append(AttentionNoiseInject(drop_prob=self.dropouts[0]))
        elif 'noiseinject' in dropout_type:
            if isinstance(self.dropouts[0], float):
                tdnn_layer1.append(NoiseInject(drop_prob=self.dropouts[0:2]))
            else:
                tdnn_layer1.append(NoiseInject(drop_prob=self.dropouts[0]))
        elif 'radionoise' in dropout_type:
            tdnn_layer1.append(RadioNoiseInject(drop_prob=self.dropouts[0]))

        elif 'vanilla' in dropout_type and self.dropouts[0] > 0:
            tdnn_layer1.append(nn.Dropout1d(self.dropouts[0]))

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
                    dropout_type=dropout_type, dropout_p=self.dropouts[i],
                    linear_step=linear_step,
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
