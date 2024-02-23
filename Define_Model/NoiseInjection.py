#!/usr/bin/env python
# encoding: utf-8
'''
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: VS Code
@File: NoiseInjection.py
@Time: 2024/02/19 20:40
@Overview: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
import numpy as np

from Define_Model.Pooling import SelfAttentionPooling_v2

class Dropout1d(nn.Module):
    def __init__(self, drop_prob, linear_step=0):
        super(Dropout1d, self).__init__()
        self.drop_prob = drop_prob
        self.linear_step = linear_step
        self.this_step = 0

    def forward(self, x):
        if self.training and self.drop_prob > 0:
            drop_prob = self.drop_prob
            if self.linear_step > 0:
                
                if self.this_step <= self.linear_step:
                    drop_prob = self.drop_prob * self.this_step / self.linear_step
                    self.this_step += 1

            x = F.dropout1d(x, p=drop_prob)
        return x

class MagnitudeDropout1d(nn.Module):
    def __init__(self, drop_prob, linear_step=0):
        super(MagnitudeDropout1d, self).__init__()
        self.drop_prob = drop_prob
        self.linear_step = linear_step
        self.this_step = 0

    def forward(self, x):
        if self.training and self.drop_prob > 0:
            freq_mag = x.abs().mean(dim=-1, keepdim=True)
            freq_mag = freq_mag / freq_mag.mean(dim=1, keepdim=True)

            drop_prob = self.drop_prob
            if self.linear_step > 0:
                if self.this_step <= self.linear_step:
                    drop_prob = self.drop_prob * self.this_step / self.linear_step
                    self.this_step += 1

            gamma = 1 - freq_mag * drop_prob
            mask = torch.bernoulli(gamma)
            x = x * mask * mask.numel() / mask.sum()
            # x = F.dropout1d(x, p=drop_prob)
        return x

class DropBlock1d(nn.Module):
    """DropBlock layer.

    Arguments
    ----------
    drop_prob : float
        The drop probability.
    block_size : int
        The size of the block.
    """

    def __init__(self, drop_prob, block_size=5, linear_step=0):
        super(DropBlock1d, self).__init__()
        self.drop_prob = 1 - drop_prob
        self.block_size = block_size
        self.linear_step = linear_step
        self.this_step = 0
    
    def forward(self, x):
        if self.training and self.drop_prob > 0:
            if self.block_size <= 0:
                raise ValueError("Block size should be greater than 0")
            
            # batch_size, channels, time = x.size()            
            gamma = self._compute_gamma(x)

            # sample mask
            # mask = torch.bernoulli(torch.ones_like(x) * gamma)
            mask = torch.bernoulli(torch.ones(x.shape[0], 1, *x.shape[2:]) * gamma)

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

        block_mask = 1 - block_mask #.squeeze(1)

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

    def __init__(self, drop_prob, linear_step=0):
        super(DropAttention1d, self).__init__()
        self.drop_prob = drop_prob
        self.linear_step = linear_step
        self.this_step = 0
        
    def forward(self, s):
        if self.training and self.drop_prob > 0:
            # T = torch.zeros_like(s)
            # T.scatter_(dim=1, index=torch.topk(s, k=int(s.shape[1]*self.drop_prob), dim=1)[1], src=torch.ones_like(s))
            T = (s - s.min()) / (s.max() - s.min())
            T = T * self.drop_prob
            # sample mask
            mask = 1 - torch.bernoulli(T)

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
    """NoiseInject layer.

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
            
            ones_prob = (torch.ones(x.shape[0]).uniform_(0, 1) < 0.8).float()
            ones_prob = ones_prob.to(x.device)
            ones_prob = ones_prob.reshape(-1, 1, 1)

            if len(x.shape) == 4:
                ones_prob = ones_prob.unsqueeze(1)

            if self.drop_prob > 0:
                # torch.randn(x.shape)
                mul_noise = 2 * torch.cuda.FloatTensor(x.shape).uniform_() - 1
                mul_noise = self.drop_prob * np.random.beta(2, 5) * mul_noise * ones_prob

                # mul_noise = mul_noise.to(x.device)
                x = (1 + mul_noise) * x

            if self.add_prob > 0:
                add_noise = torch.cuda.FloatTensor(x.shape).normal_()
                add_noise = self.add_prob * np.random.beta(2, 5) * add_noise * ones_prob
                
                # add_noise = add_noise.to(x.device)
                x = x + add_noise
            
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
                r = stats.burr12.rvs(2, 1, size=(x.shape[0], x.shape[1]))
                ones = torch.bernoulli(torch.ones(r.shape) * 0.3).numpy()
                r = np.where(ones == 1, 1, r)
                mul_noise = torch.tensor(r).float().unsqueeze(2)
                mul_noise = mul_noise.to(x.device)

                x = (1 + self.drop_prob * np.random.beta(2, 5) * (mul_noise-1)) * x

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
        self.batch_prob = 0.8 if len(drop_prob) < 3 else drop_prob[2]
    
    def forward(self, x):
        if self.training:
            time_attention = x.abs().mean(dim=1, keepdim=True)
            time_attention = time_attention / time_attention.max()

            ones_prob = (torch.ones(x.shape[0],1,1).uniform_(0, 1) < self.batch_prob).float()
            ones_prob = ones_prob.to(x.device)

            if self.drop_prob > 0:
                # torch.randn(x.shape)
                mul_noise = 2 * torch.cuda.FloatTensor(x.shape).uniform_() - 1
                mul_noise = self.drop_prob * np.random.beta(2, 5) * mul_noise * time_attention * ones_prob

                # mul_noise = mul_noise.to(x.device)
                x = (1 + mul_noise) * x

            # ones_prob = (torch.ones(x.shape[0],1,1).uniform_(0, 1) < 0.7).float()
            # ones_prob = ones_prob.to(x.device)
            if self.add_prob > 0:
                add_noise = torch.cuda.FloatTensor(x.shape).normal_() 
                add_noise = self.add_prob * np.random.beta(2, 5) * add_noise * time_attention * ones_prob
                # torch.randn(x.shape)
                # add_noise = add_noise.to(x.device)
                x = x + add_noise
            
            return x
        else:
            return x
        
    def __repr__(self):
        return "AttentionNoiseInject(mul_prob={}, add_prob={}, batch_prob={})".format(self.drop_prob, self.add_prob, self.batch_prob)


class MagCauchyNoiseInject(nn.Module):
    """AttentionNoiseInject layer.

    Arguments
    ----------
    drop_prob : float
        The drop probability.
    block_size : int
        The size of the block.
    """

    def __init__(self, drop_prob=[1, 1], linear_step=0):
        super(MagCauchyNoiseInject, self).__init__()
        self.drop_prob = drop_prob[0]
        self.add_prob = drop_prob[1]
        self.linear_step = linear_step
        self.this_step = 0
        self.batch_prob = 0.8 if len(drop_prob) < 3 else drop_prob[2]
    
    def forward(self, x):
        if self.training:
            # time magnitude
            time_attention = x.abs().mean(dim=1, keepdim=True)
            time_attention = time_attention / time_attention.max()

            ones_prob = (torch.ones(x.shape[0],1,1).uniform_(0, 1) < self.batch_prob).float()
            ones_prob = ones_prob.to(x.device)

            if self.drop_prob > 0:
                # torch.randn(x.shape)
                # mul_noise = 2 * torch.cuda.FloatTensor(x.shape).uniform_() - 1
                r = stats.skewcauchy.rvs(-0.084, 0, 0.176, size=(x.shape[0], x.shape[1]))
                # r = stats.norm.rvs(0, 1/3, size=(x.shape[0], x.shape[1]))
                
                # 0.3 to set zeros
                ones = torch.bernoulli(torch.ones(r.shape) * 0.3).numpy()
                r = np.where(ones == 1, 1, r)
                
                mul_noise = torch.tensor(r).float().unsqueeze(2)
                mul_noise = mul_noise.to(x.device)
                mul_noise = self.drop_prob * np.random.beta(2, 5) * mul_noise * time_attention * ones_prob
                
                x = (1 + mul_noise) * x

            # ones_prob = (torch.ones(x.shape[0],1,1).uniform_(0, 1) < 0.7).float()
            # ones_prob = ones_prob.to(x.device)
            if self.add_prob > 0:
                add_noise = torch.cuda.FloatTensor(x.shape).normal_() 
                add_noise = self.add_prob * np.random.beta(2, 5) * add_noise * time_attention * ones_prob
                # torch.randn(x.shape)
                # add_noise = add_noise.to(x.device)
                x = x + add_noise
            
            return x
        else:
            return x
        
    def __repr__(self):
        return "MagCauchyNoiseInject(mul_prob={}, add_prob={}, ones_prob2=0.8)".format(self.drop_prob, self.add_prob)

