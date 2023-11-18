#!/usr/bin/env python
# encoding: utf-8
'''
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: VS Code
@File: Parallel.py
@Time: 2023/11/18 14:21
@Overview: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch.nn.functional as F
from torch.autograd import Variable

def sample_gumbel(shape, eps=1e-20):
    U = torch.cuda.FloatTensor(shape).uniform_()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature = 5):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


class Parallel(nn.Module):
    def __init__(self, model, layers):
        super(Parallel, self).__init__()

        self.model = copy.deepcopy(model)
        self.agent_model = copy.deepcopy(model)

        for n,p in self.agent_model.named_modules():
            p.requires_grad = False

        # print(self.model.classifier.weight.shape)
        self.agent_model.classifier = nn.Linear(self.model.classifier.weight.shape[1],
                                                layers)
        
        for n,p in self.model.named_modules():
            p.requires_grad = False

        self.blocks = copy.deepcopy(model.blocks)
        self.mfa    = copy.deepcopy(model.mfa)
        self.asp    = copy.deepcopy(model.asp)
        self.asp_bn = copy.deepcopy(model.asp_bn)
        self.fc     = copy.deepcopy(model.fc)
        self.classifier = copy.deepcopy(model.classifier)

    def forward(self, x, policy=None):
        policy, _ = self.agent_model(x)
        policy = gumbel_softmax(policy)

        x = self.model.input_mask(x)

        if len(x.shape) == 4:
            x = x.squeeze(1).float()
        x = x.transpose(1, 2)

        xl = []
        if policy is not None:
            for i, layer in enumerate(self.blocks):
                action = policy[:, i].contiguous()
                action_mask = action.float().view(-1,1,1)

                original_x = layer(x)
                fine_x = self.blocks[i](x)

                x = original_x*(1-action_mask) + fine_x*action_mask
                
                xl.append(x)
        else:
            for i, layer in enumerate(self.blocks):
                x = layer(x)
                xl.append(x)

        # Multi-layer feature aggregation
        x = torch.cat(xl[1:], dim=1)
        x = self.mfa(x)

        # Attentive Statistical Pooling
        x = self.asp(x, lengths=None)
        x = self.asp_bn(x)

        # Final linear transformation
        embeddings = self.fc(x)
        logits = self.classifier(embeddings)

        return logits, embeddings