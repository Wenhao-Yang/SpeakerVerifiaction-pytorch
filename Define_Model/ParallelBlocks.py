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

class globalk(nn.Module):
    def __init__(self, k=3, lambda1=0.25, lambda2=0.25,
                 lambda3=0, positive_slope=0.1) -> None:
        super().__init__()
        self.k = k
        self.loss = nn.CrossEntropyLoss()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.positive_slope = positive_slope
        
    def forward(self, policy):
        
        policy = policy.view(policy.size(0), -1, 2)
        policy_one = gumbel_softmax(policy)[:, :, 1]
        
        if isinstance(self.k, float):
            lk = torch.pow(policy_one.sum(dim=1)-self.k, 2).mean()
        elif isinstance(self.k, list):
            fine_layer = policy_one.sum(dim=1)
            # lk = torch.abs((fine_layer-self.k[0])*(fine_layer-self.k[1])).mean()
            lk = (self.positive_slope * (fine_layer-self.k[0]) + (fine_layer-self.k[1]).clamp_min(0)**2).mean()
        # print(policy.view(-1, 2).shape, policy_one.view(-1, 1).shape)
        le = self.loss(policy.view(-1, 2), policy_one.view(-1).detach().long())    
        
        return policy_one, self.lambda1 * lk + self.lambda2*le + self.lambda3 * policy.std(dim=0).mean()
        


class ConstantPolicy(nn.Module):
    def __init__(self, layers):
        super(ConstantPolicy, self).__init__()

        self.policy = nn.Parameter(torch.randn(layers))

    def forward(self, x):
        this_policy = self.policy.unsqueeze(0)

        return this_policy.repeat(x.shape[0], 1), ''   

class Parallel(nn.Module):
    def __init__(self, model, layers, 
                 total_layer=9, 
                 agent_model=None,
                 lambda3=0, positive_slope=0.1,
                 k=3):
        super(Parallel, self).__init__()

        self.model = copy.deepcopy(model)
        self.layers = layers
        self.k = k

        if agent_model == None:
            self.agent_model_training = False
            self.agent_model = copy.deepcopy(model)
            for n,p in self.agent_model.named_modules():
                p.requires_grad = False
                
            self.agent_model.input_mask.__delitem__(0)
            self.agent_model.input_mask.__delitem__(0)
            self.agent_classifier = nn.Linear(self.model.classifier.weight.shape[1],
                                                layers)
        else:
            self.agent_model_training = True
            self.agent_classifier = agent_model

        # print(self.model.classifier.weight.shape)
        # self.agent_model.classifier = nn.Linear(self.model.classifier.weight.shape[1],
        #                                         layers)
        for n,p in self.model.named_modules():
            p.requires_grad = False

        self.blocks = copy.deepcopy(model.blocks)
        self.mfa    = copy.deepcopy(model.mfa)
        self.asp    = copy.deepcopy(model.asp)
        self.asp_bn = copy.deepcopy(model.asp_bn)
        self.fc     = copy.deepcopy(model.fc)
        self.classifier = copy.deepcopy(model.classifier)
        self.model_layer = total_layer
        
        self.globalk_loss = globalk(k=k, lambda3=lambda3, positive_slope=positive_slope)

    def policy(self, x):

        x = self.model.input_mask(x)
        
        if self.agent_model_training:
            policy, _ = self.agent_classifier(x)
        else:
            with torch.no_grad():
                _, emb = self.agent_model(x)
            policy = self.agent_classifier(emb)
            
        policy_one, layer_loss = self.globalk_loss(policy)
        return policy_one, layer_loss, x

    def forward(self, x, policy=None):
        
        policy, layer_loss, x = self.policy(x)

        if len(x.shape) == 4:
            x = x.squeeze(1).float()
        x = x.transpose(1, 2)

        xl = []
        if policy is not None:
            for i, layer in enumerate(self.blocks):
                action = policy[:, i].contiguous()
                action_mask = action.float().view(-1,1,1)

                original_x = self.model.blocks[i](x)
                fine_x = layer(x)

                x = original_x*(1-action_mask) + fine_x*action_mask
                
                xl.append(x)
                
            # Multi-layer feature aggregation
            x = torch.cat(xl[1:], dim=1)
            
            for j, (layer, layer_o) in enumerate([(self.mfa,self.model.mfa),
                                                (self.asp, self.model.asp), 
                                                (self.asp_bn, self.model.asp_bn), 
                                                (self.fc, self.model.fc), 
                                                (self.classifier, self.model.classifier)]):
                
                # print(j, layer, x.shape, end=' ')
                if policy.shape[1] > len(self.blocks)+ j:
                    action = policy[:, len(self.blocks)+j].contiguous()
                    action_mask = action.float().view(-1,1)
                    
                    original_x = layer_o(x)
                    fine_x = layer(x)
                    
                    if len(fine_x.shape) == 3:
                        action_mask = action_mask.unsqueeze(1)

                    x = original_x*(1-action_mask) + fine_x*action_mask  
                else:
                    x = layer(x)
                    
                # print(x.shape)
                
                if j == 3:
                    embeddings = x

            logits = x
            
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

        return (logits, layer_loss), embeddings
    

class Adapter(nn.Module):
    def __init__(self, model, layers=9, input_dim=80, activation=torch.nn.ReLU,
                k=3, scale=[32, 16, 16, 16, 16],
                channels=[512, 512, 512, 512, 1536],
                kernel_sizes=[5, 3, 3, 3, 1],
                dilations=[1, 2, 3, 4, 1],
                attention_channels=128,
                adapter_rate=0,
                res2net_scale=8,
                se_channels=128,embedding_size=192,
                global_context=True, adapter_type='append',
                groups=[1, 1, 1, 1, 1]
        ):
        super(Adapter, self).__init__()
        from Define_Model.TDNN.ECAPA_brain import AttentiveStatisticsPooling, BatchNorm1d
        from Define_Model.TDNN.ECAPA_brain import TDNNBottleBlock, SERes2NetBottleblock, TDNNBlock
        
        self.model = copy.deepcopy(model)
        self.layers = layers
        self.channels = channels
        self.scale = scale
        self.adapter_rate = adapter_rate

        self.k = k
        self.adapter_type = adapter_type
        if self.adapter_type == 'append':
            self.adapter_forward = self.append_forward
        elif self.adapter_type == 'parallel':
            self.adapter_forward = self.parallel_forward
        elif self.adapter_type == 'concat':
            self.adapter_forward = self.concat_forward

        self.blocks = nn.ModuleList()

        # The initial TDNN layer
        self.blocks.append(
            TDNNBottleBlock(
                input_dim,
                scale[0],
                channels[0],
                kernel_sizes[0],
                dilations[0],
                activation,
                groups[0],
            )
        )

        # SE-Res2Net layers
        for i in range(1, len(self.model.channels) - 1):
            self.blocks.append(
                SERes2NetBottleblock(
                    channels[i - 1],
                    scale[1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    activation=activation,
                    groups=groups[i],
                )
            )

        # self.blocks = copy.deepcopy(model.blocks)
        if layers > 5:
            self.mfa    = TDNNBottleBlock(
                        channels[-1],
                        scale[2],
                        channels[-1],
                        kernel_sizes[-1],
                        dilations[-1],
                        activation,
                        groups=groups[-1],
                        )

        if layers > 6:   
            self.asp    = AttentiveStatisticsPooling(channels[-1],
                attention_channels=scale[3])
        
        if layers > 7:
            self.asp_bn = BatchNorm1d(input_size=channels[-1] * 2)
        
        if layers > 8:
            self.fc     = nn.Sequential(
                nn.Linear(self.model.channels[-1] * 2, scale[4]),
                nn.ReLU(),
                nn.Linear(scale[4], embedding_size)
            )

        if layers > 9:
            self.classifier = copy.deepcopy(model.classifier)

        self.freeze()
        # self.model_layer = total_layer

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        
        x = self.model.input_mask(x)
        if len(x.shape) == 4:
            x = x.squeeze(1).float()
        x = x.transpose(1, 2)

        xl = []
        for i, layer in enumerate(self.blocks):
            # action = policy[:, i].contiguous()
            # action_mask = action.float().view(-1,1,1)
            x_o = x
            x = self.model.blocks[i](x_o)
            if self.layers > i+1 and self.scale[i] > 0:   
                x = self.adapter_forward(layer, x_o, x)
            # x = original_x + fine_x
            xl.append(x)
                
        # Multi-layer feature aggregation
        x_o = torch.cat(xl[1:], dim=1)
        
        x = self.model.mfa(x_o)
        if self.layers > 5 and self.scale[i] > 0:   
            x = self.adapter_forward(self.mfa, x_o, x)

        # Attentive Statistical Pooling
        x_o = x
        x = self.model.asp(x_o, lengths=None)
        if self.layers > 6:  
            x = self.adapter_forward(self.asp, x_o, x)

        x_o = x
        x = self.model.asp_bn(x_o)
        if self.layers > 7:  
            x = self.adapter_forward(self.asp_bn, x_o, x)

        # Final linear transformation
        x_o = x
        embeddings = self.model.fc(x_o)
        if self.layers > 8:  
            embeddings = self.adapter_forward(self.fc, x_o, embeddings)

        logits = self.model.classifier(embeddings)

        if self.layers > 9:  
            logits = self.adapter_forward(self.classifier, embeddings, logits)

        return logits, embeddings
    

    def append_forward(self, block, x_o, x):
        return block(x)

    def parallel_forward(self, block, x_o, x):
        if self.adapter_rate == 1:
            return x + block(x_o)
        elif self.adapter_rate> 0:
            return x * (1-self.adapter_rate) + block(x_o) * self.adapter_rate
        else:
            return x + block(x_o)
    
    def concat_forward(self, block, x_o, x):
        return torch.cat([x, block(x)], dim=1)