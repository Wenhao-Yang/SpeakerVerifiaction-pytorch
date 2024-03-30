#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: model.py
@Overview: The deep speaker model is not entirely the same as ResNet, as there are convolutional layers between blocks.
"""

import math

import torch
import torch.nn as nn
import yaml
from hyperpyyaml import load_hyperpyyaml


# Models
from Define_Model.CNN import AlexNet
from Define_Model.FilterLayer import RevGradLayer
from Define_Model.Pooling import get_encode_layer
from Define_Model.ResNet import LocalResNet, ResNet20, ThinResNet, RepeatResNet, ResNet, SimpleResNet, GradResNet, \
    TimeFreqResNet, MultiResNet
from Define_Model.Loss.SoftmaxLoss import AdditiveMarginLinear, SubMarginLinear, MarginLinearDummy
from Define_Model.TDNN.ARET import RET, RET_v2, RET_v3
from Define_Model.TDNN.DTDNN import DTDNN
from Define_Model.TDNN.ECAPA_TDNN import ECAPA_TDNN
from Define_Model.TDNN import ECAPA_brain
from Define_Model.TDNN.ETDNN import ETDNN_v4, ETDNN, ETDNN_v5
from Define_Model.TDNN.FTDNN import FTDNN
from Define_Model.TDNN.Slimmable import SlimmableTDNN
from Define_Model.TDNN.TDNN import TDNN_v2, TDNN_v4, TDNN_v5, TDNN_v6, MixTDNN_v5
from Define_Model.demucs_feature import Demucs
from tllib.modules.grl import WarmStartGradientReverseLayer
from tllib.alignment.cdan import RandomizedMultiLinearMap, MultiLinearMap
import torch.nn.functional as F


def get_layer_param(model):
    return sum([torch.numel(param) for param in model.parameters()])


class ReLU20(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU20, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'


# convert dict attribute to object attribute
class AttrDict(dict):
    """Dict as attribute trick.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, dict):
                self.__dict__[key] = AttrDict(value)
            elif isinstance(value, list):
                if isinstance(value[0], dict):
                    self.__dict__[key] = [AttrDict(item) for item in value]
                else:
                    self.__dict__[key] = value

    def yaml(self):
        """Convert object to yaml dict and return.
        """
        yaml_dict = {}
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, AttrDict):
                yaml_dict[key] = value.yaml()
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    new_l = []
                    for item in value:
                        new_l.append(item.yaml())
                    yaml_dict[key] = new_l
                else:
                    yaml_dict[key] = value
            else:
                yaml_dict[key] = value
        return yaml_dict

    def __repr__(self):
        """Print all variables.
        """
        ret_str = []
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, AttrDict):
                ret_str.append('{}:'.format(key))
                child_ret_str = value.__repr__().split('\n')
                for item in child_ret_str:
                    ret_str.append('    ' + item)
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    ret_str.append('{}:'.format(key))
                    for item in value:
                        # treat as AttrDict above
                        child_ret_str = item.__repr__().split('\n')
                        for item in child_ret_str:
                            ret_str.append('    ' + item)
                else:
                    ret_str.append('{}: {}'.format(key, value))
            else:
                ret_str.append('{}: {}'.format(key, value))
        return '\n'.join(ret_str)


class DomainDiscriminator(nn.Module):
    def __init__(self, input_size, num_classes,
                 hidden_size=0, num_logits=0,
                 warm_start=False, sigmoid=False,
                 max_iters=1000, pooling='',
                 mapping='none'):
        """
        discriminator with A gradient reversal layer.

        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super(DomainDiscriminator, self).__init__()
        self.warm_start = warm_start
        self.mapping = mapping
        
        if self.mapping == 'rand':
            self.map = RandomizedMultiLinearMap(features_dim=input_size,
                                                num_classes=num_logits,
                                                output_dim=input_size)
        elif self.mapping == 'linear':
            self.map = MultiLinearMap()
            input_size = input_size * num_logits

        elif self.mapping in ['STAP', 'SAP', 'SASP2', 'SASP']:
            encode_layer, encoder_output = get_encode_layer(encoder_type=self.mapping,
                                                            encode_input_dim=input_size//16, hidden_dim=hidden_size,
                                                            embedding_size=input_size//16, time_dim=0)
            self.map = nn.Sequential(
                nn.Conv1d(in_channels=input_size, out_channels=input_size//16, kernel_size=1, bias=False),
                encode_layer,)
            input_size= encoder_output
        
        layers = []
        if self.warm_start:
            layers.append(WarmStartGradientReverseLayer(max_iters=max_iters))
        else:
            layers.append(RevGradLayer())
        
        output_size =  hidden_size if hidden_size > 0 else input_size
        layers.extend([
            nn.Linear(input_size, output_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(output_size)
        ])
        
        if hidden_size > 0:
            layers.extend([
                nn.Linear(output_size, output_size),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(output_size)
            ])
            
        if sigmoid:
            layers.extend([
                nn.Linear(output_size, 1),
                nn.Sigmoid()
            ])
        else:
            layers.append(nn.Linear(output_size, num_classes))
            
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        if isinstance(x, tuple):
            x, logits = x
            logits = F.softmax(logits, dim=1).detach()
            
            if self.mapping != 'none':
                x = self.map(x, logits)
            
        return self.classifier(x)
    

__factory = {
    'AlexNet': AlexNet,
    'LoResNet': LocalResNet,
    'ResNet20': ResNet20,
    'SiResNet34': SimpleResNet,
    'ThinResNet': ThinResNet,
    'RepeatResNet': RepeatResNet,
    'MultiResNet': MultiResNet,
    'ResNet': ResNet,
    'DTDNN': DTDNN,
    'TDNN': TDNN_v2,
    'TDNN_v4': TDNN_v4,
    'TDNN_v5': TDNN_v5,
    'TDNN_v6': TDNN_v6,
    'MixTDNN_v5': MixTDNN_v5,
    'SlimmableTDNN': SlimmableTDNN,
    'ETDNN': ETDNN,
    'ETDNN_v4': ETDNN_v4,
    'ETDNN_v5': ETDNN_v5,
    'FTDNN': FTDNN,
    'ECAPA': ECAPA_TDNN,
    'ECAPA_brain': ECAPA_brain.ECAPA_TDNN,
    'RET': RET,
    'RET_v2': RET_v2,
    'RET_v3': RET_v3,
    'GradResNet': GradResNet,
    'TimeFreqResNet': TimeFreqResNet,
    'Demucs': Demucs
}


def create_model(name, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))

    model = __factory[name](**kwargs)
    create_classifier(model, **kwargs)

    return model


def create_classifier(encode_model, **kwargs):
    if not isinstance(encode_model.classifier, ECAPA_brain.Classifier):
        if kwargs['loss_type'] in ['asoft', 'amsoft', 'damsoft', 'arcsoft', 'arcdist', 'minarcsoft', 'minarcsoft2', 'aDCF']:
            encode_model.classifier = AdditiveMarginLinear(feat_dim=kwargs['embedding_size'],
                                                        normalize=kwargs['normalize'] if 'normalize' in kwargs else True,
                                                        num_classes=kwargs['num_classes'])
        elif 'sub' in kwargs['loss_type']:
            encode_model.classifier = SubMarginLinear(feat_dim=kwargs['embedding_size'],
                                                    num_classes=kwargs['num_classes'],
                                                    num_center=kwargs['num_center'],
                                                    output_subs=kwargs['output_subs'])
        elif kwargs['loss_type'] in ['proser']:
            encode_model.classifier = MarginLinearDummy(feat_dim=kwargs['embedding_size'],
                                                        dummy_classes=kwargs['num_center'],
                                                        num_classes=kwargs['num_classes'])


def save_model_args(model_dict, save_path):
    with open(save_path, 'w') as f:
        yamlText = yaml.dump(model_dict)
        f.write(yamlText)


def load_model_args(model_yaml):
    with open(model_yaml, 'r') as f:
        model_args = load_hyperpyyaml(f)

    return model_args