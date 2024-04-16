#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: VS code
@File: train_egs_dist.py
@Time: 2024/4/16 16:21
@Overview:
"""

from Process_Data.Datasets.KaldiDataset import ScriptTrainDataset
from Light.dataset import Sampler_Loaders, SubScriptDatasets

import os
from hyperpyyaml import load_hyperpyyaml
from Light.model import SpeakerLoss
from geomloss import SamplesLoss

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser

def argument_parser():
    
    parser = ArgumentParser()
    parser.add_argument("--data-root", type=str,
                        default='/home/yangwenhao/project/SpeakerVerification-pytorch')
    
    parser.add_argument("--model-yaml", type=str,
                        default='data/vox1/model.2024.03.15.yaml')
    parser.add_argument("--model-path", type=str,
                        default='Data/checkpoint/ECAPA_brain/Mean_batch96_SASP2_em192_official_2s/arcsoft_adam_cyclic/vox1/wave_fb80_inst_aug53/1234')
    
    parser.add_argument("--device", type=str, default='cuda:1') 
    
    parser.add_argument("--epoch",  type=str, default='avg3')
    parser.add_argument("--random-seed",  type=int, default=1234)
    
    parser.add_argument("--save-path", type=str, default='data/vox1_inst')


    return parser.parse_args()

args = argument_parser()

device = args.device
if 'cuda' in device:
    torch.cuda.set_device(1)

data_root    = args.data_root
train_config = args.model_yaml

# Dataset & Dataloader
config_args['verbose'] = 1
config_args['save_data_dir'] = data_root + '/' + config_args['save_data_dir']
train_dir, valid_dir, train_extract_dir = SubScriptDatasets(config_args)

batch_size  = config_args['batch_size'] 
num_classes = config_args['num_classes']

batch_loader = torch.utils.data.DataLoader(
    train_dir, batch_size=batch_size, num_workers=config_args['nj'])
sample_num   = len(train_dir)



epochs = args.epochs.split(',') #'avg3'

for epoch in epochs:
    with open(train_config, 'r') as f:
        config_args = load_hyperpyyaml(f)

    if 'embedding_model' in config_args:
        model = config_args['embedding_model']

    if 'classifier' in config_args:
        model.classifier = config_args['classifier']

    model.loss = SpeakerLoss(config_args)
    model      = model.cuda()
    
    resume = args.model_path + '/checkpoint_{}.pth'.format(epoch)
    checkpoint = torch.load(os.path.join(data_root, resume), map_location='cpu') 

    start_epoch = checkpoint['epoch']
    checkpoint_state_dict = checkpoint['state_dict']
    if isinstance(checkpoint_state_dict, tuple):
        checkpoint_state_dict = checkpoint_state_dict[0]

    filtered = {k: v for k, v in checkpoint_state_dict.items(
    ) if 'num_batches_tracked' not in k}

    if list(filtered.keys())[0].startswith('module'):
        new_state_dict = OrderedDict()
        for k, v in filtered.items():
            new_state_dict[k[7:]] = v  # 新字典的key值对应的value为一一对应的值。

        model.load_state_dict(new_state_dict)
    else:
        model_dict = model.state_dict()
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
    
    # Extracting 
    embedding_dim = config_args['embedding_size']
    embeddings    = torch.zeros([len(train_dir), embedding_dim], requires_grad=False)
    grads         = torch.zeros([len(train_dir), embedding_dim], requires_grad=False)
    logits        = torch.zeros([len(train_dir), config_args['num_classes']], requires_grad=False)

    
    pbar = tqdm(enumerate(batch_loader), ncols=50, total=len(batch_loader))
    model.eval()

    for i, (data, label) in pbar:

        logit, embedding = model(data.to(device))
        loss, _ = model.loss(logit, label.to(device),
                             batch_weight=None, other=True)

        with torch.no_grad():
            grad = torch.autograd.grad(loss, [embedding])[0]

            grad = grad.detach().cpu()
            embedding = embedding.detach().cpu()
            logit = logit.detach().cpu()

            embeddings[i * batch_size:min((i+1) * batch_size, sample_num)] += embedding
            grads[i * batch_size:min((i+1) * batch_size, sample_num)] += grad
            logits[i * batch_size:min((i+1) * batch_size, sample_num)] += logit

    save_path = args.save_path + '_epoch{}'.format(epoch)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    try:

        torch.save({'grads': grads}, 
                   '{}/grads.pth'.format(save_path))

        torch.save({'logits': logits}, 
                   '{}/logits.pth'.format(save_path))

        torch.save({'embeddings': embeddings}, 
                   '{}/embeddings.pth'.format(save_path))
        
        print('Saving results to {save_path}'.format())
    except Exception as e:
        print('Error saving')
        print(e)
        
