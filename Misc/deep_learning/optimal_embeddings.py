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

from collections import OrderedDict
from Process_Data.Datasets.KaldiDataset import ScriptTrainDataset
from Light.dataset import Sampler_Loaders, SubScriptDatasets

import os
from hyperpyyaml import load_hyperpyyaml
from Light.model import SpeakerLoss
from Process_Data.Datasets.SelectDataset import MultiRatioDist_loss
from geomloss import SamplesLoss
import copy

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
import pandas as pd
import numpy as np
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
    parser.add_argument("--epochs",  type=str, default='avg3')
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
with open(train_config, 'r') as f:
    config_args = load_hyperpyyaml(f)

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
    
    if 'embedding_model' in config_args:
        model = copy.deepcopy(config_args['embedding_model'])

    if 'classifier' in config_args:
        model.classifier = copy.deepcopy(config_args['classifier'])

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
    save_path = args.save_path + '_{}'.format(epoch)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    grads_f = '{}/grads.pth'.format(save_path)
    logits_f = '{}/logits.pth'.format(save_path)
    embeddings_f = '{}/embeddings.pth'.format(save_path)

    if os.path.exists(grads_f) and os.path.exists(logits_f) and os.path.exists(embeddings_f):
        grads = torch.load(grads_f)['grads']
        logits = torch.load(logits_f)['logits']
        embeddings = torch.load(embeddings_f)['embeddings']

        print('Loading vectors to {}'.format(save_path))
    else:
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
            
        try:
            torch.save({'grads': grads}, grads_f)
            torch.save({'logits': logits}, logits_f)
            torch.save({'embeddings': embeddings}, embeddings_f)
            
            print('Saving results to {}'.format(save_path))

        except Exception as e:
            print('Error saving ...')
            raise(e)


    repeat = 1
    optimizer_time = 4
    metric = 'euclidean'
    batch_size = 1024
    
    wss = []
    random_seed = args.random_seed
    sample_ratio = 5
    steps = 20
    lr = 0.1
    optimize_vector = 'embeddings'
    
    for optimize_vector in ['embeddings', 'grads', 'logits']:
        print('Computing optimial distance scores ... ', optimize_vector)
        for cur_repeat in range(repeat):
            np.random.seed(random_seed)
            random_seed = random_seed - 5

            ws =  torch.ones(sample_num, 1)
            total_set = set(np.arange(sample_num))
            
            bigger_samples = np.arange(sample_num)
            bigger_batch_samples = []
            bigger_batch_size = int(batch_size*4)

            for i in range(optimizer_time*4):
                np.random.shuffle(bigger_samples)
                bigger_batch_samples.append(bigger_samples[:int(sample_num / bigger_batch_size) * bigger_batch_size].reshape(-1, bigger_batch_size))

            bigger_batch_samples = np.concatenate(bigger_batch_samples, axis=0)
            
            samples = np.arange(sample_num)
            batch_samples = []
            for i in range(optimizer_time):
                np.random.shuffle(samples)
                batch_samples.append(samples[:int(sample_num / batch_size) * batch_size].reshape(-1, batch_size))

            batch_samples = np.concatenate(batch_samples, axis=0)

            pbar = tqdm(range(len(batch_samples)), ncols=50)

            for i in pbar:
                
                select_ids = batch_samples[i]
                other_set = bigger_batch_samples[i%(bigger_batch_samples.shape[0])]
                
                if optimize_vector == 'embeddings':
                    s_xvectors = embeddings[select_ids]
                elif optimize_vector == 'logits':
                    s_xvectors = logits[select_ids]
                elif optimize_vector == 'grads':
                    s_xvectors = grads[select_ids]
                    
                x1 = s_xvectors.clone().cuda() #.to(device)
                
                # dloss = Dist_loss(ws[select_ids]).to(x1.device)
                dloss = MultiRatioDist_loss(ws[select_ids], metric='euclidean', sample_ratio=sample_ratio).to(x1.device)
                opt = torch.optim.Adam(dloss.parameters(), lr=lr, weight_decay=0)

                for j in range(steps):
                    other_set = bigger_batch_samples[(i+j)%(bigger_batch_samples.shape[0])]
                
                    if optimize_vector == 'embeddings':
                        x2 = embeddings[other_set].clone().to(device)
                    elif optimize_vector == 'logits':
                        x2 = logits[other_set].clone().to(device)
                    elif optimize_vector == 'grads':
                        x2 = grads[other_set].clone().to(device)
                        
                    L_αβ = dloss(x1, x2)
                    L_αβ.backward()
                    opt.step()
                    # print(dloss.w.data.squeeze()[:10])
                    opt.zero_grad()
                
                w = dloss.w.data.cpu()
                ws[select_ids] = w #dloss.w.data.abs().cpu()
            
            wss.append(ws)

        ws = torch.stack(wss, dim=0)
        if ws.shape[0] > 1:
            ws = ws.mean(dim=0)
        else:
            ws = ws.squeeze()

        torch.save({'scores': ws.squeeze()}, save_path + '/scores.mulscale.lr0.1.steps20.time4.bat1024.x2s.{}.pth'.format(optimize_vector))
        
