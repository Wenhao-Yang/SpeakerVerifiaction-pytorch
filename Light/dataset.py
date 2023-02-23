#!/usr/bin/env python
# encoding: utf-8
'''
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: VS Code
@File: dataset.py
@Time: 2023/02/23 18:22
@Overview: 
'''
import torch
from Process_Data.Datasets.KaldiDataset import KaldiExtractDataset
from Process_Data.Datasets.LmdbDataset import EgsDataset

from Process_Data.audio_processing import toMFB, totensor, truncatedinput, PadCollate3d
from Process_Data.audio_processing import ConcateVarInput, tolog, ConcateOrgInput, PadCollate, read_Waveform
import torchvision.transforms as transforms
from kaldiio import load_mat
import numpy as np


def SubDatasets(config_args):
    transform = transforms.Compose([
        totensor()
    ])

    if config_args['test_input'] == 'var':
        transform_V = transforms.Compose([
            ConcateOrgInput(
                remove_vad=config_args['remove_vad'], feat_type=config_args['feat_format']),
        ])
    elif config_args['test_input'] == 'fix':
        transform_V = transforms.Compose([
            ConcateVarInput(remove_vad=config_args['remove_vad'], num_frames=config_args['chunk_size'],
                            frame_shift=config_args['chunk_size'],
                            feat_type=config_args['feat_format']),
        ])

    if config_args['log_scale']:
        transform.transforms.append(tolog())
        transform_V.transforms.append(tolog())

    loader_types = {'kaldi': load_mat, 'wav': load_mat, 'npy': np.load}
    file_loader = loader_types[config_args['feat_format']]

    return_domain = True if 'domain' in config_args and config_args['domain'] == True else False
    train_dir = EgsDataset(dir=config_args['train_dir'], feat_dim=config_args['input_dim'], loader=file_loader,
                           transform=transform, batch_size=config_args['batch_size'],
                           random_chunk=config_args['random_chunk'],
                           #    verbose=1 if torch.distributed.get_rank() == 0 else 0,
                           domain=return_domain)

    valid_dir = EgsDataset(dir=config_args['valid_dir'], feat_dim=config_args['input_dim'], loader=file_loader,
                           transform=transform,
                           #    verbose=1 if torch.distributed.get_rank() == 0 else 0
                           )

    feat_type = 'kaldi'
    if config_args['feat_format'] == 'wav':
        file_loader = read_Waveform
        feat_type = 'wav'

    train_extract_dir = KaldiExtractDataset(dir=config_args['train_test_dir'],
                                            transform=transform_V,
                                            filer_loader=file_loader, feat_type=feat_type,
                                            trials_file=config_args['train_trials'])

    return train_dir, valid_dir, train_extract_dir


def SubLoaders(train_dir, valid_dir, train_extract_dir, config_args):
    kwargs = {'num_workers': config_args['nj'],
              'pin_memory': False}  # if args.cuda else {}
    extract_kwargs = {'num_workers': 4,
                      'pin_memory': False}  # if args.cuda else {}

    min_chunk_size = int(config_args['random_chunk'][0])
    max_chunk_size = int(config_args['random_chunk'][1])
    pad_dim = 2 if config_args['feat_format'] == 'kaldi' else 3

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dir)
    return_domain = True if 'domain' in config_args and config_args['domain'] == True else False
    train_paddfunc = PadCollate3d if return_domain else PadCollate

    train_loader = torch.utils.data.DataLoader(train_dir, batch_size=config_args['batch_size'],
                                               collate_fn=train_paddfunc(dim=pad_dim,
                                                                         num_batch=int(
                                                                             np.ceil(len(train_dir) / config_args['batch_size'])),
                                                                         min_chunk_size=min_chunk_size,
                                                                         max_chunk_size=max_chunk_size,
                                                                         chisquare=False if 'chisquare' not in config_args else
                                                                         config_args['chisquare'],
                                                                         #  verbose=1 if torch.distributed.get_rank() == 0 else 0
                                                                         ),
                                               shuffle=config_args['shuffle'],  **kwargs)  # sampler=train_sampler,

    # valid_sampler = torch.utils.data.distributed.DistributedSampler(
    #     valid_dir)
    valid_loader = torch.utils.data.DataLoader(valid_dir, batch_size=int(config_args['batch_size'] / 2),
                                               collate_fn=PadCollate(dim=pad_dim, fix_len=True,
                                                                     min_chunk_size=min_chunk_size,
                                                                     max_chunk_size=max_chunk_size,
                                                                     #  verbose=1 if torch.distributed.get_rank() == 0 else 0
                                                                     ),
                                               shuffle=False, **kwargs)  # , sampler=valid_sampler

    # extract_sampler = torch.utils.data.distributed.DistributedSampler(extract_dir)
    # sampler = extract_sampler,
    # extract_loader = torch.utils.data.DataLoader(extract_dir, batch_size=1, shuffle=False,
    #                                                 sampler=extract_sampler, **extract_kwargs)

    # train_extract_sampler = torch.utils.data.distributed.DistributedSampler(train_extract_dir)
    # sampler=train_extract_sampler,
    train_extract_loader = torch.utils.data.DataLoader(train_extract_dir, batch_size=1, shuffle=False,
                                                       **extract_kwargs)

    return train_loader, valid_loader, train_extract_loader
