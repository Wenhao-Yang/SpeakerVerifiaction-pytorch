#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: output_extract.py
@Time: 2020/3/21 5:57 PM
@Overview:
"""
from __future__ import print_function

import argparse
import json
import os
import pdb
import pickle
import random
import time
from collections import OrderedDict
import Process_Data.constants as c
import h5py

import numpy as np
import torch
import torch._utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from kaldiio import load_mat
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from Define_Model.SoftmaxLoss import AngleLinear, AdditiveMarginLinear
# from Define_Model.model import PairwiseDistance
from Process_Data.Datasets.KaldiDataset import ScriptTrainDataset, \
    ScriptTestDataset, ScriptValidDataset
from Process_Data.audio_processing import ConcateOrgInput, mvnormal, ConcateVarInput, read_WaveFloat, read_WaveInt
from TrainAndTest.common_func import create_model, load_model_args, args_model, args_parse

# Version conflict
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
import warnings

warnings.filterwarnings("ignore")

# Training settings
args = args_parse('PyTorch Speaker Recognition: Gradient')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.multiprocessing.set_sharing_strategy('file_system')

if args.cuda:
    cudnn.benchmark = True

# Define visulaize SummaryWriter instance
kwargs = {'num_workers': args.nj, 'pin_memory': False} if args.cuda else {}
l2_dist = nn.CosineSimilarity(dim=1, eps=1e-6) if args.cos_sim else nn.PairwiseDistance(2)

if args.test_input == 'var':
    transform = transforms.Compose([
        ConcateOrgInput(remove_vad=args.remove_vad),
    ])
elif args.test_input == 'fix':
    transform = transforms.Compose([
        ConcateVarInput(remove_vad=args.remove_vad),
    ])

feat_type = 'kaldi'
if args.feat_format == 'npy':
    file_loader = np.load
elif args.feat_format in ['kaldi', 'klfb', 'klsp']:
    file_loader = load_mat
elif args.feat_format == 'wav':
    file_loader = read_WaveInt if args.wav_type == 'int' else read_WaveFloat
    feat_type = 'wav'

train_dir = ScriptTrainDataset(dir=args.train_dir, samples_per_speaker=args.input_per_spks,
                               feat_type=feat_type,
                               loader=file_loader, transform=transform, return_uid=True)

def train_extract(train_loader, file_dir):
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
   
    data_file = file_dir + '/data.h5py'
    inputs_uids = []
    with h5py.File(data_file, 'w') as df:
        for batch_idx, (data, label, uid) in pbar:

            # orig = data.detach().numpy().squeeze().astype(np.float32)
            spks = [train_dir.utt2spk_dict[u] for u in uid]
            target_label_index = [train_dir.spk_to_idx[s] for s in spks]
            label = torch.LongTensor(target_label_index)

            data = data.numpy().squeeze().astype(np.float32)

            df.create_dataset(uid[0], data=data)
            inputs_uids.append([uid[0], int(label.numpy()[0])])

            if batch_idx % args.log_interval == 0:
                pbar.set_description('Saving {} : [{:8d}/{:8d} ({:3.0f}%)] '.format(
                    uid,
                    batch_idx + 1,
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader)))

    with open(file_dir + '/uid_idx.json', 'w') as f:
        json.dump(inputs_uids, f)

    print('Saving %d input in %s.\n' % (len(inputs_uids), file_dir))
    torch.cuda.empty_cache()

def main():
    
    train_loader = DataLoader(train_dir, batch_size=args.batch_size, shuffle=False, **kwargs)
    file_dir = args.select_input_dir 
    
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)

    train_extract(train_loader, file_dir)


if __name__ == '__main__':
    main()
