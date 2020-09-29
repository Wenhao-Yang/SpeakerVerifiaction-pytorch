#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author: WILLIAM
@file: fratio_extract.py
@Time: 2020/9/29 
@From: ASUS Win10
@Overview: 
"""
from __future__ import print_function

import argparse
import json
import os
import pickle
import random
import time

import numpy as np
import torch
import torch._utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from kaldi_io import read_mat
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from Define_Model.SoftmaxLoss import AngleLinear, AdditiveMarginLinear
from Define_Model.model import PairwiseDistance
from Process_Data.KaldiDataset import ScriptTrainDataset, \
    ScriptTestDataset, ScriptValidDataset
from Process_Data.Subband.f_ratio import fratio
from Process_Data.Subband.fratio_dataset import SpeakerDataset
from Process_Data.audio_processing import varLengthFeat, to2tensor, mvnormal, concateinputfromMFB
from TrainAndTest.common_func import create_model

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
parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
# Data options
parser.add_argument('--file-dir', type=str, help='path to dataset')
parser.add_argument('--feat-dim', default=64, type=int, metavar='N',
                    help='acoustic feature dimension')

parser.add_argument('--input-length', choices=['var', 'fix'], default='var',
                    help='choose the acoustic features type.')
parser.add_argument('--remove-vad', action='store_true', default=False, help='using Cosine similarity')
parser.add_argument('--mvnorm', action='store_true', default=False, help='using Cosine similarity')
parser.add_argument('--sample-spk', type=int, default=0, metavar='ES', help='Dimensionality of the embedding')

parser.add_argument('--nj', default=12, type=int, metavar='NJOB', help='num of job')
parser.add_argument('--batch-size', type=int, default=1, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='BST',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--input-per-spks', type=int, default=1500, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--test-input-per-file', type=int, default=1, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')

# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=123456, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=1, metavar='LI',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

# Set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
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
l2_dist = nn.CosineSimilarity(dim=1, eps=1e-6) if args.cos_sim else PairwiseDistance(2)

transform = transforms.Compose([
    varLengthFeat(remove_vad=args.remove_vad),
])

if args.mvnorm:
    transform.transforms.append(mvnormal())

file_loader = read_mat
data_dir = SpeakerDataset(dir=args.file_dir, samples_per_speaker=args.input_per_spks,
                               loader=file_loader, transform=transform, return_uid=True)

if args.sample_spk > 0:
    speakers = list(range(len(data_dir)))
    random.shuffle(speakers)
    indices = speakers[:args.sample_spk]
    data_part = torch.utils.data.Subset(data_dir, indices)
else:
    data_part = data_dir


def frames_extract(train_loader, file_dir, set_name):
    input_data = []
    pbar = tqdm(enumerate(train_loader))

    for batch_idx, (data, label) in pbar:
        data = data.squeeze().numpy()
        input_data.append(data)

    filename = file_dir + '/%s.npy' % set_name
    np.save(filename, np.array(input_data))

def fratio_extract(file_dir, set_name):
    input_data = np.load(os.path.join((file_dir, '%s.npy'%set_name)))
    f_ratio = fratio(args.feat_dim, input_data)
    np.save(os.path.join((file_dir, 'fratio.npy')), f_ratio)


def main():
    print('\nNumber of Speakers: {}.'.format(data_part.num_spks))
    # print the experiment configuration
    print('Current time is \33[91m{}\33[0m.'.format(str(time.asctime())))
    print('Parsed options: {}'.format(vars(args)))

    # instantiate model and initialize weights
    data_loader = DataLoader(data_part, batch_size=args.batch_size, shuffle=False, **kwargs)

    if not os.path.exists(args.file_dir):
        os.makedirs(args.file_dir)

    if not args.extract_frames:
        frames_extract(data_loader, args.file_dir, 'vox1_train')

    fratio_extract(args.file_dir, 'vox1_train')


if __name__ == '__main__':
    main()
