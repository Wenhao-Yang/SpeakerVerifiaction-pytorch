#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: train_egs.py
@Time: 2020/8/21 20:30
@Overview:
"""
from __future__ import print_function

# import argparse
import os
import os.path as osp
# import random
import pdb
import shutil
import sys
import time
# Version conflict
import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from kaldiio import WriteHelper

import torch.nn.functional as F

import torchvision.transforms as transforms
from kaldi_io import read_mat, read_vec_flt
# from kaldiio import load_mat
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel
from torch.optim import lr_scheduler
from tqdm import tqdm

from Define_Model.Loss.LossFunction import CenterLoss, Wasserstein_Loss, MultiCenterLoss, CenterCosLoss
from Define_Model.Loss.SoftmaxLoss import AngleSoftmaxLoss, AMSoftmaxLoss, \
    ArcSoftmaxLoss, GaussianLoss
from Process_Data.Datasets.KaldiDataset import KaldiExtractDataset, \
    ScriptVerifyDataset
from Process_Data.Datasets.LmdbDataset import EgsDataset
from Process_Data.audio_processing import ConcateVarInput, tolog, ConcateOrgInput, PadCollate
from Process_Data.audio_processing import totensor  # , toMFB, truncatedinput
from TrainAndTest.common_func import create_optimizer, create_model, verification_test, verification_extract, \
    args_parse, args_model, load_model_args
from logger import NewLogger

warnings.filterwarnings("ignore")

# import torch._utils
#
# try:
#     torch._utils._rebuild_tensor_v2
# except AttributeError:
#     def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
#         tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
#         tensor.requires_grad = requires_grad
#         tensor._backward_hooks = backward_hooks
#         return tensor
#
#
#     torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

args = args_parse('PyTorch Speaker Recognition: Classification, Knowledge Distillation')

# Set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '29555'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
# torch.multiprocessing.set_sharing_strategy('file_system')

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

# create logger
# Define visulaize SummaryWriter instance
writer = SummaryWriter(logdir=args.check_path, filename_suffix='_first')
sys.stdout = NewLogger(osp.join(args.check_path, 'log.%s.txt' % time.strftime("%Y.%m.%d", time.localtime())))

kwargs = {'num_workers': args.nj, 'pin_memory': True} if args.cuda else {}
extract_kwargs = {'num_workers': args.nj, 'pin_memory': False} if args.cuda else {}

if not os.path.exists(args.check_path):
    print('Making checkpath...')
    os.makedirs(args.check_path)

opt_kwargs = {'lr': args.lr, 'lr_decay': args.lr_decay, 'weight_decay': args.weight_decay, 'dampening': args.dampening,
              'momentum': args.momentum}

l2_dist = nn.CosineSimilarity(dim=1, eps=1e-12) if args.cos_sim else nn.PairwiseDistance(p=2)

transform = transforms.Compose([
    totensor()
])

if args.test_input == 'var':
    transform_V = transforms.Compose([
        ConcateOrgInput(remove_vad=args.remove_vad),
    ])
elif args.test_input == 'fix':
    transform_V = transforms.Compose([
        ConcateVarInput(remove_vad=args.remove_vad, num_frames=args.chunk_size, frame_shift=args.chunk_size,
                        feat_type=args.feat_format),
    ])

if args.log_scale:
    transform.transforms.append(tolog())
    transform_V.transforms.append(tolog())

# pdb.set_trace()
if args.feat_format in ['kaldi', 'wav']:
    file_loader = read_mat
elif args.feat_format == 'npy':
    file_loader = np.load

torch.multiprocessing.set_sharing_strategy('file_system')

train_dir = EgsDataset(dir=args.train_dir, feat_dim=args.input_dim, loader=file_loader, transform=transform,
                       batch_size=args.batch_size, random_chunk=args.random_chunk)

train_extract_dir = KaldiExtractDataset(dir=args.train_test_dir,
                                        transform=transform_V,
                                        filer_loader=file_loader,
                                        trials_file=args.train_trials)

extract_dir = KaldiExtractDataset(dir=args.test_dir, transform=transform_V,
                                  trials_file=args.trials, filer_loader=file_loader)

# train_test_dir = ScriptTestDataset(dir=args.train_test_dir, loader=file_loader, transform=transform_T)
# test_dir = ScriptTestDataset(dir=args.test_dir, loader=file_loader, transform=transform_T)
valid_dir = EgsDataset(dir=args.valid_dir, feat_dim=args.input_dim, loader=file_loader, transform=transform)


def extract_smooth_label(train_loader, teacher_model, smooth_label_dir):
    # switch to evaluate mode
    teacher_model.eval()
    pbar = tqdm(enumerate(train_loader))

    if not os.path.exists(smooth_label_dir):
        os.makedirs(smooth_label_dir)
    feat_scp = os.path.join(smooth_label_dir, 'feat.scp')
    feat_ark = os.path.join(smooth_label_dir, 'feat.ark')

    writer = WriteHelper('ark,scp:%s,%s' % (feat_ark, feat_scp))#, compression_method=1)

    with torch.no_grad():

        for batch_idx, (data, label) in pbar:

            if args.cuda:
                # label = label.cuda(non_blocking=True)
                data = data.cuda()

            # data, label = Variable(data), Variable(label)

            classfier, feats = teacher_model(data)

            smooth_label = classfier.cpu().float().numpy()
            true_label = label.cpu().long().numpy()
            for x,y in zip(true_label, smooth_label):
                writer(str(x), y)

            # pdb.set_trace()
            # with torch.no_grad():
            #     t_classfier, t_feats = teacher_model(data)
            # if args.kd_type == 'vanilla':

    # torch.cuda.empty_cache()

def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive
    # test_display_triplet_distance = False
    # print the experiment configuration
    print('\nCurrent time is \33[91m{}\33[0m.'.format(str(time.asctime())))

    # Create teacher model
    teacher_model_kwargs = load_model_args(args.teacher_model_yaml)
    if args.teacher_model == '':
        args.teacher_model = args.model

    teacher_model = create_model(args.teacher_model, **teacher_model_kwargs)
    if args.teacher_resume:
        if os.path.isfile(args.teacher_resume):
            print('=> loading teacher checkpoint {}'.format(args.teacher_resume))
            checkpoint = torch.load(args.teacher_resume)
            # start_epoch = checkpoint['epoch']

            checkpoint_state_dict = checkpoint['state_dict']
            if isinstance(checkpoint_state_dict, tuple):
                checkpoint_state_dict = checkpoint_state_dict[0]
            filtered = {k: v for k, v in checkpoint_state_dict.items() if 'num_batches_tracked' not in k}

            # filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}
            if list(filtered.keys())[0].startswith('module'):
                new_state_dict = OrderedDict()
                for k, v in filtered.items():
                    name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，去掉module.
                    new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。

                teacher_model.load_state_dict(new_state_dict)
            else:
                model_dict = teacher_model.state_dict()
                model_dict.update(filtered)
                teacher_model.load_state_dict(model_dict)
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    # Save model config txt
    # start = args.start_epoch +
    # print('Start epoch is : ' + str(start))
    # # start = 0
    # end = start + args.epochs

    if len(args.random_chunk) == 2 and args.random_chunk[0] <= args.random_chunk[1]:
        min_chunk_size = int(args.random_chunk[0])
        max_chunk_size = int(args.random_chunk[1])
        pad_dim = 2 if args.feat_format == 'kaldi' else 3

        train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size,
                                                   collate_fn=PadCollate(dim=pad_dim,
                                                                         num_batch=int(
                                                                             np.ceil(len(train_dir) / args.batch_size)),
                                                                         min_chunk_size=min_chunk_size,
                                                                         max_chunk_size=max_chunk_size),
                                                   shuffle=args.shuffle, **kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_dir, batch_size=int(args.batch_size / 2),
                                                   collate_fn=PadCollate(dim=pad_dim, fix_len=True,
                                                                         min_chunk_size=args.chunk_size,
                                                                         max_chunk_size=args.chunk_size + 1),
                                                   shuffle=False, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=args.shuffle,
                                                   **kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_dir, batch_size=int(args.batch_size / 2), shuffle=False,
                                                   **kwargs)

    if args.cuda:
        if len(args.gpu_id) > 1:
            print("Continue with gpu: %s ..." % str(args.gpu_id))
            torch.distributed.init_process_group(backend="nccl",
                                                 init_method='file:///home/ssd2020/yangwenhao/lstm_speaker_verification/data/sharedfile2',
                                                 rank=0,
                                                 world_size=1)

            teacher_model = DistributedDataParallel(teacher_model.cuda())
            # model = DistributedDataParallel(model.cuda(), find_unused_parameters=True)

        else:
            teacher_model = teacher_model.cuda()

    teacher_model.eval()

    smooth_label_dir = args.check_path.replace('checkpoint', 'label')
    start_time = time.time()


    extract_smooth_label(train_loader, teacher_model, smooth_label_dir+'/train')
    extract_smooth_label(valid_loader, teacher_model, smooth_label_dir+'/valid')


    # torch.cuda.empty_cache()
    # torch.distributed.destroy_process_group()
    writer.close()
    stop_time = time.time()
    t = float(stop_time - start_time)
    print("Running %.4f minutes.\n" % (t/60) )
    # pdb.set_trace()
    # sys.exit(0)


if __name__ == '__main__':
    main()
