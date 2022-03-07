#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: extract_xvector_egs.py
@Time: 2019/12/10 下午10:32
@Overview: Exctract speakers vectors for kaldi PLDA.
"""
from __future__ import print_function

import argparse
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch._utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from kaldi_io import read_mat

from Process_Data.Datasets.KaldiDataset import KaldiExtractDataset
from Process_Data.Datasets.LmdbDataset import EgsDataset
from Process_Data.audio_processing import ConcateVarInput, ConcateOrgInput, \
    tolog
from TrainAndTest.common_func import create_model, verification_extract, args_parse, args_model, load_model_args
# Version conflict
from logger import NewLogger

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

# Extracting settings
args = args_parse('PyTorch Speaker Recognition: Extraction')

# Set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '29555'

args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# torch.multiprocessing.set_sharing_strategy('file_system')

if args.cuda:
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

# create logger
# Define visulaize SummaryWriter instance
if not os.path.exists(args.xvector_dir):
    os.makedirs(args.xvector_dir)

sys.stdout = NewLogger(os.path.join(args.xvector_dir, 'log.%s.txt' % time.strftime("%Y.%m.%d", time.localtime())))

kwargs = {'num_workers': args.nj, 'pin_memory': False} if args.cuda else {}
extract_kwargs = {'num_workers': args.nj, 'pin_memory': False} if args.cuda else {}
# opt_kwargs = {'lr': args.lr, 'lr_decay': args.lr_decay, 'weight_decay': args.weight_decay, 'dampening': args.dampening,
#               'momentum': args.momentum}

l2_dist = nn.CosineSimilarity(dim=1, eps=1e-12) if args.cos_sim else nn.PairwiseDistance(p=2)

if args.test_input == 'var':
    transform_V = transforms.Compose([
        ConcateOrgInput(remove_vad=args.remove_vad, feat_type=args.feat_format),
    ])
elif args.test_input == 'fix':
    transform_V = transforms.Compose([
        ConcateVarInput(remove_vad=args.remove_vad, num_frames=args.chunk_size,
                        frame_shift=args.frame_shift,
                        feat_type=args.feat_format),
    ])

if args.log_scale:
    transform_V.transforms.append(tolog())

# pdb.set_trace()
if args.feat_format == 'kaldi':
    file_loader = read_mat
elif args.feat_format == 'npy':
    file_loader = np.load

torch.multiprocessing.set_sharing_strategy('file_system')

# pdb.set_trace()
train_dir = EgsDataset(dir=args.train_dir, feat_dim=args.feat_dim, loader=file_loader,
                       transform=transform_V,
                       batch_size=args.batch_size, random_chunk=args.random_chunk)


# test_dir = ScriptTestDataset(dir=args.test_dir, loader=file_loader, transform=transform_T)


def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive

    # print the experiment configuration
    print('\nCurrent time is\33[91m {}\33[0m.'.format(str(time.asctime())))
    opts = vars(args)
    keys = list(opts.keys())
    keys.sort()

    options = []
    for k in keys:
        options.append("\'%s\': \'%s\'" % (str(k), str(opts[k])))

    print('Parsed options: \n{ %s }' % (', '.join(options)))
    print('Number of Speakers in training set: {}\n'.format(train_dir.num_spks))

    if os.path.exists(args.check_yaml):
        model_kwargs = load_model_args(args.check_yaml)
    else:
        model_kwargs = args_model(args, train_dir)

    keys = list(model_kwargs.keys())
    keys.sort()
    model_options = ["\'%s\': \'%s\'" % (str(k), str(model_kwargs[k])) for k in keys]
    print('Model options: \n{ %s }' % (', '.join(model_options)))

    model = create_model(args.model, **model_kwargs)

    # optionally resume from a checkpoint
    # resume = args.ckp_dir + '/checkpoint_{}.pth'.format(args.epoch)
    assert os.path.isfile(args.resume), print('=> no checkpoint found at {}'.format(args.resume))

    print('=> loading checkpoint {}'.format(args.resume))
    checkpoint = torch.load(args.resume)
    epoch = checkpoint['epoch']

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

        model.load_state_dict(new_state_dict)
    else:
        model_dict = model.state_dict()
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
    # model.dropout.p = args.dropout_p

    if args.cuda:
        model.cuda()

    extracted_set = []

    vec_type = 'xvectors_a' if args.xvector else 'xvectors_b'
    if args.train_dir != '':
        train_extract_dir = KaldiExtractDataset(dir=args.train_extract_dir, filer_loader=file_loader,
                                                transform=transform_V,
                                                extract_trials=False)
        train_loader = torch.utils.data.DataLoader(train_extract_dir, batch_size=args.batch_size, shuffle=False,
                                                   **kwargs)
        # Extract Train set vectors
        # extract(train_loader, model, dataset='train', extract_path=args.extract_path + '/x_vector')
        train_xvector_dir = args.xvector_dir + '/%s/epoch_%d/train' % (vec_type, epoch)
        verification_extract(train_loader, model, train_xvector_dir, epoch=epoch, test_input=args.test_input,
                             mean_vector=args.mean_vector,
                             verbose=args.verbose, xvector=args.xvector)
        # copy wav.scp and utt2spk ...
        extracted_set.append('train')

    assert args.test_dir != ''
    test_dir = KaldiExtractDataset(dir=args.test_dir, filer_loader=file_loader, transform=transform_V,
                                   extract_trials=False)
    test_loader = torch.utils.data.DataLoader(test_dir, batch_size=args.batch_size, shuffle=False, **kwargs)

    # Extract test set vectors
    test_xvector_dir = args.xvector_dir + '/%s/epoch_%d/test' % (vec_type, epoch)
    # extract(test_loader, model, set_id='test', extract_path=args.extract_path + '/x_vector')
    verification_extract(test_loader, model, test_xvector_dir, epoch=epoch, test_input=args.test_input,
                         mean_vector=args.mean_vector,
                         verbose=args.verbose, xvector=args.xvector)
    # copy wav.scp and utt2spk ...
    extracted_set.append('test')

    if len(extracted_set) > 0:
        print('Extract x-vector completed for %s in %s!\n' % (
        ','.join(extracted_set), args.xvector_dir + '/%s' % vec_type))


if __name__ == '__main__':
    main()

