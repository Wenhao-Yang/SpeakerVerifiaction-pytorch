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

import argparse
import os
import os.path as osp
import sys
import time
# Version conflict
import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from kaldi_io import read_mat, read_vec_flt
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel
from torch.optim import lr_scheduler
from tqdm import tqdm

from Define_Model.LossFunction import CenterLoss, Wasserstein_Loss, MultiCenterLoss, CenterCosLoss, RingLoss
from Define_Model.SoftmaxLoss import AngleSoftmaxLoss, AngleLinear, AdditiveMarginLinear, AMSoftmaxLoss, ArcSoftmaxLoss, \
    GaussianLoss
from Process_Data.Datasets.KaldiDataset import KaldiExtractDataset, \
    ScriptVerifyDataset
from Process_Data.Datasets.LmdbDataset import EgsDataset
from Process_Data.audio_processing import ConcateVarInput, tolog, ConcateOrgInput
from Process_Data.audio_processing import toMFB, totensor, truncatedinput, read_audio
from TrainAndTest.common_func import create_optimizer, create_model, verification_test, verification_extract
from logger import NewLogger

warnings.filterwarnings("ignore")

import torch._utils

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor


    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')

# Data options
parser.add_argument('--train-dir', type=str, required=True, help='path to dataset')
parser.add_argument('--train-test-dir', type=str, required=True, help='path to dataset')
parser.add_argument('--valid-dir', type=str, required=True, help='path to dataset')
parser.add_argument('--test-dir', type=str, required=True, help='path to voxceleb1 test dataset')
parser.add_argument('--log-scale', action='store_true', default=False, help='log power spectogram')

parser.add_argument('--trials', type=str, default='trials', help='path to voxceleb1 test dataset')
parser.add_argument('--train-trials', type=str, default='trials', help='path to voxceleb1 test dataset')

parser.add_argument('--sitw-dir', type=str, help='path to voxceleb1 test dataset')
parser.add_argument('--fix-length', action='store_true', default=True, help='need to make mfb file')
parser.add_argument('--test-input', type=str, default='fix', choices=['var', 'fix'],
                    help='batchnorm with instance norm')
parser.add_argument('--random-chunk', nargs='+', type=int, default=[], metavar='MINCHUNK')
parser.add_argument('--chunk-size', type=int, default=300, metavar='CHUNK')

parser.add_argument('--remove-vad', action='store_true', default=False, help='using Cosine similarity')
parser.add_argument('--extract', action='store_true', default=True, help='need to make mfb file')

parser.add_argument('--nj', default=10, type=int, metavar='NJOB', help='num of job')
parser.add_argument('--feat-format', type=str, default='kaldi', choices=['kaldi', 'npy'],
                    help='number of jobs to make feats (default: 10)')

parser.add_argument('--check-path', default='Data/checkpoint/GradResNet8/vox1/spect_egs/soft_dp25',
                    help='folder to output model checkpoints')
parser.add_argument('--save-init', action='store_true', default=True, help='need to make mfb file')
parser.add_argument('--resume',
                    default='Data/checkpoint/GradResNet8/vox1/spect_egs/soft_dp25/checkpoint_10.pth', type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=20, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--scheduler', default='multi', type=str,
                    metavar='SCH', help='The optimizer to use (default: Adagrad)')
parser.add_argument('--patience', default=2, type=int,
                    metavar='PAT', help='patience for scheduler (default: 4)')
parser.add_argument('--gamma', default=0.75, type=float,
                    metavar='GAMMA', help='The optimizer to use (default: Adagrad)')
parser.add_argument('--milestones', default='10,15', type=str,
                    metavar='MIL', help='The optimizer to use (default: Adagrad)')
parser.add_argument('--min-softmax-epoch', type=int, default=40, metavar='MINEPOCH',
                    help='minimum epoch for initial parameter using softmax (default: 2')
parser.add_argument('--veri-pairs', type=int, default=20000, metavar='VP',
                    help='number of epochs to train (default: 10)')

# Training options
# Model options
parser.add_argument('--model', type=str, help='path to voxceleb1 test dataset')
parser.add_argument('--resnet-size', default=8, type=int,
                    metavar='RES', help='The channels of convs layers)')
parser.add_argument('--filter', type=str, default='None', help='replace batchnorm with instance norm')
parser.add_argument('--input-norm', type=str, default='Mean', help='batchnorm with instance norm')

parser.add_argument('--mask-layer', type=str, default='None', help='time or freq masking layers')
parser.add_argument('--mask-len', type=int, default=20, help='maximum length of time or freq masking layers')
parser.add_argument('--block-type', type=str, default='basic', help='replace batchnorm with instance norm')
parser.add_argument('--relu-type', type=str, default='relu', help='replace batchnorm with instance norm')
parser.add_argument('--transform', type=str, default="None", help='add a transform layer after embedding layer')

parser.add_argument('--vad', action='store_true', default=False, help='vad layers')
parser.add_argument('--inception', action='store_true', default=False, help='multi size conv layer')
parser.add_argument('--inst-norm', action='store_true', default=False, help='batchnorm with instance norm')

parser.add_argument('--encoder-type', type=str, default='None', help='path to voxceleb1 test dataset')
parser.add_argument('--channels', default='64,128,256', type=str, metavar='CHA', help='The channels of convs layers)')
parser.add_argument('--feat-dim', default=64, type=int, metavar='N', help='acoustic feature dimension')
parser.add_argument('--input-dim', default=257, type=int, metavar='N', help='acoustic feature dimension')
parser.add_argument('--accu-steps', default=1, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument('--alpha', default=12, type=float, metavar='FEAT', help='acoustic feature dimension')
parser.add_argument('--ring', default=12, type=float, metavar='RING', help='acoustic feature dimension')

parser.add_argument('--kernel-size', default='5,5', type=str, metavar='KE', help='kernel size of conv filters')
parser.add_argument('--context', default='5,3,3,5', type=str, metavar='KE', help='kernel size of conv filters')

parser.add_argument('--padding', default='', type=str, metavar='KE', help='padding size of conv filters')
parser.add_argument('--stride', default='2', type=str, metavar='ST', help='stride size of conv filters')
parser.add_argument('--fast', action='store_true', default=False, help='max pooling for fast')

parser.add_argument('--cos-sim', action='store_true', default=False, help='using Cosine similarity')
parser.add_argument('--avg-size', type=int, default=4, metavar='ES', help='Dimensionality of the embedding')
parser.add_argument('--time-dim', default=1, type=int, metavar='FEAT', help='acoustic feature dimension')
parser.add_argument('--embedding-size', type=int, default=128, metavar='ES',
                    help='Dimensionality of the embedding')
parser.add_argument('--batch-size', type=int, default=128, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--input-per-spks', type=int, default=224, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--num-valid', type=int, default=5, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--test-input-per-file', type=int, default=4, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--test-batch-size', type=int, default=4, metavar='BST',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--dropout-p', type=float, default=0.25, metavar='BST',
                    help='input batch size for testing (default: 64)')

# loss configure
parser.add_argument('--loss-type', type=str, default='soft',
                    help='path to voxceleb1 test dataset')
parser.add_argument('--num-center', type=int, default=2, help='the num of source classes')
parser.add_argument('--source-cls', type=int, default=1951,
                    help='the num of source classes')
parser.add_argument('--finetune', action='store_true', default=False,
                    help='using Cosine similarity')
parser.add_argument('--loss-ratio', type=float, default=0.1, metavar='LOSSRATIO',
                    help='the ratio softmax loss - triplet loss (default: 2.0')

# args for additive margin-softmax
parser.add_argument('--margin', type=float, default=0.3, metavar='MARGIN',
                    help='the margin value for the angualr softmax loss function (default: 3.0')
parser.add_argument('--s', type=float, default=15, metavar='S',
                    help='the margin value for the angualr softmax loss function (default: 3.0')

# args for a-softmax
parser.add_argument('--all-iteraion', type=int, default=100, metavar='M',
                    help='the margin value for the angualr softmax loss function (default: 3.0')
parser.add_argument('--m', type=int, default=3, metavar='M',
                    help='the margin value for the angualr softmax loss function (default: 3.0')
parser.add_argument('--lambda-min', type=int, default=5, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--lambda-max', type=float, default=1000, metavar='S',
                    help='random seed (default: 0)')

parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.125)')
parser.add_argument('--lr-decay', default=0, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-4')
parser.add_argument('--weight-decay', default=5e-4, type=float,
                    metavar='WEI', help='weight decay (default: 0.0)')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='MOM', help='momentum for sgd (default: 0.9)')
parser.add_argument('--dampening', default=0, type=float,
                    metavar='DAM', help='dampening for sgd (default: 0.0)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')
parser.add_argument('--grad-clip', default=0., type=float,
                    help='momentum for sgd (default: 0.9)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=123456, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')

parser.add_argument('--acoustic-feature', choices=['fbank', 'spectrogram', 'mfcc'], default='fbank',
                    help='choose the acoustic features type.')
parser.add_argument('--makemfb', action='store_true', default=False,
                    help='need to make mfb file')
parser.add_argument('--makespec', action='store_true', default=False,
                    help='need to make spectrograms file')

args = parser.parse_args()

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
writer = SummaryWriter(logdir=args.check_path, filename_suffix='_first')

sys.stdout = NewLogger(osp.join(args.check_path, 'log.%s.txt' % time.strftime("%Y.%m.%d", time.localtime())))

kwargs = {'num_workers': args.nj, 'pin_memory': False} if args.cuda else {}
extract_kwargs = {'num_workers': args.nj, 'pin_memory': False} if args.cuda else {}

if not os.path.exists(args.check_path):
    os.makedirs(args.check_path)

opt_kwargs = {'lr': args.lr, 'lr_decay': args.lr_decay, 'weight_decay': args.weight_decay, 'dampening': args.dampening,
              'momentum': args.momentum}

l2_dist = nn.CosineSimilarity(dim=1, eps=1e-12) if args.cos_sim else nn.PairwiseDistance(p=2)

if args.acoustic_feature == 'fbank':
    transform = transforms.Compose([
        ConcateOrgInput(remove_vad=args.remove_vad),
    ])
else:
    transform = transforms.Compose([
        truncatedinput(),
        toMFB(),
        totensor(),
        # tonormal()
    ])
    file_loader = read_audio

if args.test_input == 'var':
    transform_V = transforms.Compose([
        ConcateOrgInput(remove_vad=args.remove_vad),
    ])
elif args.test_input == 'fix':
    transform_V = transforms.Compose([
        ConcateVarInput(remove_vad=args.remove_vad),
    ])

if args.log_scale:
    transform.transforms.append(tolog())
    transform_V.transforms.append(tolog())

# pdb.set_trace()
if args.feat_format == 'kaldi':
    file_loader = read_mat
elif args.feat_format == 'npy':
    file_loader = np.load
torch.multiprocessing.set_sharing_strategy('file_system')

train_dir = EgsDataset(dir=args.train_dir, feat_dim=args.feat_dim, loader=file_loader, transform=transform,
                       batch_size=args.batch_size, random_chunk=args.random_chunk)

train_extract_dir = KaldiExtractDataset(dir=args.train_test_dir,
                                        transform=transform_V,
                                        filer_loader=file_loader,
                                        trials_file=args.train_trials)

extract_dir = KaldiExtractDataset(dir=args.test_dir, transform=transform_V, filer_loader=file_loader)

# train_test_dir = ScriptTestDataset(dir=args.train_test_dir, loader=file_loader, transform=transform_T)
# test_dir = ScriptTestDataset(dir=args.test_dir, loader=file_loader, transform=transform_T)
valid_dir = EgsDataset(dir=args.valid_dir, feat_dim=args.feat_dim, loader=file_loader, transform=transform)


def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive
    # test_display_triplet_distance = False
    # print the experiment configuration
    print('\nCurrent time is \33[91m{}\33[0m.'.format(str(time.asctime())))
    opts = vars(args)
    keys = list(opts.keys())
    keys.sort()

    options = []
    for k in keys:
        options.append("\'%s\': \'%s\'" % (str(k), str(opts[k])))

    print('Parsed options: \n{ %s }' % (', '.join(options)))
    print('Number of Speakers: {}.\n'.format(train_dir.num_spks))

    # instantiate model and initialize weights
    kernel_size = args.kernel_size.split(',')
    kernel_size = [int(x) for x in kernel_size]

    context = args.context.split(',')
    context = [int(x) for x in context]
    if args.padding == '':
        padding = [int((x - 1) / 2) for x in kernel_size]
    else:
        padding = args.padding.split(',')
        padding = [int(x) for x in padding]

    kernel_size = tuple(kernel_size)
    padding = tuple(padding)
    stride = args.stride.split(',')
    stride = [int(x) for x in stride]

    channels = args.channels.split(',')
    channels = [int(x) for x in channels]

    model_kwargs = {'input_dim': args.input_dim, 'feat_dim': args.feat_dim, 'kernel_size': kernel_size,
                    'context': context,
                    'mask': args.mask_layer, 'mask_len': args.mask_len, 'block_type': args.block_type,
                    'filter': args.filter, 'inst_norm': args.inst_norm, 'input_norm': args.input_norm,
                    'stride': stride, 'fast': args.fast, 'avg_size': args.avg_size, 'time_dim': args.time_dim,
                    'padding': padding, 'encoder_type': args.encoder_type, 'vad': args.vad,
                    'transform': args.transform, 'embedding_size': args.embedding_size, 'ince': args.inception,
                    'resnet_size': args.resnet_size, 'num_classes': train_dir.num_spks,
                    'channels': channels, 'alpha': args.alpha, 'dropout_p': args.dropout_p}

    print('Model options: {}'.format(model_kwargs))
    dist_type = 'cos' if args.cos_sim else 'l2'
    print('Testing with %s distance, ' % dist_type)

    model = create_model(args.model, **model_kwargs)

    start_epoch = 0
    if args.save_init and not args.finetune:
        check_path = '{}/checkpoint_{}.pth'.format(args.check_path, start_epoch)
        torch.save(model, check_path)

    iteration = 100 if args.resume else 0
    if args.finetune and args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']

            checkpoint_state_dict = checkpoint['state_dict']
            if isinstance(checkpoint_state_dict, tuple):
                checkpoint_state_dict = checkpoint_state_dict[0]
            filtered = {k: v for k, v in checkpoint_state_dict.items() if 'num_batches_tracked' not in k}
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
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    ce_criterion = nn.CrossEntropyLoss()
    if args.loss_type == 'soft':
        xe_criterion = None
    elif args.loss_type == 'asoft':
        ce_criterion = None
        model.classifier = AngleLinear(in_features=args.embedding_size, out_features=train_dir.num_spks, m=args.m)
        xe_criterion = AngleSoftmaxLoss(lambda_min=args.lambda_min, lambda_max=args.lambda_max)
    elif args.loss_type == 'center':
        xe_criterion = CenterLoss(num_classes=train_dir.num_spks, feat_dim=args.embedding_size)
    elif args.loss_type == 'gaussian':
        xe_criterion = GaussianLoss(num_classes=train_dir.num_spks, feat_dim=args.embedding_size)
    elif args.loss_type == 'coscenter':
        xe_criterion = CenterCosLoss(num_classes=train_dir.num_spks, feat_dim=args.embedding_size)
    elif args.loss_type == 'mulcenter':
        xe_criterion = MultiCenterLoss(num_classes=train_dir.num_spks, feat_dim=args.embedding_size,
                                       num_center=args.num_center)
    elif args.loss_type == 'amsoft':
        ce_criterion = None
        model.classifier = AdditiveMarginLinear(feat_dim=args.embedding_size, n_classes=train_dir.num_spks)
        xe_criterion = AMSoftmaxLoss(margin=args.margin, s=args.s)
    elif args.loss_type == 'arcsoft':
        ce_criterion = None
        model.classifier = AdditiveMarginLinear(feat_dim=args.embedding_size, n_classes=train_dir.num_spks)
        xe_criterion = ArcSoftmaxLoss(margin=args.margin, s=args.s, iteraion=iteration, all_iteraion=args.all_iteraion)
    elif args.loss_type == 'wasse':
        xe_criterion = Wasserstein_Loss(source_cls=args.source_cls)
    elif args.loss_type == 'ring':
        xe_criterion = RingLoss(ring=args.ring)
        args.alpha = 0.0

    optimizer = create_optimizer(model.parameters(), args.optimizer, **opt_kwargs)
    if args.loss_type in ['center', 'mulcenter', 'gaussian', 'coscenter', 'ring']:
        optimizer = torch.optim.SGD([{'params': xe_criterion.parameters(), 'lr': args.lr * 5},
                                     {'params': model.parameters()}],
                                    lr=args.lr, weight_decay=args.weight_decay,
                                    momentum=args.momentum)
    if args.finetune:
        if args.loss_type == 'asoft' or args.loss_type == 'amsoft':
            classifier_params = list(map(id, model.classifier.parameters()))
            rest_params = filter(lambda p: id(p) not in classifier_params, model.parameters())
            optimizer = torch.optim.SGD([{'params': model.classifier.parameters(), 'lr': args.lr * 10},
                                         {'params': rest_params}],
                                        lr=args.lr, weight_decay=args.weight_decay,
                                        momentum=args.momentum)
    if args.filter == 'fDLR':
        filter_params = list(map(id, model.filter_layer.parameters()))
        rest_params = filter(lambda p: id(p) not in filter_params, model.parameters())
        optimizer = torch.optim.SGD([{'params': model.filter_layer.parameters(), 'lr': args.lr * 0.05},
                                     {'params': rest_params}],
                                    lr=args.lr, weight_decay=args.weight_decay,
                                    momentum=args.momentum)

    if not args.finetune and args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']

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
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    # Save model config txt
    with open(osp.join(args.check_path, 'model.%s.conf' % time.strftime("%Y.%m.%d", time.localtime())), 'w') as f:
        f.write('model: ' + str(model) + '\n')
        f.write('CrossEntropy: ' + str(ce_criterion) + '\n')
        f.write('Other Loss: ' + str(xe_criterion) + '\n')
        f.write('Optimizer: ' + str(optimizer) + '\n')

    milestones = args.milestones.split(',')
    milestones = [int(x) for x in milestones]
    milestones.sort()
    if args.scheduler == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    elif args.scheduler == 'rop':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, min_lr=1e-5)
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    ce = [ce_criterion, xe_criterion]

    start = args.start_epoch + start_epoch
    print('Start epoch is : ' + str(start))
    # start = 0
    end = start + args.epochs

    train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=False,
                                               **kwargs)

    valid_loader = torch.utils.data.DataLoader(valid_dir, batch_size=int(args.batch_size / 2), shuffle=False, **kwargs)
    train_extract_loader = torch.utils.data.DataLoader(train_extract_dir, batch_size=1, shuffle=False, **extract_kwargs)

    if args.cuda:
        if len(args.gpu_id) > 1:
            print("Continue with gpu: %s ..." % str(args.gpu_id))
            torch.distributed.init_process_group(backend="nccl",
                                                 # init_method='tcp://localhost:23456',
                                                 init_method='file:///home/work2020/yangwenhao/project/lstm_speaker_verification/data/sharedfile',
                                                 rank=0,
                                                 world_size=1)
            model = model.cuda()
            model = DistributedDataParallel(model)

        else:
            model = model.cuda()

        for i in range(len(ce)):
            if ce[i] != None:
                ce[i] = ce[i].cuda()
        try:
            print('Dropout is {}.'.format(model.dropout_p))
        except:
            pass

    xvector_dir = args.check_path
    xvector_dir = xvector_dir.replace('checkpoint', 'xvector')

    start_time = time.time()

    for epoch in range(start, end):
        # pdb.set_trace()
        lr_string = '\n\33[1;34m Current \'{}\' learning rate is '.format(args.optimizer)
        for param_group in optimizer.param_groups:
            lr_string += '{:.5f} '.format(param_group['lr'])
        print('%s \33[0m' % lr_string)

        train(train_loader, model, ce, optimizer, epoch)
        valid_loss = valid_class(valid_loader, model, ce, epoch)

        if (epoch == 1 or epoch != (end - 2)) and (epoch % 4 == 1 or epoch in milestones or epoch == (end - 1)):
            model.eval()
            check_path = '{}/checkpoint_{}.pth'.format(args.check_path, epoch)
            model_state_dict = model.module.state_dict() \
                                   if isinstance(model, DistributedDataParallel) else model.state_dict(),
            torch.save({'epoch': epoch,
                        'state_dict': model_state_dict,
                        'criterion': ce},
                       check_path)

            valid_test(train_extract_loader, model, epoch, xvector_dir)
            test(model, epoch, writer, xvector_dir)

        if args.scheduler == 'rop':
            scheduler.step(valid_loss)
        else:
            scheduler.step()

        # exit(1)

    writer.close()
    stop_time = time.time()
    t = float(stop_time - start_time)
    print("Running %.4f minutes for each epoch.\n" % (t / 60 / (end - start)))


def train(train_loader, model, ce, optimizer, epoch):
    # switch to evaluate mode
    model.train()

    correct = 0.
    total_datasize = 0.
    total_loss = 0.
    orth_err = 0

    ce_criterion, xe_criterion = ce
    pbar = tqdm(enumerate(train_loader))
    output_softmax = nn.Softmax(dim=1)
    # start_time = time.time()
    # pdb.set_trace()
    for batch_idx, (data, label) in pbar:
        if args.cuda:
            # label = label.cuda(non_blocking=True)
            # data = data.cuda(non_blocking=True)
            label = label.cuda()
            data = data.cuda()

        data, label = Variable(data), Variable(label)

        classfier, feats = model(data)
        # cos_theta, phi_theta = classfier
        classfier_label = classfier

        if args.loss_type == 'soft':
            loss = ce_criterion(classfier, label)

        elif args.loss_type == 'asoft':
            classfier_label, _ = classfier
            loss = xe_criterion(classfier, label)

        elif args.loss_type in ['center', 'mulcenter', 'gaussian', 'coscenter']:
            loss_cent = ce_criterion(classfier, label)
            loss_xent = xe_criterion(feats, label)

            loss = args.loss_ratio * loss_xent + loss_cent

        elif args.loss_type == 'ring':
            loss_cent = ce_criterion(classfier, label)
            loss_xent = xe_criterion(feats)
            loss = args.loss_ratio * loss_xent + loss_cent

        elif args.loss_type == 'amsoft' or args.loss_type == 'arcsoft':
            loss = xe_criterion(classfier, label)

        predicted_labels = output_softmax(classfier_label)
        predicted_one_labels = torch.max(predicted_labels, dim=1)[1]
        minibatch_correct = float((predicted_one_labels.cpu() == label.cpu()).sum().item())
        minibatch_acc = minibatch_correct / len(predicted_one_labels)
        correct += minibatch_correct

        total_datasize += len(predicted_one_labels)
        total_loss += float(loss.item())

        if np.isnan(loss.item()):
            raise ValueError('Loss value is NaN!')

        # compute gradient and update weights
        loss.backward()

        if ((batch_idx + 1) % args.accu_steps) == 0:
            # optimizer the net
            optimizer.step()  # update parameters of net
            optimizer.zero_grad()  # reset gradient

            if args.model == 'FTDNN':
                if isinstance(model, DistributedDataParallel):
                    model.module.step_ftdnn_layers()  # The key method to constrain the first two convolutions, perform after every SGD step
                    orth_err += model.module.get_orth_errors()
                else:
                    model.step_ftdnn_layers()  # The key method to constrain the first two convolutions, perform after every SGD step
                    orth_err += model.get_orth_errors()

        # optimizer.zero_grad()
        # loss.backward()

        if args.loss_ratio != 0:
            if args.loss_type in ['center', 'mulcenter', 'gaussian', 'coscenter']:
                for param in xe_criterion.parameters():
                    param.grad.data *= (1. / args.loss_ratio)

        if args.grad_clip > 0:
            this_lr = args.lr
            for param_group in optimizer.param_groups:
                this_lr = min(param_group['lr'], this_lr)
            torch.nn.utils.clip_grad_norm_(model.parameters(), this_lr * args.grad_clip)

        # optimizer.step()

        if (batch_idx + 1) % args.log_interval == 0:
            epoch_str = 'Train Epoch {}: [{:8d}/{:8d} ({:3.0f}%)]'.format(epoch, batch_idx * len(data),
                                                                          len(train_loader.dataset),
                                                                          100. * batch_idx / len(train_loader))
            if not args.fix_length:
                epoch_str += ' Batch Length: {:>3d}'.format(data.shape[-2])

            if orth_err > 0:
                epoch_str += ' Orth_err: {:>5d}'.format(int(orth_err))

            if args.loss_type in ['center', 'mulcenter', 'gaussian', 'coscenter']:
                epoch_str += ' Center Loss: {:.4f}'.format(loss_xent.float())
            epoch_str += ' Avg Loss: {:.4f} Batch Accuracy: {:.4f}%'.format(total_loss / (batch_idx + 1),
                                                                            100. * minibatch_acc)
            pbar.set_description(epoch_str)

    print('\nEpoch {:>2d}: \33[91mTrain Accuracy: {:.6f}%, Avg loss: {:6f}.\33[0m'.format(epoch, 100 * float(
        correct) / total_datasize, total_loss / len(train_loader)))
    writer.add_scalar('Train/Accuracy', correct / total_datasize, epoch)
    writer.add_scalar('Train/Loss', total_loss / len(train_loader), epoch)

    torch.cuda.empty_cache()


def valid_class(valid_loader, model, ce, epoch):
    # switch to evaluate mode
    model.eval()

    total_loss = 0.
    ce_criterion, xe_criterion = ce
    softmax = nn.Softmax(dim=1)

    correct = 0.
    total_datasize = 0.

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(valid_loader):
            data = data.cuda()
            label = label.cuda()

            # compute output
            out, feats = model(data)
            if args.loss_type == 'asoft':
                predicted_labels, _ = out
            else:
                predicted_labels = out

            classfier = predicted_labels
            if args.loss_type == 'soft':
                loss = ce_criterion(classfier, label)
            elif args.loss_type == 'asoft':
                classfier_label, _ = classfier
                loss = xe_criterion(classfier, label)
            elif args.loss_type in ['center', 'mulcenter', 'gaussian', 'coscenter']:
                loss_cent = ce_criterion(classfier, label)
                loss_xent = xe_criterion(feats, label)

                loss = args.loss_ratio * loss_xent + loss_cent
            elif args.loss_type == 'amsoft' or args.loss_type == 'arcsoft':
                loss = xe_criterion(classfier, label)

            total_loss += float(loss.item())
            # pdb.set_trace()
            predicted_one_labels = softmax(predicted_labels)
            predicted_one_labels = torch.max(predicted_one_labels, dim=1)[1]

            batch_correct = (predicted_one_labels.cuda() == label).sum().item()
            correct += batch_correct
            total_datasize += len(predicted_one_labels)

    valid_loss = total_loss / len(valid_loader)
    valid_accuracy = 100. * correct / total_datasize
    writer.add_scalar('Train/Valid_Loss', valid_loss, epoch)
    writer.add_scalar('Train/Valid_Accuracy', valid_accuracy, epoch)
    torch.cuda.empty_cache()
    print('          \33[91mValid Accuracy: {:.6f}%, Avg loss: {:.6f}.\33[0m'.format(valid_accuracy,
                                                                                     valid_loss))

    return valid_loss

def valid_test(train_extract_loader, model, epoch, xvector_dir):
    # switch to evaluate mode
    model.eval()

    this_xvector_dir = "%s/train/epoch_%s" % (xvector_dir, epoch)
    verification_extract(train_extract_loader, model, this_xvector_dir, epoch, test_input=args.test_input)

    verify_dir = ScriptVerifyDataset(dir=args.train_test_dir, trials_file=args.train_trials,
                                     xvectors_dir=this_xvector_dir,
                                     loader=read_vec_flt)
    verify_loader = torch.utils.data.DataLoader(verify_dir, batch_size=128, shuffle=False, **kwargs)
    eer, eer_threshold, mindcf_01, mindcf_001 = verification_test(test_loader=verify_loader,
                                                                  dist_type=('cos' if args.cos_sim else 'l2'),
                                                                  log_interval=args.log_interval,
                                                                  xvector_dir=this_xvector_dir,
                                                                  epoch=epoch)

    print('          \33[91mTrain EER: {:.4f}%, Threshold: {:.4f}, ' \
          'mindcf-0.01: {:.4f}, mindcf-0.001: {:.4f}. \33[0m'.format(100. * eer,
                                                                     eer_threshold,
                                                                     mindcf_01,
                                                                     mindcf_001))

    writer.add_scalar('Train/EER', 100. * eer, epoch)
    writer.add_scalar('Train/Threshold', eer_threshold, epoch)
    writer.add_scalar('Train/mindcf-0.01', mindcf_01, epoch)
    writer.add_scalar('Train/mindcf-0.001', mindcf_001, epoch)

    torch.cuda.empty_cache()

def test(model, epoch, writer, xvector_dir):
    this_xvector_dir = "%s/test/epoch_%s" % (xvector_dir, epoch)

    extract_loader = torch.utils.data.DataLoader(extract_dir, batch_size=1, shuffle=False, **extract_kwargs)
    verification_extract(extract_loader, model, this_xvector_dir, epoch, test_input=args.test_input)

    verify_dir = ScriptVerifyDataset(dir=args.test_dir, trials_file=args.trials, xvectors_dir=this_xvector_dir,
                                     loader=read_vec_flt)
    verify_loader = torch.utils.data.DataLoader(verify_dir, batch_size=128, shuffle=False, **kwargs)
    eer, eer_threshold, mindcf_01, mindcf_001 = verification_test(test_loader=verify_loader,
                                                                  dist_type=('cos' if args.cos_sim else 'l2'),
                                                                  log_interval=args.log_interval,
                                                                  xvector_dir=this_xvector_dir,
                                                                  epoch=epoch)
    print(
        '          \33[91mTest  ERR: {:.4f}%, Threshold: {:.4f}, mindcf-0.01: {:.4f}, mindcf-0.001: {:.4f}.\33[0m\n'.format(
            100. * eer, eer_threshold, mindcf_01, mindcf_001))

    writer.add_scalar('Test/EER', 100. * eer, epoch)
    writer.add_scalar('Test/Threshold', eer_threshold, epoch)
    writer.add_scalar('Test/mindcf-0.01', mindcf_01, epoch)
    writer.add_scalar('Test/mindcf-0.001', mindcf_001, epoch)


if __name__ == '__main__':
    main()
