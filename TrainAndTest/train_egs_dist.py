#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: train_egs_dist.py
@Time: 2022/4/20 16:21
@Overview:
"""
from __future__ import print_function

import argparse
import signal
import yaml
import os
import os.path as osp
import pdb
import random
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
import torchvision.transforms as transforms
from hyperpyyaml import load_hyperpyyaml
from kaldi_io import read_mat, read_vec_flt
from kaldiio import load_mat
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.distributed as dist

from Define_Model.Loss.LossFunction import CenterLoss, Wasserstein_Loss, MultiCenterLoss, CenterCosLoss, RingLoss, \
    VarianceLoss, DistributeLoss, MMD_Loss
from Define_Model.Loss.SoftmaxLoss import AngleSoftmaxLoss, AngleLinear, AdditiveMarginLinear, AMSoftmaxLoss, \
    ArcSoftmaxLoss, \
    GaussianLoss, MinArcSoftmaxLoss, MinArcSoftmaxLoss_v2
from Define_Model.Optimizer import EarlyStopping
from Process_Data.Datasets.KaldiDataset import KaldiExtractDataset, \
    ScriptVerifyDataset
from Process_Data.Datasets.LmdbDataset import EgsDataset
import Process_Data.constants as C
from Process_Data.audio_processing import ConcateVarInput, tolog, ConcateOrgInput, PadCollate, read_Waveform
from Process_Data.audio_processing import toMFB, totensor, truncatedinput
from TrainAndTest.common_func import create_optimizer, create_model, verification_test, verification_extract, \
    args_parse, args_model, save_model_args
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
# args = args_parse('PyTorch Speaker Recognition: Classification')

parser = argparse.ArgumentParser(description='PyTorch ( Distributed ) Speaker Recognition: Classification')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')

parser.add_argument('--train-config', default='', type=str, help='node rank for distributed training')
parser.add_argument('--seed', type=int, default=123456, help='random seed (default: 0)')

args = parser.parse_args()

# Set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '29555'

# args.cuda = not args.no_cuda and torch.cuda.is_available()
# setting seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

# torch.multiprocessing.set_sharing_strategy('file_system')
torch.distributed.init_process_group(backend='nccl')  # ,
# init_method='file:///home/yangwenhao/lstm_speaker_verification/data/sharedfile2',
# rank=args.local_rank,
# world_size=2)
torch.cuda.set_device(args.local_rank)

# load train config file
# args.train_config
with open(args.train_config, 'r') as f:
    # config_args = yaml.load(f, Loader=yaml.FullLoader)
    config_args = load_hyperpyyaml(f)

# create logger
# Define visulaize SummaryWriter instance
check_path = config_args['check_path'] + '/' + str(args.seed)

if torch.distributed.get_rank() == 0:
    if not os.path.exists(check_path):
        print('Making checkpath...')
        os.makedirs(check_path)

    writer = SummaryWriter(logdir=check_path, filename_suffix='SV')
    sys.stdout = NewLogger(
        os.path.join(check_path, 'log.%s.txt' % time.strftime("%Y.%m.%d", time.localtime())))

kwargs = {'num_workers': config_args['nj'], 'pin_memory': False}  # if args.cuda else {}
extract_kwargs = {'num_workers': 4, 'pin_memory': False}  # if args.cuda else {}

opt_kwargs = {'lr': config_args['lr'], 'lr_decay': config_args['lr_decay'],
              'weight_decay': config_args['weight_decay'],
              'dampening': config_args['dampening'],
              'momentum': config_args['momentum'],
              'nesterov': config_args['nesterov']}

l2_dist = nn.CosineSimilarity(dim=1, eps=1e-12) if config_args['cos_sim'] else nn.PairwiseDistance(p=2)

transform = transforms.Compose([
    totensor()
])

if config_args['test_input'] == 'var':
    transform_V = transforms.Compose([
        ConcateOrgInput(remove_vad=config_args['remove_vad'], feat_type=config_args['feat_format']),
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

# pdb.set_trace()
if config_args['feat_format'] in ['kaldi', 'wav']:
    # file_loader = read_mat
    file_loader = load_mat
elif config_args['feat_format'] == 'npy':
    file_loader = np.load

# torch.multiprocessing.set_sharing_strategy('file_system')

train_dir = EgsDataset(dir=config_args['train_dir'], feat_dim=config_args['input_dim'], loader=file_loader,
                       transform=transform,
                       batch_size=config_args['batch_size'], random_chunk=config_args['random_chunk'],
                       verbose=1 if torch.distributed.get_rank() == 0 else 0)
valid_dir = EgsDataset(dir=config_args['valid_dir'], feat_dim=config_args['input_dim'], loader=file_loader,
                       transform=transform,
                       verbose=1 if torch.distributed.get_rank() == 0 else 0)

if config_args['feat_format'] == 'wav':
    file_loader = read_Waveform
    feat_type = 'wav'
else:
    feat_type = 'kaldi'

train_extract_dir = KaldiExtractDataset(dir=config_args['train_test_dir'],
                                        transform=transform_V,
                                        filer_loader=file_loader, feat_type=feat_type,
                                        trials_file=config_args['train_trials'])

extract_dir = KaldiExtractDataset(dir=config_args['test_dir'], transform=transform_V, feat_type=feat_type,
                                  trials_file=config_args['trials'], filer_loader=file_loader)


def train(train_loader, model, ce, optimizer, epoch, scheduler):
    # switch to evaluate mode
    model.train()

    correct = 0.
    total_datasize = 0.
    total_loss = 0.
    orth_err = 0
    other_loss = 0.

    ce_criterion, xe_criterion = ce
    pbar = tqdm(enumerate(train_loader))
    output_softmax = nn.Softmax(dim=1)
    lambda_ = (epoch / config_args['epochs']) ** 2

    # start_time = time.time()
    # pdb.set_trace()
    for batch_idx, (data, label) in pbar:
        if torch.cuda.is_available():
            # label = label.cuda(non_blocking=True)
            # data = data.cuda(non_blocking=True)
            label = label.cuda()
            data = data.cuda()

        data, label = Variable(data), Variable(label)
        # pdb.set_trace()
        classfier, feats = model(data)
        # cos_theta, phi_theta = classfier
        classfier_label = classfier
        # print('max logit is ', classfier_label.max())

        if config_args['loss_type'] == 'soft':
            loss = ce_criterion(classfier, label)
        elif config_args['loss_type'] == 'asoft':
            classfier_label, _ = classfier
            loss = xe_criterion(classfier, label)
        elif config_args['loss_type'] in ['center', 'mulcenter', 'gaussian', 'coscenter', 'variance']:
            loss_cent = ce_criterion(classfier, label)
            loss_xent = config_args['loss_ratio'] * xe_criterion(feats, label)
            other_loss += loss_xent

            loss = loss_xent + loss_cent
        elif config_args['loss_type'] == 'ring':
            loss_cent = ce_criterion(classfier, label)
            loss_xent = config_args['loss_ratio'] * xe_criterion(feats)

            other_loss += loss_xent
            loss = loss_xent + loss_cent
        elif config_args['loss_type'] in ['amsoft', 'arcsoft', 'minarcsoft', 'minarcsoft2', 'subarc', ]:
            loss = xe_criterion(classfier, label)
        elif 'arcdist' in config_args['loss_type']:
            # pdb.set_trace()
            loss_cent = config_args['loss_ratio'] * ce_criterion(classfier, label)
            if 'loss_lambda' in config_args and config_args['loss_lambda']:
                loss_cent = loss_cent * lambda_

            loss_xent = xe_criterion(classfier, label)

            other_loss += loss_cent
            loss = loss_xent + loss_cent

        predicted_labels = output_softmax(classfier_label)
        predicted_one_labels = torch.max(predicted_labels, dim=1)[1]

        if 'lncl' in config_args and config_args['lncl']:
            if config_args['loss_type'] in ['amsoft', 'arcsoft', 'minarcsoft', 'minarcsoft2', 'subarc', 'arcdist']:
                predict_loss = xe_criterion(classfier, predicted_one_labels)
            else:
                predict_loss = ce_criterion(classfier, predicted_one_labels)

            alpha_t = np.clip(config_args['alpha_t'] * (epoch / config_args['epochs']) ** 2, a_min=0, a_max=1)
            mp = predicted_labels.mean(dim=0) * predicted_labels.shape[1]

            loss = (1 - alpha_t) * loss + alpha_t * predict_loss + config_args['beta'] * torch.mean(-torch.log(mp))

        minibatch_correct = float((predicted_one_labels.cpu() == label.cpu()).sum().item())
        minibatch_acc = minibatch_correct / len(predicted_one_labels)
        correct += minibatch_correct

        total_datasize += len(predicted_one_labels)
        total_loss += float(loss.item())
        if torch.distributed.get_rank() == 0:
            writer.add_scalar('Train/All_Loss', float(loss.item()), int((epoch - 1) * len(train_loader) + batch_idx + 1))

        if np.isnan(loss.item()):
            raise ValueError('Loss value is NaN!')

        # compute gradient and update weights
        loss.backward()

        if 'grad_clip' in config_args and config_args['grad_clip'] > 0:
            this_lr = config_args['lr']
            for param_group in optimizer.param_groups:
                this_lr = min(param_group['lr'], this_lr)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config_args['grad_clip'])

        if ((batch_idx + 1) % config_args['accu_steps']) == 0:
            # optimizer the net
            optimizer.step()  # update parameters of net
            optimizer.zero_grad()  # reset gradient

            if config_args['model'] == 'FTDNN' and ((batch_idx + 1) % 4) == 0:
                if isinstance(model, DistributedDataParallel):
                    model.module.step_ftdnn_layers()  # The key method to constrain the first two convolutions, perform after every SGD step
                    orth_err += model.module.get_orth_errors()
                else:
                    model.step_ftdnn_layers()  # The key method to constrain the first two convolutions, perform after every SGD step
                    orth_err += model.get_orth_errors()

        # optimizer.zero_grad()
        # loss.backward()

        if config_args['loss_ratio'] != 0:
            if config_args['loss_type'] in ['center', 'mulcenter', 'gaussian', 'coscenter']:
                for param in xe_criterion.parameters():
                    param.grad.data *= (1. / config_args['loss_ratio'])

        # optimizer.step()
        if config_args['scheduler'] == 'cyclic':
            scheduler.step()

        # if torch.distributed.get_rank() == 0:
        if (batch_idx + 1) % config_args['log_interval'] == 0:
            epoch_str = 'Train Epoch {}: [{:8d}/{:8d} ({:3.0f}%)]'.format(epoch, batch_idx * len(data),
                                                                          len(train_loader.dataset),
                                                                          100. * batch_idx / len(train_loader))

            if len(config_args['random_chunk']) == 2 and config_args['random_chunk'][0] <= \
                    config_args['random_chunk'][
                        1]:
                batch_length = data.shape[-1] if config_args['feat_format'] == 'wav' else data.shape[-2]
                epoch_str += ' Batch Len: {:>3d}'.format(batch_length)

            epoch_str += ' Accuracy(%): {:>6.2f}%'.format(100. * minibatch_acc)

            if orth_err > 0:
                epoch_str += ' Orth_err: {:>5d}'.format(int(orth_err))

            if config_args['loss_type'] in ['center', 'variance', 'mulcenter', 'gaussian', 'coscenter']:
                epoch_str += ' Center Loss: {:.4f}'.format(loss_xent.float())
            if 'arcdist' in config_args['loss_type']:
                epoch_str += ' Dist Loss: {:.4f}'.format(loss_cent.float())
            epoch_str += ' Avg Loss: {:.4f}'.format(total_loss / (batch_idx + 1))

            pbar.set_description(epoch_str)

    if config_args['batch_shuffle']:
        train_dir.__shuffle__()

    this_epoch_str = 'Epoch {:>2d}: \33[91mTrain Accuracy: {:.6f}%, Avg loss: {:6f}'.format(epoch, 100 * float(
        correct) / total_datasize, total_loss / len(train_loader))

    if other_loss != 0:
        this_epoch_str += ' {} Loss: {:6f}'.format(config_args['loss_type'], other_loss / len(train_loader))

    this_epoch_str += '.\33[0m'

    if torch.distributed.get_rank() == 0:
        print(this_epoch_str)
        writer.add_scalar('Train/Accuracy', correct / total_datasize, epoch)
        writer.add_scalar('Train/Loss', total_loss / len(train_loader), epoch)

    torch.cuda.empty_cache()


def valid_class(valid_loader, model, ce, epoch):
    # switch to evaluate mode
    model.eval()

    total_loss = 0.
    other_loss = 0.
    ce_criterion, xe_criterion = ce
    softmax = nn.Softmax(dim=1)

    correct = 0.
    total_datasize = 0.
    lambda_ = (epoch / config_args['epochs']) ** 2

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(valid_loader):

            if torch.cuda.is_available():
                data = data.cuda()
                label = label.cuda()

            # compute output
            # pdb.set_trace()
            out, feats = model(data)
            if config_args['loss_type'] == 'asoft':
                predicted_labels, _ = out
            else:
                predicted_labels = out

            classfier = predicted_labels
            if config_args['loss_type'] == 'soft':
                loss = ce_criterion(classfier, label)
            elif config_args['loss_type'] == 'asoft':
                classfier_label, _ = classfier
                loss = xe_criterion(classfier, label)
            elif config_args['loss_type'] in ['variance', 'center', 'mulcenter', 'gaussian', 'coscenter']:
                loss_cent = ce_criterion(classfier, label)
                loss_xent = config_args['loss_ratio'] * xe_criterion(feats, label)
                other_loss += float(loss_xent.item())

                loss = loss_xent + loss_cent
            elif config_args['loss_type'] in ['amsoft', 'arcsoft', 'minarcsoft', 'minarcsoft2', 'subarc']:
                loss = xe_criterion(classfier, label)
            elif 'arcdist' in config_args['loss_type']:
                loss_cent = config_args['loss_ratio'] * ce_criterion(classfier, label)
                if 'loss_lambda' in config_args:
                    loss_cent = loss_cent * lambda_

                loss_xent = xe_criterion(classfier, label)

                other_loss += float(loss_cent.item())
                loss = loss_xent + loss_cent

            total_loss += float(loss.item())
            # pdb.set_trace()
            predicted_one_labels = softmax(predicted_labels)
            predicted_one_labels = torch.max(predicted_one_labels, dim=1)[1]

            batch_correct = (predicted_one_labels.cuda() == label).sum().item()
            correct += batch_correct
            total_datasize += len(predicted_one_labels)

    total_batch = len(valid_loader)
    all_total_loss = [None for _ in range(torch.distributed.get_world_size())]
    all_correct = [None for _ in range(torch.distributed.get_world_size())]
    all_total_batch = [None for _ in range(torch.distributed.get_world_size())]
    all_total_datasize = [None for _ in range(torch.distributed.get_world_size())]

    torch.distributed.all_gather_object(all_total_loss, total_loss)
    torch.distributed.all_gather_object(all_correct, correct)
    torch.distributed.all_gather_object(all_total_batch, total_batch)
    torch.distributed.all_gather_object(all_total_datasize, total_datasize)

    total_loss = np.sum(all_total_loss)
    correct = np.sum(all_correct)
    total_batch = np.sum(all_total_batch)
    total_datasize = np.sum(all_total_datasize)

    valid_loss = total_loss / total_batch
    valid_accuracy = 100. * correct / total_datasize

    if torch.distributed.get_rank() == 0:
        writer.add_scalar('Train/Valid_Loss', valid_loss, epoch)
        writer.add_scalar('Train/Valid_Accuracy', valid_accuracy, epoch)
        torch.cuda.empty_cache()

        this_epoch_str = '          \33[91mValid Accuracy: {:.6f}%, Avg loss: {:.6f}'.format(valid_accuracy, valid_loss)

        if other_loss != 0:
            this_epoch_str += ' {} Loss: {:6f}'.format(config_args['loss_type'], other_loss / len(valid_loader))

        this_epoch_str += '.\33[0m'
        print(this_epoch_str)

    return valid_loss


def valid_test(train_extract_loader, model, epoch, xvector_dir):
    # switch to evaluate mode
    model.eval()

    this_xvector_dir = "%s/train/epoch_%s" % (xvector_dir, epoch)
    verification_extract(train_extract_loader, model, this_xvector_dir, epoch, test_input=config_args['test_input'])

    verify_dir = ScriptVerifyDataset(dir=config_args['train_test_dir'], trials_file=config_args['train_trials'],
                                     xvectors_dir=this_xvector_dir,
                                     loader=read_vec_flt)
    verify_loader = torch.utils.data.DataLoader(verify_dir, batch_size=128, shuffle=False, **kwargs)
    eer, eer_threshold, mindcf_01, mindcf_001 = verification_test(test_loader=verify_loader,
                                                                  dist_type=('cos' if config_args['cos_sim'] else 'l2'),
                                                                  log_interval=config_args['log_interval'],
                                                                  xvector_dir=this_xvector_dir,
                                                                  epoch=epoch)
    mix3 = 100. * eer * mindcf_01 * mindcf_001

    if torch.distributed.get_rank() == 0:
        print('          \33[91mTrain EER: {:.4f}%, Threshold: {:.4f}, ' \
              'mindcf-0.01: {:.4f}, mindcf-0.001: {:.4f}, mix: {:.4f}. \33[0m'.format(100. * eer,
                                                                                      eer_threshold,
                                                                                      mindcf_01,
                                                                                      mindcf_001, mix3))

        writer.add_scalar('Train/EER', 100. * eer, epoch)
        writer.add_scalar('Train/Threshold', eer_threshold, epoch)
        writer.add_scalar('Train/mindcf-0.01', mindcf_01, epoch)
        writer.add_scalar('Train/mindcf-0.001', mindcf_001, epoch)
        writer.add_scalar('Train/mix3', mix3, epoch)

    torch.cuda.empty_cache()

    return {'EER': 100. * eer, 'Threshold': eer_threshold,
            'MinDCF_01': mindcf_01, 'MinDCF_001': mindcf_001, 'mix3': mix3}


def test(extract_loader, model, epoch, xvector_dir):
    this_xvector_dir = "%s/test/epoch_%s" % (xvector_dir, epoch)

    verification_extract(extract_loader, model, this_xvector_dir, epoch, test_input=config_args['test_input'])

    verify_dir = ScriptVerifyDataset(dir=config_args['test_dir'], trials_file=config_args['trials'],
                                     xvectors_dir=this_xvector_dir,
                                     loader=read_vec_flt)

    verify_sampler = torch.utils.data.distributed.DistributedSampler(verify_dir)
    # sampler = verify_sampler,
    verify_loader = torch.utils.data.DataLoader(verify_dir, batch_size=128, shuffle=False,
                                                **extract_kwargs)

    # pdb.set_trace()
    eer, eer_threshold, mindcf_01, mindcf_001 = verification_test(test_loader=verify_loader,
                                                                  dist_type='cos' if config_args[
                                                                      'cos_sim'] else 'l2',
                                                                  log_interval=config_args['log_interval'],
                                                                  xvector_dir=this_xvector_dir,
                                                                  epoch=epoch)

    if torch.distributed.get_rank() == 0:
        print(
            '          \33[91mTest  ERR: {:.4f}%, Threshold: {:.4f}, mindcf-0.01: {:.4f}, mindcf-0.001: {:.4f}.\33[0m\n'.format(
                100. * eer, eer_threshold, mindcf_01, mindcf_001))

        writer.add_scalar('Test/EER', 100. * eer, epoch)
        writer.add_scalar('Test/Threshold', eer_threshold, epoch)
        writer.add_scalar('Test/mindcf-0.01', mindcf_01, epoch)
        writer.add_scalar('Test/mindcf-0.001', mindcf_001, epoch)


def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive
    # test_display_triplet_distance = False
    # print the experiment configuration
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('\nCurrent time is \33[91m{}\33[0m.'.format(str(time.asctime())))
        # opts = vars(config_args)
        # keys = list(config_args.keys())
        # keys.sort()
        # options = ["\'%s\': \'%s\'" % (str(k), str(config_args[k])) for k in keys]
        # print('Parsed options: \n{ %s }' % (', '.join(options)))
        print('Number of Speakers: {}.\n'.format(train_dir.num_spks))
        print('Testing with %s distance, ' % ('cos' if config_args['cos_sim'] else 'l2'))
    # model = create_model(config_args['model'], **model_kwargs)
    model = config_args['embedding_model']

    if 'classifier' in config_args:
        model.classifier = config_args['classifier']

    # model_yaml_path = os.path.join(args.check_path, 'model.%s.yaml' % time.strftime("%Y.%m.%d", time.localtime()))
    # save_model_args(model_kwargs, model_yaml_path)
    # exit(0)

    start_epoch = 0
    check_path = config_args['check_path'] + '/' + str(args.seed)
    if 'finetune' not in config_args or not config_args['finetune']:
        this_check_path = '{}/checkpoint_{}_{}.pth'.format(check_path, start_epoch,
                                                           time.strftime('%Y_%b_%d_%H:%M', time.localtime()))
        if not os.path.exists(this_check_path):
            torch.save({'state_dict': model.state_dict()}, this_check_path)

    # Load checkpoint
    iteration = 0  # if args.resume else 0
    if 'fintune' in config_args:
        if os.path.isfile(config_args['resume']):
            print('=> loading checkpoint {}'.format(config_args['resume']))
            checkpoint = torch.load(config_args['resume'])
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
            print('=> no checkpoint found at {}'.format(config_args['resume']))

    ce_criterion = nn.CrossEntropyLoss()
    if config_args['loss_type'] == 'soft':
        xe_criterion = None
    elif config_args['loss_type'] == 'asoft':
        ce_criterion = None
        xe_criterion = AngleSoftmaxLoss(lambda_min=config_args['lambda_min'], lambda_max=config_args['lambda_max'])
    elif config_args['loss_type'] == 'center':
        xe_criterion = CenterLoss(num_classes=train_dir.num_spks, feat_dim=config_args['embedding_size'])
    elif config_args['loss_type'] == 'variance':
        xe_criterion = VarianceLoss(num_classes=train_dir.num_spks, feat_dim=config_args['embedding_size'])
    elif config_args['loss_type'] == 'gaussian':
        xe_criterion = GaussianLoss(num_classes=train_dir.num_spks, feat_dim=config_args['embedding_size'])
    elif config_args['loss_type'] == 'coscenter':
        xe_criterion = CenterCosLoss(num_classes=train_dir.num_spks, feat_dim=config_args['embedding_size'])
    elif config_args['loss_type'] == 'mulcenter':
        xe_criterion = MultiCenterLoss(num_classes=train_dir.num_spks, feat_dim=config_args['embedding_size'],
                                       num_center=config_args['num_center'])
    elif config_args['loss_type'] == 'amsoft':
        ce_criterion = None
        xe_criterion = AMSoftmaxLoss(margin=config_args['margin'], s=config_args['s'])

    elif config_args['loss_type'] in ['arcsoft', 'subarc']:
        ce_criterion = None
        if 'class_weight' in config_args and config_args['class_weight'] == 'cnc1':
            class_weight = torch.tensor(C.CNC1_WEIGHT)
            if len(class_weight) != train_dir.num_spks:
                class_weight = None
        else:
            class_weight = None

        all_iteraion = 0 if 'all_iteraion' not in config_args else config_args['all_iteraion']
        smooth_ratio = 0 if 'smooth_ratio' not in config_args else config_args['smooth_ratio']
        xe_criterion = ArcSoftmaxLoss(margin=config_args['margin'], s=config_args['s'], iteraion=iteration,
                                      all_iteraion=all_iteraion,
                                      smooth_ratio=smooth_ratio,
                                      class_weight=class_weight)
    elif config_args['loss_type'] == 'minarcsoft':
        ce_criterion = None
        xe_criterion = MinArcSoftmaxLoss(margin=config_args['margin'], s=config_args['s'], iteraion=iteration,
                                         all_iteraion=config_args['all_iteraion'])
    elif config_args['loss_type'] == 'minarcsoft2':
        ce_criterion = None
        xe_criterion = MinArcSoftmaxLoss_v2(margin=config_args['margin'], s=config_args['s'], iteraion=iteration,
                                            all_iteraion=config_args['all_iteraion'])
    elif config_args['loss_type'] == 'wasse':
        xe_criterion = Wasserstein_Loss(source_cls=config_args['source_cls'])
    elif config_args['loss_type'] == 'mmd':
        xe_criterion = MMD_Loss()
    elif config_args['loss_type'] == 'ring':
        xe_criterion = RingLoss(ring=config_args['ring'])
        args.alpha = 0.0
    elif 'arcdist' in config_args['loss_type']:
        ce_criterion = DistributeLoss(stat_type=config_args['stat_type'], margin=config_args['m'])
        xe_criterion = ArcSoftmaxLoss(margin=config_args['margin'], s=config_args['s'], iteraion=iteration,
                                      all_iteraion=config_args['all_iteraion'])

    model_para = [{'params': model.parameters()}]
    if config_args['loss_type'] in ['center', 'variance', 'mulcenter', 'gaussian', 'coscenter', 'ring']:
        assert config_args['lr_ratio'] > 0
        model_para.append({'params': xe_criterion.parameters(), 'lr': config_args['lr'] * config_args['lr_ratio']})

    if 'second_wd' in config_args and config_args['config_args'] > 0:
        # if config_args['loss_type in ['asoft', 'amsoft']:
        classifier_params = list(map(id, model.classifier.parameters()))
        rest_params = filter(lambda p: id(p) not in classifier_params, model.parameters())
        init_lr = config_args['lr'] * config_args['lr_ratio'] if config_args['lr_ratio'] > 0 else config_args['lr']
        init_wd = config_args['second_wd'] if config_args['second_wd'] > 0 else config_args['weight_decay']
        print('Set the lr and weight_decay of classifier to %f and %f' % (init_lr, init_wd))
        model_para = [{'params': rest_params},
                      {'params': model.classifier.parameters(), 'lr': init_lr, 'weight_decay': init_wd}]

    if 'filter' in config_args:
        if config_args['filter'] in ['fDLR', 'fBLayer', 'fLLayer', 'fBPLayer', 'sinc2down']:
            filter_params = list(map(id, model.filter_layer.parameters()))
            rest_params = filter(lambda p: id(p) not in filter_params, model_para[0]['params'])
            init_wd = config_args['filter_wd'] if args.filter_wd > 0 else config_args['weight_decay']
            init_lr = config_args['lr'] * config_args['lr_ratio'] if config_args['lr_ratio'] > 0 else config_args['lr']
            print('Set the lr and weight_decay of filter layer to %f and %f' % (init_lr, init_wd))
            model_para[0]['params'] = rest_params
            model_para.append({'params': model.filter_layer.parameters(), 'lr': init_lr,
                               'weight_decay': init_wd})

    optimizer = create_optimizer(model_para, config_args['optimizer'], **opt_kwargs)
    early_stopping_scheduler = EarlyStopping(patience=config_args['early_patience'],
                                             min_delta=config_args['early_delta'])

    milestones = config_args['milestones']
    if config_args['scheduler'] == 'exp':
        gamma = np.power(config_args['base_lr'] / config_args['lr'],
                         1 / config_args['epochs']) if config_args['gamma'] == 0 else config_args['gamma']
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif config_args['scheduler'] == 'rop':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=config_args['patience'], min_lr=1e-5)
    elif config_args['scheduler'] == 'cyclic':
        cycle_momentum = False if config_args['optimizer'] == 'adam' else True
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=config_args['base_lr'],
                                          max_lr=config_args['lr'],
                                          step_size_up=config_args['cyclic_epoch'] * int(
                                              np.ceil(len(train_dir) / config_args['batch_size'])),
                                          cycle_momentum=cycle_momentum,
                                          mode='triangular2')
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # Save model config txt
    if torch.distributed.get_rank() == 0:
        with open(os.path.join(check_path,
                               'model.%s.conf' % time.strftime("%Y.%m.%d", time.localtime())),
                  'w') as f:
            f.write('model: ' + str(model) + '\n')
            f.write('CrossEntropy: ' + str(ce_criterion) + '\n')
            f.write('Other Loss: ' + str(xe_criterion) + '\n')
            f.write('Optimizer: ' + str(optimizer) + '\n')
            f.write('Scheduler: ' + str(scheduler) + '\n')

    if 'resume' in config_args:
        if os.path.isfile(config_args['resume']):
            if torch.distributed.get_rank() == 0:
                print('=> loading checkpoint {}'.format(config_args['resume']))
            checkpoint = torch.load(config_args['resume'])
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

            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            if 'optimizer' in optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])

            # model.dropout.p = args.dropout_p
        else:
            if torch.distributed.get_rank() == 0:
                print('=> no checkpoint found at {}'.format(config_args['resume']))

    ce = [ce_criterion, xe_criterion]

    start = 1 + start_epoch
    if torch.distributed.get_rank() == 0:
        print('Start epoch is : ' + str(start))
    # start = 0
    end = start + config_args['epochs']

    if len(config_args['random_chunk']) == 2 and config_args['random_chunk'][0] <= config_args['random_chunk'][1]:
        min_chunk_size = int(config_args['random_chunk'][0])
        max_chunk_size = int(config_args['random_chunk'][1])
        pad_dim = 2 if config_args['feat_format'] == 'kaldi' else 3

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dir)
        train_loader = torch.utils.data.DataLoader(train_dir, batch_size=config_args['batch_size'],
                                                   collate_fn=PadCollate(dim=pad_dim,
                                                                         num_batch=int(
                                                                             np.ceil(
                                                                                 len(train_dir) / config_args[
                                                                                     'batch_size'])),
                                                                         min_chunk_size=min_chunk_size,
                                                                         max_chunk_size=max_chunk_size,
                                                                         chisquare=False if 'chisquare' not in config_args else
                                                                         config_args['chisquare'],
                                                                         verbose=1 if torch.distributed.get_rank() == 0 else 0),
                                                   shuffle=config_args['shuffle'], sampler=train_sampler, **kwargs)

        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dir)
        valid_loader = torch.utils.data.DataLoader(valid_dir, batch_size=int(config_args['batch_size'] / 2),
                                                   collate_fn=PadCollate(dim=pad_dim, fix_len=True,
                                                                         min_chunk_size=min_chunk_size,
                                                                         max_chunk_size=max_chunk_size,
                                                                         verbose=1 if torch.distributed.get_rank() == 0 else 0),
                                                   shuffle=False, sampler=valid_sampler, **kwargs)

        extract_sampler = torch.utils.data.distributed.DistributedSampler(extract_dir)
        # sampler = extract_sampler,
        extract_loader = torch.utils.data.DataLoader(extract_dir, batch_size=1, shuffle=False,
                                                     sampler=extract_sampler, **extract_kwargs)


    else:
        train_loader = torch.utils.data.DataLoader(train_dir, batch_size=config_args['batch_size'],
                                                   shuffle=config_args['shuffle'],
                                                   **kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_dir, batch_size=int(config_args['batch_size'] / 2),
                                                   shuffle=False,
                                                   **kwargs)
        extract_loader = torch.utils.data.DataLoader(extract_dir, batch_size=1, shuffle=False,
                                                     **extract_kwargs)

    train_extract_sampler = torch.utils.data.distributed.DistributedSampler(train_extract_dir)
    train_extract_loader = torch.utils.data.DataLoader(train_extract_dir, batch_size=1, shuffle=False,
                                                       sampler=train_extract_sampler, **extract_kwargs)

    # if config_args['cuda']:
    if len(config_args['gpu_id']) > 1:
        print("Continue with gpu: %s ..." % str(args.local_rank))
        # torch.distributed.init_process_group(backend="nccl",
        #                                      init_method='file:///home/yangwenhao/lstm_speaker_verification/data/sharedfile',
        #                                      rank=0,
        #                                      world_size=1)
        #
        # try:
        #     torch.distributed.init_process_group(backend="nccl", init_method='tcp://localhost:32456', rank=0,
        #                                          world_size=1)
        # except RuntimeError as r:
        #     torch.distributed.init_process_group(backend="nccl", init_method='tcp://localhost:32454', rank=0,
        #                                          world_size=1)
        # if args.gain
        # model = DistributedDataParallel(model.cuda(), find_unused_parameters=True)
        model = DistributedDataParallel(model.cuda(), device_ids=[args.local_rank])


    else:
        model = model.cuda()

    for i in range(len(ce)):
        if ce[i] != None:
            ce[i] = ce[i].cuda()
    try:
        print('Dropout is {}.'.format(model.dropout_p))
    except:
        pass

    xvector_dir = check_path
    xvector_dir = xvector_dir.replace('checkpoint', 'xvector')
    start_time = time.time()

    all_lr = []
    valid_test_result = []
    try:
        for epoch in range(start, end):

            # if torch. is_distributed():
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)
            train_extract_sampler.set_epoch(epoch)
            extract_sampler.set_epoch(epoch)

            # pdb.set_trace()
            # if torch.distributed.get_rank() == 0:
            lr_string = '\33[1;34m Ranking {}: Current \'{}\' learning rate is '.format(torch.distributed.get_rank(),
                                                                                        config_args['optimizer'])
            this_lr = []
            for param_group in optimizer.param_groups:
                this_lr.append(param_group['lr'])
                lr_string += '{:.10f} '.format(param_group['lr'])
            print('%s \33[0m' % lr_string)
            all_lr.append(this_lr[0])
            if torch.distributed.get_rank() == 0:
                writer.add_scalar('Train/lr', this_lr[0], epoch)

            torch.distributed.barrier()
            if not torch.distributed.is_initialized():
                break
            train(train_loader, model, ce, optimizer, epoch, scheduler)
            valid_loss = valid_class(valid_loader, model, ce, epoch)
            if config_args['early_stopping'] or (
                    epoch % config_args['test_interval'] == 1 or epoch in milestones or epoch == (end - 1)):
                valid_test_dict = valid_test(train_extract_loader, model, epoch, xvector_dir)
            else:
                valid_test_dict = {}

            flag_tensor = torch.zeros(1).cuda()
            valid_test_dict['Valid_Loss'] = valid_loss
            if torch.distributed.get_rank() == 0:
                valid_test_result.append(valid_test_dict)

            if torch.distributed.get_rank() == 0 and config_args['early_stopping']:
                early_stopping_scheduler(valid_test_dict[config_args['early_meta']], epoch)

                if early_stopping_scheduler.best_epoch + early_stopping_scheduler.patience >= end:
                    early_stopping_scheduler.early_stop = True

                if config_args['scheduler'] != 'cyclic' and this_lr[0] <= 0.1 ** 3 * config_args['lr']:
                    if len(all_lr) > 5 and all_lr[-5] >= this_lr[0]:
                        early_stopping_scheduler.early_stop = True

                # if torch.distributed.get_rank() == 0:
                #     flag_tensor += 1

            if torch.distributed.get_rank() == 0 and (
                    epoch % config_args['test_interval'] == 1 or epoch in milestones or epoch == (
                    end - 1) or early_stopping_scheduler.best_epoch == epoch):

                # if (epoch == 1 or epoch != (end - 2)) and (
                #     epoch % config_args['test_interval'] == 1 or epoch in milestones or epoch == (end - 1)):
                model.eval()
                this_check_path = '{}/checkpoint_{}.pth'.format(check_path, epoch)
                model_state_dict = model.module.state_dict() \
                    if isinstance(model, DistributedDataParallel) else model.state_dict()
                torch.save({'epoch': epoch, 'state_dict': model_state_dict,
                            'scheduler': scheduler.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'criterion': ce}, this_check_path)

                # valid_test(train_extract_loader, model, epoch, xvector_dir)
                # test(extract_loader, model, epoch, xvector_dir)

                if config_args['early_stopping']:
                    pass
                # elif early_stopping_scheduler.best_epoch == epoch or (
                #         args.early_stopping == False and epoch % args.test_interval == 1):
                elif epoch % config_args['test_interval'] == 1:
                    test(extract_loader, model, epoch, xvector_dir)

                if early_stopping_scheduler.early_stop:
                    print('Best Epoch is %d:' % (early_stopping_scheduler.best_epoch))
                    best_epoch = early_stopping_scheduler.best_epoch
                    best_res = valid_test_result[int(best_epoch - 1)]

                    best_str = 'EER(%):       ' + '{:>6.2f} '.format(best_res['EER'])
                    best_str += '   Threshold: ' + '{:>7.4f} '.format(best_res['Threshold'])
                    best_str += ' MinDcf-0.01: ' + '{:.4f} '.format(best_res['MinDCF_01'])
                    best_str += ' MinDcf-0.001: ' + '{:.4f} '.format(best_res['MinDCF_001'])
                    best_str += ' Mix3: ' + '{:.4f}\n'.format(best_res['mix3'])
                    print(best_str)

                    try:
                        shutil.copy('{}/checkpoint_{}.pth'.format(check_path,
                                                                  early_stopping_scheduler.best_epoch),
                                    '{}/best.pth'.format(check_path))
                    except Exception as e:
                        print(e)

                    flag_tensor += 1

            dist.all_reduce(flag_tensor, op=dist.ReduceOp.SUM)
            # torch.distributed.barrier()
            if flag_tensor >= 1:
                end = epoch
                # print('Rank      ', torch.distributed.get_rank(), '      stopped')
                break

            if config_args['scheduler'] == 'rop':
                scheduler.step(valid_loss)
            elif config_args['scheduler'] == 'cyclic':
                continue
            else:
                scheduler.step()

    except KeyboardInterrupt:
        end = epoch

    stop_time = time.time()
    t = float(stop_time - start_time)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        writer.close()
        print("Running %.4f minutes for each epoch.\n" % (t / 60 / (max(end - start, 1))))
    # torch.distributed.des
    exit(0)


if __name__ == '__main__':
    main()
