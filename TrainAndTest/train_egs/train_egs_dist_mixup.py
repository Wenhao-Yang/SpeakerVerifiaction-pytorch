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
import torch._utils

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
from Light.dataset import Sampler_Loaders, SubDatasets
from Light.model import SpeakerLoss
from TrainAndTest.train_egs.train_egs_dist import all_seed
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
    GaussianLoss, MinArcSoftmaxLoss, MinArcSoftmaxLoss_v2, MixupLoss
from Define_Model.Optimizer import EarlyStopping
from Process_Data.Datasets.KaldiDataset import KaldiExtractDataset, \
    ScriptVerifyDataset
from Process_Data.Datasets.LmdbDataset import EgsDataset
import Process_Data.constants as C
from Process_Data.audio_processing import ConcateVarInput, tolog, ConcateOrgInput, PadCollate, read_Waveform
from Process_Data.audio_processing import toMFB, totensor, truncatedinput
from TrainAndTest.common_func import create_optimizer, create_classifier, create_scheduler, verification_test, verification_extract, \
    args_parse, args_model, save_model_args
from TrainAndTest.train_egs.train_egs_dist import valid_class, valid_test
from logger import NewLogger

warnings.filterwarnings("ignore")


try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(
            storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor

    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

# Set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '29555'

def train(train_loader, model, optimizer, epoch, scheduler, config_args, writer):
    # switch to evaluate mode
    model.train()

    correct = 0.
    total_datasize = 0.
    total_loss = 0.
    orth_err = 0
    total_other_loss = 0.

    # ce_criterion, xe_criterion = ce
    pbar = tqdm(enumerate(train_loader))
    output_softmax = nn.Softmax(dim=1)
    # lambda_ = (epoch / config_args['epochs']) ** 2

    # start_time = time.time()
    # pdb.set_trace()
    for batch_idx, (data, label) in pbar:

        if 'mix_ratio'in config_args and np.random.uniform(0,1) <= config_args['mix_ratio']:
            lamda_beta = np.random.beta(config_args['lamda_beta'], config_args['lamda_beta'])
            half_data = int(len(data) / 2)

            if config_args['mixup_type'] != 'manifold':
                rand_idx = torch.randperm(half_data)
                mix_data = lamda_beta * data[half_data:] + \
                    (1 - lamda_beta) * data[half_data:][rand_idx]
                data = torch.cat([data[:half_data], mix_data], dim=0)
                label = torch.cat([label, label[half_data:][rand_idx]], dim=0)
            else:
                rand_idx = torch.randperm(half_data)
                if 'index_list' not in config_args:
                    label = torch.cat([label, label[half_data:][rand_idx]], dim=0)
                else:
                    half_label = label[half_data:]
                    rand_label = half_label.clone()[rand_idx]
                    relabel = []
                    for x, y in zip(half_label, rand_label):
                        if x == y:
                            relabel.append(int(x))
                        else:
                            relabel.append(config_args['index_list']['%d_%d' % (x, y)])

                    relabel = torch.LongTensor(relabel)
                    label = torch.cat([label[:half_data], relabel], dim=0)
        else:
            lamda_beta = 0
            rand_idx = None
            half_data = 0


        if torch.cuda.is_available():
            # label = label.cuda(non_blocking=True)
            # data = data.cuda(non_blocking=True)
            label = label.cuda()
            data = data.cuda()

        data, label = Variable(data), Variable(label)
        # pdb.set_trace()
        classfier, feats = model(data, proser=rand_idx, lamda_beta=lamda_beta,
                                 mixup_alpha=config_args['mixup_layer'])

        predicted_labels = output_softmax(classfier)
        predicted_one_labels = torch.max(predicted_labels, dim=1)[1]

        loss, other_loss = model.loss(classfier, feats, label, 
                                      half_data=half_data, lamda_beta=lamda_beta)

        predicted_one_labels = predicted_one_labels.cpu()
        label = label.cpu()
        
        if half_data == 0 or 'index_list' in config_args:
            minibatch_correct = predicted_one_labels.eq(label).cpu().sum().float()
        elif config_args['mixup_type'] == 'manifold':
            # print(predicted_one_labels.shape, label.shape)
            minibatch_correct = predicted_one_labels[:half_data].eq(
                label[:half_data]).cpu().sum().float() + \
                lamda_beta * predicted_one_labels[half_data:].eq(
                label[half_data:(half_data + half_data)]).cpu().sum().float() + \
                (1 - lamda_beta) * predicted_one_labels[half_data:].eq(
                label[-half_data:]).cpu().sum().float()
        else:
            minibatch_correct = lamda_beta * predicted_one_labels.eq(
                label[:len(predicted_one_labels)]).cpu().sum().float() + \
                (1 - lamda_beta) * predicted_one_labels.eq(
                label[-len(predicted_one_labels):]).cpu().sum().float()

        # minibatch_correct = float((predicted_one_labels.cpu() == label.cpu()).sum().item())
        minibatch_acc = minibatch_correct / len(predicted_one_labels)
        correct += minibatch_correct

        total_datasize += len(predicted_one_labels)
        total_loss += float(loss.item())
        total_other_loss += other_loss

        if torch.distributed.get_rank() == 0:
            writer.add_scalar('Train/All_Loss', float(loss.item()),
                              int((epoch - 1) * len(train_loader) + batch_idx + 1))

        if np.isnan(loss.item()):
            pdb.set_trace()
            raise ValueError('Loss value is NaN!')

        # compute gradient and update weights
        loss.backward()

        if 'grad_clip' in config_args and config_args['grad_clip'] > 0:
            this_lr = config_args['lr']
            for param_group in optimizer.param_groups:
                this_lr = min(param_group['lr'], this_lr)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config_args['grad_clip'])

        if ((batch_idx + 1) % config_args['accu_steps']) == 0:
            # optimizer the net
            optimizer.step()  # update parameters of net
            optimizer.zero_grad()  # reset gradient

            if config_args['model'] == 'FTDNN' and ((batch_idx + 1) % 4) == 0:
                if isinstance(model, DistributedDataParallel):
                    # The key method to constrain the first two convolutions, perform after every SGD step
                    model.module.step_ftdnn_layers()
                    orth_err += model.module.get_orth_errors()
                else:
                    # The key method to constrain the first two convolutions, perform after every SGD step
                    model.step_ftdnn_layers()
                    orth_err += model.get_orth_errors()

        # optimizer.zero_grad()
        # loss.backward()

        if config_args['loss_ratio'] != 0:
            if config_args['loss_type'] in ['center', 'mulcenter', 'gaussian', 'coscenter']:
                for param in model.loss.xe_criterion.parameters():
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
                    config_args['random_chunk'][1]:
                batch_length = data.shape[-1] if config_args['feat_format'] == 'wav' else data.shape[-2]
                epoch_str += ' Batch Length: {:>3d}'.format(batch_length)

            epoch_str += ' Lamda: {:>4.2f}'.format(lamda_beta)
            epoch_str += ' Accuracy(%): {:>6.2f}%'.format(100. * minibatch_acc)

            if other_loss != 0:
                epoch_str += ' Other Loss: {:.4f}'.format(other_loss)

            epoch_str += ' Avg Loss: {:.4f}'.format(
                total_loss / (batch_idx + 1))

            pbar.set_description(epoch_str)

    this_epoch_str = 'Epoch {:>2d}: \33[91mTrain Accuracy: {:.6f}%, Avg loss: {:6f}'.format(epoch, 100 * float(
        correct) / total_datasize, total_loss / len(train_loader))

    if total_other_loss != 0:
        this_epoch_str += ' {} Loss: {:6f}'.format(
            config_args['loss_type'], total_other_loss / len(train_loader))

    this_epoch_str += '.\33[0m'

    if torch.distributed.get_rank() == 0:
        print(this_epoch_str)
        writer.add_scalar('Train/Accuracy', correct / total_datasize, epoch)
        writer.add_scalar('Train/Loss', total_loss / len(train_loader), epoch)

    torch.cuda.empty_cache()


def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive
    # Training settings
    parser = argparse.ArgumentParser(
        description='PyTorch ( Distributed ) Speaker Recognition: Classification')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--train-config', default='', type=str,
                        help='node rank for distributed training')
    parser.add_argument('--seed', type=int, default=123456,
                        help='random seed (default: 0)')
    parser.add_argument('--lamda-beta', type=float, default=2.0,
                        help='random seed (default: 0)')

    args = parser.parse_args()

    all_seed(args.seed)
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    # load train config file
    # args.train_config
    with open(args.train_config, 'r') as f:
        # config_args = yaml.load(f, Loader=yaml.FullLoader)
        config_args = load_hyperpyyaml(f)

    # create logger
    # Define visulaize SummaryWriter instance
    if isinstance(config_args['mixup_layer'], list):
        mixup_layer_str = ''.join([str(s) for s in config_args['mixup_layer']])
    else:
        mixup_layer_str = str(config_args['mixup_layer'])

    lambda_str = '_lamda' + str(args.lamda_beta)
    config_args['lamda_beta'] = args.lamda_beta

    mixup_str = '/clsaug_mani' + mixup_layer_str + lambda_str
    if 'mix_ratio'in config_args and config_args['mix_ratio'] < 1:
        mixup_str += '_mix_ratio_' + str(config_args['mix_ratio'])

    if 'mix_type' in config_args and config_args['mix_type'] == 'addup':
        mixup_str += 'addup'

    check_path = config_args['check_path'] + mixup_str + '/' + str(args.seed)
    if torch.distributed.get_rank() == 0:
        if not os.path.exists(check_path):
            print('Making checkpath...')
            os.makedirs(check_path)

        writer = SummaryWriter(logdir=check_path, filename_suffix='SV')
        sys.stdout = NewLogger(
            os.path.join(check_path, 'log.%s.txt' % time.strftime("%Y.%m.%d", time.localtime())))

    # Dataset
    train_dir, valid_dir, train_extract_dir = SubDatasets(config_args)
    train_loader, train_sampler, valid_loader, valid_sampler, train_extract_loader, train_extract_sampler = Sampler_Loaders(
        train_dir, valid_dir, train_extract_dir, config_args)
    
    if 'mix_class' in config_args and config_args['mix_class'] == True:
        index_list = {}
        idx = train_dir.num_spks
        merge_spks = set([])

        if 'mix_type' in config_args and config_args['mix_type'] == 'addup':
            for i in range(train_dir.num_spks):
                for j in range(train_dir.num_spks):
                    if i != j:
                        index_list['%d_%d' % (i, j)] = int(np.floor(idx))
                        merge_spks.add(int(np.floor(idx)))
                        idx += 0.2
                idx = int(np.ceil(idx-0.2))
        else:
            for i in range(train_dir.num_spks):
                for j in range(i+1, train_dir.num_spks):
                    index_list['%d_%d' % (i, j)] = int(np.floor(idx))
                    index_list['%d_%d' % (j, i)] = int(np.floor(idx))

                    merge_spks.add(int(np.floor(idx)))
                    idx += 0.2
                idx = int(np.ceil(idx-0.2))
        config_args['index_list'] = index_list

    torch.distributed.barrier()
    check_path = config_args['check_path'] + mixup_str + '/' + str(args.seed)

    if torch.distributed.get_rank() == 0:
        print('\nCurrent time: \33[91m{}\33[0m.'.format(str(time.asctime())))
        # opts = vars(config_args)
        # keys = list(config_args.keys())
        # keys.sort()
        # options = ["\'%s\': \'%s\'" % (str(k), str(config_args[k])) for k in keys]
        # print('Parsed options: \n{ %s }' % (', '.join(options)))
        print('Number of Speakers: {}.\n'.format(train_dir.num_spks))
        print('Testing with %s distance, ' %
              ('cos' if config_args['cos_sim'] else 'l2'))
        print('Checkpoint path: ', check_path)

    # model = create_model(config_args['model'], **model_kwargs)
    model = config_args['embedding_model']

    if 'classifier' in config_args:
        model.classifier = config_args['classifier']
    else:
        create_classifier(model, **config_args)
        # model_yaml_path = os.path.join(args.check_path, 'model.%s.yaml' % time.strftime("%Y.%m.%d", time.localtime()))
        # save_model_args(model_kwargs, model_yaml_path)
        # exit(0)

    start_epoch = 0
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
            filtered = {k: v for k, v in checkpoint_state_dict.items(
            ) if 'num_batches_tracked' not in k}
            if list(filtered.keys())[0].startswith('module'):
                new_state_dict = OrderedDict()
                for k, v in filtered.items():
                    # remove `module.`，表面从第7个key值字符取到最后一个字符，去掉module.
                    name = k[7:]
                    new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。

                model.load_state_dict(new_state_dict)
            else:
                model_dict = model.state_dict()
                model_dict.update(filtered)
                model.load_state_dict(model_dict)
            # model.dropout.p = args.dropout_p
        else:
            print('=> no checkpoint found at {}'.format(config_args['resume']))

    model.loss = SpeakerLoss(config_args)
    model.loss.xe_criterion = MixupLoss(model.loss.xe_criterion, gamma=config_args['proser_gamma'])

    model_para = [{'params': model.parameters()}]
    if config_args['loss_type'] in ['center', 'variance', 'mulcenter', 'gaussian', 'coscenter', 'ring']:
        assert config_args['lr_ratio'] > 0
        model_para.append({'params': model.loss.xe_criterion.parameters(
        ), 'lr': config_args['lr'] * config_args['lr_ratio']})

    if 'second_wd' in config_args and config_args['config_args'] > 0:
        # if config_args['loss_type in ['asoft', 'amsoft']:
        classifier_params = list(map(id, model.classifier.parameters()))
        rest_params = filter(lambda p: id(
            p) not in classifier_params, model.parameters())
        init_lr = config_args['lr'] * \
            config_args['lr_ratio'] if config_args['lr_ratio'] > 0 else config_args['lr']
        init_wd = config_args['second_wd'] if config_args['second_wd'] > 0 else config_args['weight_decay']
        print('Set the lr and weight_decay of classifier to %f and %f' %
              (init_lr, init_wd))
        model_para = [{'params': rest_params},
                      {'params': model.classifier.parameters(), 'lr': init_lr, 'weight_decay': init_wd}]

    if 'filter' in config_args:
        if config_args['filter'] in ['fDLR', 'fBLayer', 'fLLayer', 'fBPLayer', 'sinc2down']:
            filter_params = list(map(id, model.filter_layer.parameters()))
            rest_params = filter(lambda p: id(
                p) not in filter_params, model_para[0]['params'])
            init_wd = config_args['filter_wd'] if args.filter_wd > 0 else config_args['weight_decay']
            init_lr = config_args['lr'] * \
                config_args['lr_ratio'] if config_args['lr_ratio'] > 0 else config_args['lr']
            print('Set the lr and weight_decay of filter layer to %f and %f' %
                  (init_lr, init_wd))
            model_para[0]['params'] = rest_params
            model_para.append({'params': model.filter_layer.parameters(), 'lr': init_lr,
                               'weight_decay': init_wd})

    opt_kwargs = {'lr': config_args['lr'], 'lr_decay': config_args['lr_decay'],
                    'weight_decay': config_args['weight_decay'],
                    'dampening': config_args['dampening'],
                    'momentum': config_args['momentum'],
                    'nesterov': config_args['nesterov']}

    optimizer = create_optimizer(
        model_para, config_args['optimizer'], **opt_kwargs)
    scheduler = create_scheduler(optimizer, config_args)
    early_stopping_scheduler = EarlyStopping(patience=config_args['early_patience'],
                                             min_delta=config_args['early_delta'])

    if 'resume' in config_args:
        if os.path.isfile(config_args['resume']):
            if torch.distributed.get_rank() == 0:
                print('=> loading checkpoint {}'.format(config_args['resume']))
            checkpoint = torch.load(config_args['resume'])
            start_epoch = checkpoint['epoch']

            checkpoint_state_dict = checkpoint['state_dict']
            if isinstance(checkpoint_state_dict, tuple):
                checkpoint_state_dict = checkpoint_state_dict[0]

            filtered = {k: v for k, v in checkpoint_state_dict.items(
            ) if 'num_batches_tracked' not in k}

            # filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}
            if list(filtered.keys())[0].startswith('module'):
                new_state_dict = OrderedDict()
                for k, v in filtered.items():
                    # remove `module.`，表面从第7个key值字符取到最后一个字符，去掉module.
                    new_state_dict[k[7:]] = v  # 新字典的key值对应的value为一一对应的值。

                model.load_state_dict(new_state_dict)
            else:
                model_dict = model.state_dict()
                model_dict.update(filtered)
                model.load_state_dict(model_dict)
            # model.dropout.p = args.dropout_p
        else:
            if torch.distributed.get_rank() == 0:
                print('=> no checkpoint found at {}'.format(
                    config_args['resume']))

    # Save model config txt
    if torch.distributed.get_rank() == 0:
        with open(os.path.join(check_path,
                               'model.%s.conf' % time.strftime("%Y.%m.%d", time.localtime())),
                  'w') as f:
            f.write('model: ' + str(model) + '\n')
            f.write('Optimizer: ' + str(optimizer) + '\n')

    start = 1 + start_epoch
    if torch.distributed.get_rank() == 0:
        print('Start epoch is : ' + str(start))
    # start = 0
    end = start + config_args['epochs']

    # if config_args['cuda']:
    if len(config_args['gpu_id']) > 1:
        print("Continue with gpu: %s ..." % str(args.local_rank))
        model = DistributedDataParallel(
            model.cuda(), device_ids=[args.local_rank])
    else:
        model = model.cuda()

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
            # extract_sampler.set_epoch(epoch)

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
            train(train_loader, model, optimizer, epoch, scheduler, config_args, writer)
            if config_args['batch_shuffle']:
                train_dir.__shuffle__()

            valid_loss = valid_class(valid_loader, model, epoch, config_args, writer)

            if config_args['early_stopping'] or (
                    epoch % config_args['test_interval'] == 1 or epoch in config_args['milestones'] or epoch == (end - 1)):
                valid_test_dict = valid_test(
                    train_extract_loader, model, epoch, xvector_dir, config_args, writer)
            else:
                valid_test_dict = {}

            # valid_test_dict = valid_test(train_extract_loader, model, epoch, xvector_dir)
            flag_tensor = torch.zeros(1).cuda()
            valid_test_dict['Valid_Loss'] = valid_loss
            if torch.distributed.get_rank() == 0:
                valid_test_result.append(valid_test_dict)

            if torch.distributed.get_rank() == 0 and config_args['early_stopping']:
                early_stopping_scheduler(
                    valid_test_dict[config_args['early_meta']], epoch)

                if early_stopping_scheduler.best_epoch + early_stopping_scheduler.patience >= end:
                    early_stopping_scheduler.early_stop = True

                if config_args['scheduler'] != 'cyclic' and this_lr[0] <= 0.1 ** 3 * config_args['lr']:
                    if len(all_lr) > 5 and all_lr[-5] == this_lr[0]:
                        early_stopping_scheduler.early_stop = True

            # if torch.distributed.get_rank() == 0:
            #     flag_tensor += 1

            if torch.distributed.get_rank() == 0 and (
                    epoch % config_args['test_interval'] == 1 or epoch in config_args['milestones'] or epoch == (
                    end - 1) or early_stopping_scheduler.best_epoch == epoch):

                # if (epoch == 1 or epoch != (end - 2)) and (
                #     epoch % config_args['test_interval'] == 1 or epoch in milestones or epoch == (end - 1)):
                model.eval()
                this_check_path = '{}/checkpoint_{}.pth'.format(
                    check_path, epoch)
                model_state_dict = model.module.state_dict() \
                    if isinstance(model, DistributedDataParallel) else model.state_dict()
                torch.save({'epoch': epoch, 'state_dict': model_state_dict,
                            }, this_check_path)

                # valid_test(train_extract_loader, model, epoch, xvector_dir)
                # test(extract_loader, model, epoch, xvector_dir)

                # if config_args['early_stopping']:
                #     pass
                # elif early_stopping_scheduler.best_epoch == epoch or (
                #         args.early_stopping == False and epoch % args.test_interval == 1):
                # elif epoch % config_args['test_interval'] == 1:
                #     test(extract_loader, model, epoch, xvector_dir)

                if early_stopping_scheduler.early_stop:
                    print('Best Epoch is %d:' %
                          (early_stopping_scheduler.best_epoch))
                    best_epoch = early_stopping_scheduler.best_epoch
                    best_res = valid_test_result[int(best_epoch - 1)]

                    best_str = 'EER(%):       ' + \
                        '{:>6.2f} '.format(best_res['EER'])
                    best_str += '   Threshold: ' + \
                        '{:>7.4f} '.format(best_res['Threshold'])
                    best_str += ' MinDcf-0.01: ' + \
                        '{:.4f} '.format(best_res['MinDCF_01'])
                    best_str += ' MinDcf-0.001: ' + \
                        '{:.4f} '.format(best_res['MinDCF_001'])
                    best_str += ' Mix2,3: ' + \
                        '{:.4f}, {:.4f}\n'.format(
                            best_res['mix2'], best_res['mix3'])
                    print(best_str)
                    with open(os.path.join(check_path, 'result.%s.txt' % time.strftime("%Y.%m.%d", time.localtime())), 'a+') as f:
                        f.write(best_str + '\n')

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
        print("Running %.4f minutes for each epoch.\n" %
              (t / 60 / (max(end - start, 1))))
    # pdb.set_trace()
    # torch.distributed.destroy_process_group()
    # torch.distributed.des
    # exit(0)
    os.kill(os.getpid(), signal.SIGKILL)


if __name__ == '__main__':
    main()
