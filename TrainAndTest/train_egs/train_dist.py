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
from Light.dataset import Sampler_Loaders, SubDatasets, SubScriptDatasets
from Light.model import SpeakerLoss
from TrainAndTest.train_egs.train_egs import select_samples
import torch._utils

import argparse
import signal
# import yaml
import os
# import os.path as osp
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
# import torchvision.transforms as transforms
from hyperpyyaml import load_hyperpyyaml
from kaldi_io import read_vec_flt
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel
# from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.distributed as dist

from Define_Model.Optimizer import EarlyStopping
from Process_Data.Datasets.KaldiDataset import KaldiExtractDataset, \
    ScriptVerifyDataset
import Process_Data.constants as C
from TrainAndTest.common_func import create_classifier, create_optimizer, create_scheduler, create_model, verification_test, verification_extract, \
    args_parse, args_model, save_model_args
from logger import NewLogger
# import pytorch_lightning as pl

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

# Training settings
# args = args_parse('PyTorch Speaker Recognition: Classification')
# Set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '29555'

# args.cuda = not args.no_cuda and torch.cuda.is_available()
# setting seeds
# pl.seed_everything(args.seed)


def all_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True


def train(train_loader, model, optimizer, epoch, scheduler, config_args, writer):
    # switch to train mode
    model.train()

    correct = 0.
    total_datasize = 0.
    total_loss = 0.
    orth_err = 0
    total_other_loss = 0.

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=True) if torch.distributed.get_rank(
    ) == 0 else enumerate(train_loader)

    output_softmax = nn.Softmax(dim=1)
    return_domain = True if 'domain' in config_args and config_args['domain'] == True else False
    # lambda_ = (epoch / config_args['epochs']) ** 2

    # start_time = time.time()
    # pdb.set_trace()
    for batch_idx, data_cols in pbar:

        if not return_domain:
            data, label = data_cols
            batch_weight = None
        else:
            data, label, domain_label = data_cols
            domain_weight = torch.Tensor(C.DOMAIN_WEIGHT).cuda()
            domain_weight = torch.exp(6*(-domain_weight+0.75))
            domain_weight /= domain_weight.min()

            batch_weight = domain_weight[domain_label]
            model.module.loss.xe_criterion.ce.reduction = 'none'

        if torch.cuda.is_available():
            # label = label.cuda(non_blocking=True)
            # data = data.cuda(non_blocking=True)
            label = label.cuda()
            data = data.cuda()

        data, label = Variable(data), Variable(label)
        # print(data.shape)
        # pdb.set_trace()
        classfier, feats = model(data)
        # print('max logit is ', classfier_label.max())

        loss, other_loss = model.module.loss(classfier, feats, label,
                                             batch_weight=batch_weight, epoch=epoch)

        predicted_labels = output_softmax(classfier.clone())
        predicted_one_labels = torch.max(predicted_labels, dim=1)[1]

        minibatch_correct = float(
            (predicted_one_labels.cpu() == label.cpu()).sum().item())
        minibatch_acc = minibatch_correct / len(predicted_one_labels)
        correct += minibatch_correct

        total_datasize += len(predicted_one_labels)
        # print(loss.shape)
        total_loss += float(loss.item())
        total_other_loss += other_loss

        if torch.distributed.get_rank() == 0:
            writer.add_scalar('Train/All_Loss', float(loss.item()),
                              int((epoch - 1) * len(train_loader) + batch_idx + 1))

        if np.isnan(loss.item()):
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

        if config_args['loss_ratio'] != 0:
            if config_args['loss_type'] in ['center', 'mulcenter', 'gaussian', 'coscenter']:
                for param in model.module.loss.xe_criterion.parameters():
                    param.grad.data *= (1. / config_args['loss_ratio'])

        # optimizer.step()
        if config_args['scheduler'] == 'cyclic':
            scheduler.step()

        # if torch.distributed.get_rank() == 0:
        if torch.distributed.get_rank() == 0 and (batch_idx + 1) % config_args['log_interval'] == 0:
            epoch_str = 'Train Epoch {} '.format(epoch)

            if len(config_args['random_chunk']) == 2 and config_args['random_chunk'][0] <= \
                    config_args['random_chunk'][
                        1]:
                batch_length = data.shape[-1] if config_args['feat_format'] == 'wav' and 'trans_fbank' not in config_args else data.shape[-2]
            #     epoch_str += ' Batch Len: {:>3d}'.format(batch_length)
            # epoch_str += ' Accuracy(%): {:>6.2f}%'.format(100. * minibatch_acc)
            # if other_loss != 0:
            #     epoch_str += ' Other Loss: {:.4f}'.format(other_loss)
            # epoch_str += ' Avg Loss: {:.4f}'.format(
            #     total_loss / (batch_idx + 1))

            pbar.set_description(epoch_str)
            pbar.set_postfix(batch_length=batch_length, accuracy='{:>6.2f}%'.format(
                100. * minibatch_acc), average_loss='{:.4f}'.format(total_loss / (batch_idx + 1)))

        # if (batch_idx + 1) == 100:
        #     break

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


def valid_class(valid_loader, model, epoch, config_args, writer):
    # switch to evaluate mode
    model.eval()

    total_loss = 0.
    total_other_loss = 0.
    # ce_criterion, xe_criterion = ce
    softmax = nn.Softmax(dim=1)

    correct = 0.
    total_datasize = 0.
    # lambda_ = (epoch / config_args['epochs']) ** 2

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(valid_loader):

            if torch.cuda.is_available():
                data = data.cuda()
                label = label.cuda()

            # pdb.set_trace()
            classfier, feats = model(data)
            loss, other_loss = model.module.loss(classfier, feats, label)

            total_loss += float(loss.item())
            total_other_loss += other_loss
            # pdb.set_trace()
            predicted_one_labels = softmax(classfier)
            predicted_one_labels = torch.max(predicted_one_labels, dim=1)[1]

            batch_correct = (predicted_one_labels.cuda() == label).sum().item()
            correct += batch_correct
            total_datasize += len(predicted_one_labels)

    total_batch = len(valid_loader)
    all_total_loss = [None for _ in range(torch.distributed.get_world_size())]
    all_correct = [None for _ in range(torch.distributed.get_world_size())]
    all_total_batch = [None for _ in range(torch.distributed.get_world_size())]
    all_total_datasize = [None for _ in range(
        torch.distributed.get_world_size())]
    all_other_loss = [None for _ in range(torch.distributed.get_world_size())]

    torch.distributed.all_gather_object(all_total_loss, total_loss)
    torch.distributed.all_gather_object(all_correct, correct)
    torch.distributed.all_gather_object(all_total_batch, total_batch)
    torch.distributed.all_gather_object(all_total_datasize, total_datasize)
    torch.distributed.all_gather_object(all_other_loss, total_other_loss)

    # torch.distributed.all_reduce()
    total_loss = np.sum(all_total_loss)
    correct = np.sum(all_correct)
    total_batch = np.sum(all_total_batch)
    total_datasize = np.sum(all_total_datasize)
    all_other_loss = np.sum(all_other_loss)

    valid_loss = total_loss / total_batch
    valid_accuracy = 100. * correct / total_datasize

    if torch.distributed.get_rank() == 0:
        writer.add_scalar('Train/Valid_Loss', valid_loss, epoch)
        writer.add_scalar('Train/Valid_Accuracy', valid_accuracy, epoch)
        torch.cuda.empty_cache()

        this_epoch_str = '          \33[91mValid Accuracy: {:.6f}%, Avg loss: {:.6f}'.format(
            valid_accuracy, valid_loss)

        if all_other_loss != 0:
            this_epoch_str += ' {} Loss: {:6f}'.format(
                config_args['loss_type'], all_other_loss / len(valid_loader))

        this_epoch_str += '.\33[0m'
        print(this_epoch_str)

    return valid_loss


def valid_test(train_extract_loader, model, epoch, xvector_dir, config_args, writer):
    # switch to evaluate mode
    model.eval()

    this_xvector_dir = "%s/train/epoch_%s" % (xvector_dir, epoch)
    verification_extract(train_extract_loader, model, this_xvector_dir,
                         epoch, test_input=config_args['test_input'])

    verify_dir = ScriptVerifyDataset(dir=config_args['train_test_dir'], trials_file=config_args['train_trials'],
                                     xvectors_dir=this_xvector_dir,
                                     loader=read_vec_flt)

    kwargs = {'num_workers': config_args['nj'], 'pin_memory': False}
    verify_loader = torch.utils.data.DataLoader(
        verify_dir, batch_size=128, shuffle=False, **kwargs)
    eer, eer_threshold, mindcf_01, mindcf_001 = verification_test(test_loader=verify_loader,
                                                                  dist_type=(
                                                                      'cos' if config_args['cos_sim'] else 'l2'),
                                                                  log_interval=config_args['log_interval'],
                                                                  xvector_dir=this_xvector_dir,
                                                                  epoch=epoch)
    mix3 = 100. * eer * mindcf_01 * mindcf_001
    mix2 = 100. * eer * mindcf_001

    if torch.distributed.get_rank() == 0:
        print('          \33[91mTrain EER: {:.4f}%, Threshold: {:.4f}, '
              'mindcf-0.01: {:.4f}, mindcf-0.001: {:.4f}, mix2,3: {:.4f}, {:.4f}. \33[0m'.format(100. * eer,
                                                                                                 eer_threshold,
                                                                                                 mindcf_01, mindcf_001, mix2, mix3))

        writer.add_scalar('Train/EER', 100. * eer, epoch)
        writer.add_scalar('Train/Threshold', eer_threshold, epoch)
        writer.add_scalar('Train/mindcf-0.01', mindcf_01, epoch)
        writer.add_scalar('Train/mindcf-0.001', mindcf_001, epoch)
        writer.add_scalar('Train/mix3', mix3, epoch)
        writer.add_scalar('Train/mix2', mix2, epoch)

    torch.cuda.empty_cache()

    return {'EER': 100. * eer, 'Threshold': eer_threshold, 'MinDCF_01': mindcf_01,
            'MinDCF_001': mindcf_001, 'mix3': mix3, 'mix2': mix2}


def valid(valid_loader, train_extract_loader,
          model, optimizer, scheduler,
          check_path, xvector_dir, 
          config_args,
          writer, early_stopping_scheduler,
          epoch, step=None,):
    valid_loss = valid_class(valid_loader, model, epoch, config_args, writer)
    valid_test_dict = valid_test(train_extract_loader, model, epoch, xvector_dir, config_args, writer)

    early_stopping_scheduler(valid_test_dict[config_args['early_meta']], epoch)

    if torch.distributed.get_rank() == 0 and (
                    epoch % config_args['test_interval'] == 0 or early_stopping_scheduler.best_epoch == epoch):
        # save model
        model.eval()
        this_check_path = '{}/checkpoint_{}.pth'.format(check_path, epoch)
        model_state_dict = model.module.state_dict() \
            if isinstance(model, DistributedDataParallel) else model.state_dict()
        torch.save({'epoch': epoch, 'state_dict': model_state_dict,
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    }, this_check_path)
        

def main():
    parser = argparse.ArgumentParser(
        description='PyTorch ( Distributed ) Speaker Recognition: Classification')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')

    parser.add_argument('--train-config', default='', type=str,
                        help='node rank for distributed training')
    parser.add_argument('--seed', type=int, default=123456,
                        help='random seed (default: 0)')

    args = parser.parse_args()
    # Views the training images and displays the distance on anchor-negative and anchor-positive
    # print the experiment configuration
    all_seed(args.seed)
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    torch.multiprocessing.set_sharing_strategy('file_system')

    # load train config file args.train_config
    with open(args.train_config, 'r') as f:
        # config_args = yaml.load(f, Loader=yaml.FullLoader)
        config_args = load_hyperpyyaml(f)

    # Create logger & Define visulaize SummaryWriter instance
    check_path = config_args['check_path'] + '/' + str(args.seed)
    if torch.distributed.get_rank() == 0:
        if not os.path.exists(check_path):
            print('Making checkpath...', check_path)
            os.makedirs(check_path)

        shutil.copy(args.train_config, check_path + '/model.%s.yaml' %
                    time.strftime("%Y.%m.%d", time.localtime()))
        writer = SummaryWriter(logdir=check_path, filename_suffix='SV')
        sys.stdout = NewLogger(
            os.path.join(check_path, 'log.%s.txt' % time.strftime("%Y.%m.%d", time.localtime())))
    else:
        writer = None

    # Dataset
    train_dir, valid_dir, train_extract_dir = SubScriptDatasets(config_args)
    train_loader, train_sampler, valid_loader, valid_sampler, train_extract_loader, train_extract_sampler = Sampler_Loaders(
        train_dir, valid_dir, train_extract_dir, config_args)

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('\nCurrent time is \33[91m{}\33[0m.'.format(str(time.asctime())))
        print('Number of Speakers: {}.\n'.format(train_dir.num_spks))
        if train_dir.num_spks != config_args['num_classes']:
            print('Number of Speakers in training set is not equal to the asigned number.\n'.format(
                train_dir.num_spks))

        print('Testing with %s distance, ' %
              ('cos' if config_args['cos_sim'] else 'l2'))

    # model = create_model(config_args['model'], **model_kwargs)
    if 'embedding_model' in config_args:
        model = config_args['embedding_model']

    if 'classifier' in config_args:
        model.classifier = config_args['classifier']
    else:
        create_classifier(model, **config_args)

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
            filtered = {k: v for k, v in checkpoint_state_dict.items(
            ) if 'num_batches_tracked' not in k}
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
            print('=> no checkpoint found at {}'.format(config_args['resume']))

    model.loss = SpeakerLoss(config_args)

    model_para = [{'params': model.parameters()}]
    if config_args['loss_type'] in ['center', 'variance', 'mulcenter', 'gaussian', 'coscenter', 'ring']:
        assert config_args['lr_ratio'] > 0
        model_para.append({'params': model.loss.xe_criterion.parameters(
        ), 'lr': config_args['lr'] * config_args['lr_ratio']})

    if 'second_wd' in config_args and config_args['second_wd'] > 0:
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

    if 'filter_wd' in config_args:
        # if config_args['filter'] in ['fDLR', 'fBLayer', 'fLLayer', 'fBPLayer', 'sinc2down']:
        filter_params = list(map(id, model.input_mask[0].parameters()))
        rest_params = filter(lambda p: id(
            p) not in filter_params, model_para[0]['params'])
        init_wd = config_args['filter_wd'] if 'filter_wd' in config_args else config_args['weight_decay']
        init_lr = config_args['lr'] * \
            config_args['lr_ratio'] if config_args['lr_ratio'] > 0 else config_args['lr']
        print('Set the lr and weight_decay of filter layer to %f and %f' % (init_lr, init_wd))
        
        model_para[0]['params'] = rest_params
        model_para.append({'params': model.input_mask[0].parameters(), 'lr': init_lr,
                            'weight_decay': init_wd})

    opt_kwargs = {'lr': config_args['lr'], 'lr_decay': config_args['lr_decay'],
                  'weight_decay': config_args['weight_decay'],
                  'dampening': config_args['dampening'],
                  'momentum': config_args['momentum'],
                  'nesterov': config_args['nesterov']}

    optimizer = create_optimizer(
        model_para, config_args['optimizer'], **opt_kwargs)
    scheduler = create_scheduler(optimizer, config_args, train_dir)
    early_stopping_scheduler = EarlyStopping(patience=config_args['early_patience'],
                                             min_delta=config_args['early_delta'])

    # scheduler = create_scheduler(optimizer, config_args)
    # Save model config txt
    if torch.distributed.get_rank() == 0:
        with open(os.path.join(check_path,
                               'model.%s.conf' % time.strftime("%Y.%m.%d", time.localtime())),
                  'w') as f:
            f.write('Model:     ' + str(model) + '\n')
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

            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])

            # model.dropout.p = args.dropout_p
        else:
            if torch.distributed.get_rank() == 0:
                print('=> no checkpoint found at {}'.format(
                    config_args['resume']))

    start = 1 + start_epoch
    if torch.distributed.get_rank() == 0:
        print('Start epoch is : ' + str(start))
    end = start + config_args['epochs']

    # if config_args['cuda']:
    if len(config_args['gpu_id']) > 1:
        print("Continue with gpu: %s ..." % str(args.local_rank))
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model.cuda(), device_ids=[args.local_rank])
    else:
        model = model.cuda()

    try:
        print('Dropout is {}.'.format(model.dropout_p))
    except:
        pass

    xvector_dir = check_path.replace('checkpoint', 'xvector')
    start_time = time.time()

    all_lr = []
    valid_test_result = []

    try:
        for epoch in range(start, end):

            # if torch. is_distributed():
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)
            train_extract_sampler.set_epoch(epoch)
            # if torch.distributed.get_rank() == 0:
            lr_string = '\33[1;34m Ranking {}: \'{}\' learning rate: '.format(torch.distributed.get_rank(),
                                                                                      config_args['optimizer'])
            this_lr = [ param_group['lr'] for param_group in optimizer.param_groups]
            lr_string += " ".join(['{:.8f} '.format(i) for i in this_lr])
            print('%s \33[0m' % lr_string)

            all_lr.append(this_lr[0])
            if torch.distributed.get_rank() == 0:
                writer.add_scalar('Train/lr', this_lr[0], epoch)

            torch.distributed.barrier()
            if not torch.distributed.is_initialized():
                break

            if 'coreset_percent' in config_args and config_args['coreset_percent'] > 0 and epoch % config_args['select_interval'] == 1:
                select_samples(train_loader, model, config_args,
                               config_args['select_score'])

            train(train_loader, model, optimizer,
                  epoch, scheduler, config_args, writer)

            valid_loss = valid_class(
                valid_loader, model, epoch, config_args, writer)
            # if config_args['early_stopping'] or (
            #         epoch % config_args['test_interval'] == 1 or epoch in config_args['milestones'] or epoch == (end - 1)):
            valid_test_dict = valid_test(
                train_extract_loader, model, epoch, xvector_dir, config_args, writer)

            valid_test_dict['Valid_Loss'] = valid_loss

            if torch.distributed.get_rank() == 0:
                valid_test_result.append(valid_test_dict)

            if torch.distributed.get_rank() == 0 and config_args['early_stopping']:
                early_stopping_scheduler(
                    valid_test_dict[config_args['early_meta']], epoch)

                if early_stopping_scheduler.best_epoch + early_stopping_scheduler.patience >= end and this_lr[0] <= 0.1 ** 3 * config_args['lr']:
                    early_stopping_scheduler.early_stop = True

                if config_args['scheduler'] != 'cyclic' and this_lr[0] <= 0.1 ** 3 * config_args['lr']:
                    if len(all_lr) > 5 and all_lr[-5] >= this_lr[0]:
                        early_stopping_scheduler.early_stop = True

            if torch.distributed.get_rank() == 0 and (
                    epoch % config_args['test_interval'] == 0 or epoch in config_args['milestones'] or epoch == (
                    end - 1) or early_stopping_scheduler.best_epoch == epoch):

                model.eval()
                this_check_path = '{}/checkpoint_{}.pth'.format(
                    check_path, epoch)
                model_state_dict = model.module.state_dict() \
                    if isinstance(model, DistributedDataParallel) else model.state_dict()
                torch.save({'epoch': epoch, 'state_dict': model_state_dict,
                            'scheduler': scheduler.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            }, this_check_path)

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
                        '{:.4f}, {:.4f}'.format(
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
    time.sleep(10)
    os.kill(os.getpid(), signal.SIGKILL)
    exit(0)


if __name__ == '__main__':
    main()
