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
from Light.dataset import Sampler_Loaders, SubScriptDatasets
from Light.model import SpeakerLoss
from TrainAndTest.train_egs.train_dist import all_seed, valid_class, valid_test
from TrainAndTest.train_egs.train_egs import select_samples
import torch._utils

import argparse
import signal
import yaml
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
from TrainAndTest.common_func import create_classifier, create_optimizer, create_scheduler
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


def train_mix(train_loader, model, optimizer, epoch, scheduler, config_args, writer):
    # switch to train mode
    model.train()

    correct = 0.
    total_datasize = 0.
    total_loss = 0.
    orth_err = 0
    total_other_loss = 0.

    pbar = tqdm(enumerate(train_loader)) if torch.distributed.get_rank(
    ) == 0 else enumerate(train_loader)

    output_softmax = nn.Softmax(dim=1)
    return_domain = True if 'domain' in config_args and config_args['domain'] == True else False

    for batch_idx, data_cols in pbar:
        # if not return_domain:
        data, label = data_cols
        batch_weight = None

        
        lamda_beta = np.random.beta(config_args['lamda_beta'], config_args['lamda_beta'])
        mixup_percent = 0.5 if 'batmix_ratio' not in config_args else config_args['batmix_ratio']
        half_data = int(len(data) * mixup_percent)

        if 'mix_ratio'in config_args and np.random.uniform(0, 1) <= config_args['mix_ratio']:
            if config_args['mixup_type'] == 'style':
                rand_idx = torch.randperm(half_data)
                label = torch.cat([label, label[half_data:][rand_idx]], dim=0)
                
                perm = torch.arange(half_data-1, -1, -1)  # inverse index crossdomain mixup
                perm_b, perm_a = perm.chunk(2)
                perm_b = perm_b[torch.randperm(perm_b.shape[0])]
                perm_a = perm_a[torch.randperm(perm_a.shape[0])]
                rand_idx = torch.cat([perm_b, perm_a], 0)
                label = torch.cat([label, label[half_data:][rand_idx]], dim=0)

            elif config_args['mixup_type'] != '':
                rand_idx = torch.randperm(half_data)
                label = torch.cat([label, label[half_data:][rand_idx]], dim=0)
        else:
            lamda_beta = 0
            rand_idx = None
            half_data = 0

        if torch.cuda.is_available():
            label = label.cuda()
            data = data.cuda()

        data, label = Variable(data), Variable(label)
        classfier, feats = model(data, proser=rand_idx, lamda_beta=lamda_beta,
                                 mixup_alpha=config_args['mixup_layer'])
        # print('max logit is ', classfier_label.max())

        loss, other_loss = model.module.loss(classfier, feats, label,
                                             batch_weight=batch_weight, epoch=epoch,
                                             half_data=half_data, lamda_beta=lamda_beta)

        predicted_labels = output_softmax(classfier.clone())
        predicted_one_labels = torch.max(predicted_labels, dim=1)[1]

        minibatch_correct = predicted_one_labels[:half_data].eq(
                label[:half_data]).cpu().sum().float() + \
                lamda_beta * predicted_one_labels[half_data:].eq(
                label[half_data:(half_data + half_data)]).cpu().sum().float() + \
                (1 - lamda_beta) * predicted_one_labels[half_data:].eq(
                label[-half_data:]).cpu().sum().float()
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
            epoch_str = 'Train Epoch {}: [ {:5>3.1f}% ]'.format(
                epoch, 100. * batch_idx / len(train_loader))

            if len(config_args['random_chunk']) == 2 and config_args['random_chunk'][0] <= \
                    config_args['random_chunk'][1]:
                batch_length = data.shape[-1] if config_args['feat_format'] == 'wav' and 'trans_fbank' not in config_args else data.shape[-2]
                epoch_str += ' Batch Len: {:>3d}'.format(batch_length)

            epoch_str += ' Lamda: {:>4.2f}'.format(lamda_beta)
            epoch_str += ' Accuracy(%): {:>6.2f}%'.format(100. * minibatch_acc)

            if other_loss != 0:
                epoch_str += ' Other Loss: {:.4f}'.format(other_loss)

            epoch_str += ' Avg Loss: {:.4f}'.format(
                total_loss / (batch_idx + 1))

            pbar.set_description(epoch_str)

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


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch ( Distributed ) Speaker Recognition: Classification')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')

    parser.add_argument('--train-config', default='', type=str,
                        help='node rank for distributed training')
    parser.add_argument('--seed', type=int, default=123456,
                        help='random seed (default: 0)')
    parser.add_argument('--lamda-beta', type=float, default=0.2,
                    help='random seed (default: 0)')

    args = parser.parse_args()
    # Views the training images and displays the distance on anchor-negative and anchor-positive
    # test_display_triplet_distance = False
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
    if isinstance(config_args['mixup_layer'], list):
        mixup_layer_str = ''.join([str(s) for s in config_args['mixup_layer']])
    else:
        mixup_layer_str = str(config_args['mixup_layer'])

    lambda_str = '_lamda' + str(args.lamda_beta)
    mixup_str = '_%s'%(config_args['mixup_type'][:4]) + mixup_layer_str + lambda_str
    if 'batmix_ratio' in config_args and config_args['batmix_ratio'] != 0.5:
        mixup_str += '_batmix{:.2f}'.format(config_args['batmix_ratio'])
        
    config_args['lamda_beta'] = args.lamda_beta

    check_path = config_args['check_path'] + mixup_str + '/' + str(args.seed)
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
    scheduler = create_scheduler(optimizer, config_args, train_dir)
    early_stopping_scheduler = EarlyStopping(patience=config_args['early_patience'],
                                             min_delta=config_args['early_delta'])

    # scheduler = create_scheduler(optimizer, config_args)

    # Save model config txt
    if torch.distributed.get_rank() == 0:
        with open(os.path.join(check_path,
                               'model.%s.conf' % time.strftime("%Y.%m.%d", time.localtime())),
                  'w') as f:
            f.write('model: ' + str(model) + '\n')
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
            # extract_sampler.set_epoch(epoch)

            # pdb.set_trace()
            # if torch.distributed.get_rank() == 0:
            lr_string = '\33[1;34m Ranking {}: Current \'{}\' learning rate: '.format(torch.distributed.get_rank(),
                                                                                      config_args['optimizer'])
            this_lr = []

            for param_group in optimizer.param_groups:
                this_lr.append(param_group['lr'])
                lr_string += '{:.8f} '.format(param_group['lr'])

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

            train_mix(train_loader, model, optimizer,
                  epoch, scheduler, config_args, writer)
            # if config_args['batch_shuffle']:
            #     train_dir.__shuffle__()

            valid_loss = valid_class(
                valid_loader, model, epoch, config_args, writer)

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

                if early_stopping_scheduler.best_epoch + early_stopping_scheduler.patience >= end and this_lr[0] <= 0.1 ** 3 * config_args['lr']:
                    early_stopping_scheduler.early_stop = True

                if config_args['scheduler'] != 'cyclic' and this_lr[0] <= 0.1 ** 3 * config_args['lr']:
                    if len(all_lr) > 5 and all_lr[-5] >= this_lr[0]:
                        early_stopping_scheduler.early_stop = True

            # if torch.distributed.get_rank() == 0:
            #     flag_tensor += 1

            if torch.distributed.get_rank() == 0 and (
                    epoch % config_args['test_interval'] == 0 or epoch in config_args['milestones'] or epoch == (
                    end - 1) or early_stopping_scheduler.best_epoch == epoch):

                # if (epoch == 1 or epoch != (end - 2)) and (
                #     epoch % config_args['test_interval'] == 1 or epoch in milestones or epoch == (end - 1)):
                model.eval()
                this_check_path = '{}/checkpoint_{}.pth'.format(
                    check_path, epoch)
                model_state_dict = model.module.state_dict() \
                    if isinstance(model, DistributedDataParallel) else model.state_dict()
                torch.save({'epoch': epoch, 'state_dict': model_state_dict,
                            'scheduler': scheduler.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            }, this_check_path)

                if config_args['early_stopping']:
                    pass

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
