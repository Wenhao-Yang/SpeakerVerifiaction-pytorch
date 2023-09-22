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
from Light.dataset import Sampler_Loaders, ScriptGenderDatasets, SubScriptDatasets
from Light.model import SpeakerLoss
from TrainAndTest.train_egs.train_egs import select_samples
import torch._utils

import argparse
# import signal
# import yaml
import os
# import os.path as osp
# import pdb
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
from Process_Data.Datasets.KaldiDataset import ScriptVerifyDataset
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


def train(train_loader, model, optimizer, epoch, scheduler, config_args, writer,
          trans):
    # switch to train mode
    model.train()

    total_loss = 0.
    orth_err = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=True) if torch.distributed.get_rank(
    ) == 0 else enumerate(train_loader)

    mse = config_args['loss']

    trans = trans.cuda()

    if 'augment_pipeline' in config_args:
        augment_pipeline = []
        for _, augment in enumerate(config_args['augment_pipeline']):
            augment_pipeline.append(augment.cuda())

    # pdb.set_trace()
    for batch_idx, data_cols in pbar:

        data, label = data_cols

        # print(data.shape)
        # pdb.set_trace()
        if 'augment_pipeline' in config_args:
            with torch.no_grad():
                feat_aug_tot = []
                # wavs_aug_tot.append(data.cuda()) # data_shape [batch, 1,1,time]
                wavs = data.squeeze().cuda()

                for augment in augment_pipeline:
                    # augment = np.random.choice(augment_pipeline)
                    # for count, augment in enumerate(augment_pipeline):
                    # Apply augment
                    wavs_aug = augment(wavs, torch.tensor([1]*len(data)).cuda())
                    # Managing speed change
                    if wavs_aug.shape[1] > wavs.shape[1]:
                        wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
                    else:
                        zero_sig = torch.zeros_like(wavs)
                        zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                        wavs_aug = zero_sig

                    feat_aug = trans(torch.tensor(wavs_aug).unsqueeze(1).float())
                    # if 'concat_augment' in config_args and config_args['concat_augment']:
                    feat_aug_tot.append(feat_aug)
                    # else:
                    #     wavs = feat_aug
                    #     wavs_aug_tot[0] = wavs.unsqueeze(1).unsqueeze(1)
                
                n_augment = len(feat_aug_tot)
                input_feat = torch.cat(feat_aug_tot, dim=0).unsqueeze(1)
                # label = torch.cat([label] * n_augment)
                feat = trans(torch.tensor(wavs).unsqueeze(1).float())
                real_feat = torch.cat([feat] * n_augment).unsqueeze(1)

        if torch.cuda.is_available():
            input_feat = input_feat.detach().cuda()
            real_feat = real_feat.detach().cuda()
            # print(input_feat.shape, real_feat.shape)

        denoise_feat = model(input_feat)
        loss = mse(denoise_feat, real_feat)

        total_loss += float(loss.item())

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

        # optimizer.step()
        if config_args['scheduler'] == 'cyclic':
            scheduler.step()

        # if torch.distributed.get_rank() == 0:
        if torch.distributed.get_rank() == 0 and (batch_idx + 1) % config_args['log_interval'] == 0:
            epoch_str = 'Train Epoch {} '.format(epoch)

            batch_length = input_feat.shape[-2]

            pbar.set_description(epoch_str)
            pbar.set_postfix(batch_length=batch_length, average_loss='{:.4f}'.format(total_loss / (batch_idx + 1)))

        # if (batch_idx + 1) == 10:
        #     break

    this_epoch_str = 'Epoch {:>2d}: \33[91mTrain Loss: {:6f}'.format(epoch, total_loss / len(train_loader))
    this_epoch_str += '.\33[0m'

    if torch.distributed.get_rank() == 0:
        print(this_epoch_str)
        writer.add_scalar('Train/Loss', total_loss / len(train_loader), epoch)

    torch.cuda.empty_cache()


def valid_class(valid_loader, model, epoch, config_args, writer, trans):
    # switch to evaluate mode
    model.eval()

    total_loss = 0.
    total_other_loss = 0.
    # ce_criterion, xe_criterion = ce
    mse = config_args['loss']
    trans = trans.cuda()
    # correct = 0.
    # total_datasize = 0.
    # lambda_ = (epoch / config_args['epochs']) ** 2
    # if 'augment_pipeline' in config_args:
    augment_pipeline = []
    for _, augment in enumerate(config_args['augment_pipeline']):
        augment_pipeline.append(augment.cuda())

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(valid_loader):

            if torch.cuda.is_available():
                data = data.cuda()

            real_feat = trans(data.squeeze())

            augment = np.random.choice(augment_pipeline)
            aug_data = augment(data, torch.tensor([1]*len(data)).cuda())

            aug_feat = trans(aug_data.unsqueeze(1).float())
            # pdb.set_trace()
            denoise_feat = model(aug_feat)
            loss = mse(denoise_feat, real_feat)
            total_loss += float(loss.item())
            
    total_batch = len(valid_loader)
    all_total_loss = [None for _ in range(torch.distributed.get_world_size())]
    all_total_batch = [None for _ in range(torch.distributed.get_world_size())]

    torch.distributed.all_gather_object(all_total_loss, total_loss)
    torch.distributed.all_gather_object(all_total_batch, total_batch)

    # torch.distributed.all_reduce()
    total_loss = np.sum(all_total_loss)
    total_batch = np.sum(all_total_batch)
    all_other_loss = np.sum(all_other_loss)

    valid_loss = total_loss / total_batch

    if torch.distributed.get_rank() == 0:
        writer.add_scalar('Train/Valid_Loss', valid_loss, epoch)
        torch.cuda.empty_cache()

        this_epoch_str = '          \33[91mValid Loss: {:.6f}'.format(valid_loss)

        if all_other_loss != 0:
            this_epoch_str += ' {} Loss: {:6f}'.format(
                config_args['loss_type'], all_other_loss / len(valid_loader))

        this_epoch_str += '.\33[0m'
        print(this_epoch_str)

    return valid_loss



def main():
    parser = argparse.ArgumentParser(
        description='PyTorch ( Distributed ) Speaker Recognition: Classification')
    # parser.add_argument('--local_rank', default=-1, type=int,
    #                     help='node rank for distributed training')
    parser.add_argument('--train-config', default='', type=str,
                        help='node rank for distributed training')
    parser.add_argument('--seed', type=int, default=123456,
                        help='random seed (default: 0)')

    args = parser.parse_args()
    # print the experiment configuration
    all_seed(args.seed)
    torch.distributed.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # torch.multiprocessing.set_sharing_strategy('file_system')

    # load train config file args.train_config
    with open(args.train_config, 'r') as f:
        config_args = load_hyperpyyaml(f)

    # Create logger & Define visulaize SummaryWriter instance
    check_path = config_args['check_path'] + '/' + str(args.seed)
    if torch.distributed.get_rank() == 0:
        if not os.path.exists(check_path):
            print('Making checkpath...', check_path)
            os.makedirs(check_path)
        else:
            print('Checkpath exists...', check_path)

        shutil.copy(args.train_config, check_path + '/model.%s.yaml' %
                    time.strftime("%Y.%m.%d", time.localtime()))
        writer = SummaryWriter(logdir=check_path, filename_suffix='SV')
        sys.stdout = NewLogger(
            os.path.join(check_path, 'log.%s.txt' % time.strftime("%Y.%m.%d", time.localtime())))
    else:
        writer = None
    # Dataset
    # train_dir, valid_dir, train_extract_dir = SubScriptDatasets(config_args)
    # train_loader, train_sampler, valid_loader, valid_sampler, train_extract_loader, train_extract_sampler = Sampler_Loaders(
    #     train_dir, valid_dir, train_extract_dir, config_args)
    train_loader, eval_loader, train_sampler, train_dataset = ScriptGenderDatasets(config_args)

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('\nCurrent time is \33[91m{}\33[0m.'.format(str(time.asctime())))
        # print('Number of Speakers: {}.\n'.format(train_dir.num_spks))
        # if train_dir.num_spks != config_args['num_classes']:
        #     print('Number of Speakers in training set is not equal to the asigned number.\n'.format(
        #         train_dir.num_spks))

        # print('Testing with %s distance, ' %
        #       ('cos' if config_args['cos_sim'] else 'l2'))

    # model = create_model(config_args['model'], **model_kwargs)
    # if 'embedding_model' in config_args:
    model = config_args['model']
    
    trans = nn.Sequential(
        OrderedDict([
          ('fbank', config_args['transforms'][0]),
          ('mean', config_args['transforms'][1]),
        ]))
    
    # loss_func = config_args['loss']

    start_epoch = 0
    check_path = config_args['check_path'] + '/' + str(args.seed)
    if 'finetune' not in config_args or not config_args['finetune']:
        this_check_path = '{}/checkpoint_{}_{}.pth'.format(check_path, start_epoch,
                                                           time.strftime('%Y_%b_%d_%H:%M', time.localtime()))
        if not os.path.exists(this_check_path):
            torch.save({'state_dict': model.state_dict()}, this_check_path)

    # Load checkpoint
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

    # model.loss = SpeakerLoss(config_args)

    model_para = [{'params': model.parameters()}]
    # if config_args['loss_type'] in ['center', 'variance', 'mulcenter', 'gaussian', 'coscenter', 'ring']:
    #     assert config_args['lr_ratio'] > 0
    #     model_para.append({'params': model.loss.xe_criterion.parameters(
    #     ), 'lr': config_args['lr'] * config_args['lr_ratio']})

    # if 'second_wd' in config_args and config_args['second_wd'] > 0:
    #     # if config_args['loss_type in ['asoft', 'amsoft']:
    #     classifier_params = list(map(id, model.classifier.parameters()))
    #     rest_params = filter(lambda p: id(
    #         p) not in classifier_params, model.parameters())

    #     init_lr = config_args['lr'] * \
    #         config_args['lr_ratio'] if config_args['lr_ratio'] > 0 else config_args['lr']
    #     init_wd = config_args['second_wd'] if config_args['second_wd'] > 0 else config_args['weight_decay']
    #     print('Set the lr and weight_decay of classifier to %f and %f' %
    #           (init_lr, init_wd))
    #     model_para = [{'params': rest_params},
    #                   {'params': model.classifier.parameters(), 'lr': init_lr, 'weight_decay': init_wd}]

    # if 'filter_wd' in config_args:
    #     # if config_args['filter'] in ['fDLR', 'fBLayer', 'fLLayer', 'fBPLayer', 'sinc2down']:
    #     filter_params = list(map(id, model.input_mask[0].parameters()))
    #     rest_params = filter(lambda p: id(
    #         p) not in filter_params, model_para[0]['params'])
    #     init_wd = config_args['filter_wd'] if 'filter_wd' in config_args else config_args['weight_decay']
    #     init_lr = config_args['lr'] * \
    #         config_args['lr_ratio'] if config_args['lr_ratio'] > 0 else config_args['lr']
    #     print('Set the lr and weight_decay of filter layer to %f and %f' % (init_lr, init_wd))

    #     model_para[0]['params'] = rest_params
    #     model_para.append({'params': model.input_mask[0].parameters(), 'lr': init_lr,
    #                         'weight_decay': init_wd})

    opt_kwargs = {'lr': config_args['lr'], 'lr_decay': config_args['lr_decay'],
                  'weight_decay': config_args['weight_decay'],
                  'dampening': config_args['dampening'],
                  'momentum': config_args['momentum'],
                  'nesterov': config_args['nesterov']}

    optimizer = create_optimizer(
        model_para, config_args['optimizer'], **opt_kwargs)
    
    scheduler = create_scheduler(optimizer, config_args, train_dataset)
    early_stopping_scheduler = EarlyStopping(patience=config_args['early_patience'],
                                             min_delta=config_args['early_delta'])

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
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

            # model.dropout.p = args.dropout_p
        else:
            if torch.distributed.get_rank() == 0:
                print('=> no checkpoint found at {}'.format(
                    config_args['resume']))

    start = 1 + start_epoch
    if torch.distributed.get_rank() == 0:
        print('Start epoch is : ' + str(start))
    end = start + config_args['epochs']

    if len(config_args['gpu_id']) > 1:
        print("Continue with gpu: %s ..." % str(local_rank))
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model.cuda(), device_ids=[local_rank])#, find_unused_parameters=True)
    else:
        model = model.cuda()

    try:
        print('Dropout is {}.'.format(model.dropout_p))
    except:
        pass

    # xvector_dir = check_path.replace('checkpoint', 'xvector')
    start_time = time.time()

    all_lr = []
    valid_test_result = []

    try:
        for epoch in range(start, end):
            train_sampler.set_epoch(epoch)

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
            # if 'coreset_percent' in config_args and config_args['coreset_percent'] > 0 and epoch % config_args['select_interval'] == 1:
            #     select_samples(train_loader, model, config_args,
            #                    config_args['select_score'])

            train(train_loader, model, optimizer,
                  epoch, scheduler, config_args, writer, trans)

            valid_loss = valid_class(
                eval_loader, model, epoch, config_args, writer, trans)
            # valid_test_dict = valid_test(
            #     train_extract_loader, model, epoch, xvector_dir, config_args, writer)
            # valid_test_dict['Valid_Loss'] = valid_loss

            if torch.distributed.get_rank() == 0 and config_args['early_stopping']:
                valid_test_result.append(valid_loss)

                early_stopping_scheduler(valid_loss, epoch)

                if early_stopping_scheduler.best_epoch + early_stopping_scheduler.patience >= end and this_lr[0] <= 0.1 ** 3 * config_args['lr']:
                    early_stopping_scheduler.early_stop = True

                if config_args['scheduler'] != 'cyclic' and this_lr[0] <= 0.1 ** 3 * config_args['lr']:
                    if len(all_lr) > 5 and all_lr[-5] >= this_lr[0]:
                        early_stopping_scheduler.early_stop = True

            if torch.distributed.get_rank() == 0:
                top_k = early_stopping_scheduler.top_k()
            else:
                top_k = []
            if torch.distributed.get_rank() == 0 and (
                    epoch % config_args['test_interval'] == 0 or epoch in config_args['milestones'] or epoch == (
                    end - 1) or early_stopping_scheduler.best_epoch == epoch or epoch in top_k):

                model.eval()
                this_check_path = '{}/checkpoint_{}.pth'.format(
                    check_path, epoch)
                model_state_dict = model.module.state_dict() \
                    if isinstance(model, DistributedDataParallel) else model.state_dict()
                torch.save({'epoch': epoch, 'state_dict': model_state_dict,
                            'scheduler': scheduler.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            }, this_check_path)

            check_stop = torch.tensor(
                int(early_stopping_scheduler.early_stop)).cuda()
            dist.all_reduce(check_stop, op=dist.ReduceOp.SUM)

            if check_stop:
                end = epoch
                if torch.distributed.get_rank() == 0:
                    print('Best Epochs : ', top_k)
                    best_epoch = early_stopping_scheduler.best_epoch
                    best_res = valid_test_result[int(best_epoch - 1)]

                    best_str = 'Loss(%):  ' + '{:>6.2f} '.format(best_res)
                    print(best_str)

                    with open(os.path.join(check_path, 'result.%s.txt' % time.strftime("%Y.%m.%d", time.localtime())), 'a+') as f:
                        f.write(best_str + '\n')

                    try:
                        shutil.copy('{}/checkpoint_{}.pth'.format(check_path,
                                                                  early_stopping_scheduler.best_epoch),
                                    '{}/best.pth'.format(check_path))
                    except Exception as e:
                        print(e)
                break

            if config_args['scheduler'] == 'rop':
                scheduler.step(valid_loss)
            elif config_args['scheduler'] == 'cyclic':
                pass
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

    time.sleep(5)
    exit(0)


if __name__ == '__main__':
    main()
