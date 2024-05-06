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
import hashlib
from Light.dataset import Sampler_Loaders, SubScriptDatasets
from Light.model import SpeakerLoss
from Process_Data.audio_processing import AdaptiveBandPass, BandPass
from TrainAndTest.common_func import on_main, resume_checkpoint
from TrainAndTest.train_egs.train_dist import save_checkpoint, test_results
from speechbrain.lobes.augment import TimeDomainSpecAugment

from TrainAndTest.train_egs.train_dist import all_seed, check_earlystop_break, valid_class, valid_test
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
from Define_Model.model import create_classifier, create_model, get_layer_param, get_trainable_param
from Define_Model.Optimizer import create_optimizer, create_scheduler
from logger import NewLogger
# import pytorch_lightning as pl
import Process_Data.constants as C

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

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=True, ncols=150) if torch.distributed.get_rank(
    ) == 0 else enumerate(train_loader)

    output_softmax = nn.Softmax(dim=1)
    return_domain = True if 'domain' in config_args and config_args['domain'] == True else False
    mix_epoch = config_args['mix_epoch'] if 'mix_epoch' in config_args else config_args['epochs']
    return_idx = False if 'return_idx' not in config_args else config_args['return_idx']

    if 'augment_pipeline' in config_args:
        num_pipes = config_args['num_pipes'] if 'num_pipes' in config_args else 1
        augment_pipeline = []
        for _, augment in enumerate(config_args['augment_pipeline']):
            if isinstance(augment, AdaptiveBandPass) or isinstance(augment, BandPass):
                augment_pipeline.append(augment)
            else:
                augment_pipeline.append(augment.cuda())
        
        # augment pipeline reverse
        if 'augment_prob' in config_args and isinstance(config_args['augment_prob'], list):
            p = np.array(config_args['augment_prob'])
            rp = 1/p
            rp /= rp.sum()
            min_score, max_score = 0.1, 20
            max_diff = max_score - min_score

            augp_a = (rp - p) / max_diff
            augp_b = p - augp_a * min_score
                
    for batch_idx, data_cols in pbar:
        if 'sample_score' in config_args and 'sample_ratio' in config_args:
            data, label, scores = data_cols
            batch_weight = None
        elif return_idx:
            data, label, batch_idxs = data_cols
            batch_weight = None
        elif not return_domain:
            data, label = data_cols
            if 'train_second_dir' in config_args:
                repeats = data.shape[-2]
                
                label = label.view((-1,1)).repeat(1, repeats).reshape(-1)
                # print(data.shape, label.shape)
                if data.shape[-2] > 1:
                    data = data.reshape(-1, 1, 1, data.shape[-1])
            
            batch_weight = None
        else:
            data, label, domain_label = data_cols
            domain_weight = torch.Tensor(C.DOMAIN_WEIGHT).cuda()
            domain_weight = torch.exp(6*(-domain_weight+0.75))
            domain_weight /= domain_weight.min()

            batch_weight = domain_weight[domain_label]
            model.module.loss.xe_criterion.ce.reduction = 'none'

        if 'augment_pipeline' in config_args:
            with torch.no_grad():
                wavs_aug_tot = []
                labels_aug_tot = []

                if 'second_as_augment' in config_args and config_args['second_as_augment'] == True:
                    data = data.reshape(-1, 1, 2, data.shape[-1])
                    second_data  = data[:,:,1,:].unsqueeze(1)
                    data = data[:,:,0,:].unsqueeze(1)
                    wavs_aug_tot.append(second_data.cuda())

                    label = label.reshape(-1, repeats)[:, 0]
                    labels_aug_tot.append(label.cuda())

                wavs_aug_tot.append(data.cuda()) # data_shape [batch, 1,1,time]
                labels_aug_tot.append(label.cuda())

                wavs = data.squeeze().cuda()
                wav_label = label.squeeze().cuda()

                if 'sample_score' in config_args and 'sample_ratio' in config_args:
                    if config_args['sample_ratio'] < 1:
                        sample_ratio = int(config_args['sample_ratio'] * len(wavs))
                        
                        if 'batch_sample' not in config_args or config_args['batch_sample'] == 'norm':
                            scores = scores/scores.sum()
                            score_idx = np.random.choice(len(wavs), sample_ratio,
                                                            p=scores.squeeze().numpy(), replace=False)
                        elif config_args['batch_sample'] == 'norm_mean':
                            if 'score_mean_ratio' in config_args:
                                scores -= scores.mean() * config_args['score_mean_ratio']
                            else:
                                scores -= scores.mean()
                                
                            scores = scores.abs()
                            scores = scores/scores.sum()
                            score_idx = np.random.choice(len(wavs), sample_ratio,
                                                            p=scores.squeeze().numpy(), replace=False)
                        elif config_args['batch_sample'] == 'max':
                            score_idx = np.argsort(scores)[0][-sample_ratio:]
                        elif config_args['batch_sample'] == 'min':
                            if len(scores.shape) > 1:
                                scores = scores.mean(dim=1)
                            score_idx = np.argsort(scores)[:sample_ratio]
                        elif config_args['batch_sample'] == 'soft':
                            scores = scores.squeeze().unsqueeze(0)
                            scores = output_softmax(scores/scores.mean()) #overflow
                            score_idx = np.random.choice(len(wavs), sample_ratio,
                                                        p=scores.squeeze().numpy(), replace=False)
                        elif config_args['batch_sample'] == 'rand':
                            score_idx = np.random.choice(len(wavs), sample_ratio, replace=False)
                        if 'repeat_batch' in config_args:
                            # repeat sampled samples to make batch_size equal 
                            while len(score_idx) < len(wavs):
                                score_idx = np.concatenate((score_idx, score_idx))
                            score_idx = score_idx[:len(wavs)]

                        scores = scores[score_idx]
                        wavs = wavs[score_idx]
                        wav_label = wav_label[score_idx]

                if 'augment_prob' not in config_args and num_pipes <= len(augment_pipeline):
                    if 'sample_score' in config_args and config_args['sample_ratio'] == 1:
                        # print(config_args['sample_score'])
                        augs_idx = []
                        for s in scores:
                            augs_idx.append(np.argsort(s.numpy())[:num_pipes])

                        augs_idx = np.array(augs_idx)
                        # print(augs_idx)
                        sample_idxs = [np.where(np.sum(augs_idx == i, axis=1) >= 1)[0] for i in range(len(augment_pipeline))]
                        augs_idx = [i for i in range(len(augment_pipeline))]
                    else:
                        if num_pipes <= len(augment_pipeline):
                            augs_idx = np.random.choice(len(augment_pipeline), size=num_pipes, replace=False)
                        else:
                            augs_idx = np.random.choice(len(augment_pipeline), size=num_pipes-len(augment_pipeline), replace=False)
                            augs_idx = np.concatenate([augs_idx, np.arange(len(augment_pipeline))])

                        sample_idxs = [np.arange(len(wavs))] * len(augs_idx)
                elif 'augment_prob' in config_args:
                    if isinstance(config_args['augment_prob'], list) and 'sample_score' in config_args:
                        augs_idx = []
                        for s in scores:
                            this_p = augp_a * float(s) + augp_b
                            this_p /= this_p.sum()
                            augs_idx.append(np.random.choice(len(augment_pipeline), size=num_pipes, 
                                                        p=this_p, replace=False))

                        augs_idx = np.array(augs_idx)
                        sample_idxs = [np.where(np.sum(augs_idx == i, axis=1) >= 1)[0] for i in range(len(augment_pipeline))]
                        augs_idx = [i for i in range(len(augment_pipeline))]
                    elif isinstance(config_args['augment_prob'], list):
                        augs_idx = []
                        this_p = np.array(config_args['augment_prob'])
                        this_p /= this_p.sum()

                        for i in range(len(wavs)):
                            augs_idx.append(np.random.choice(len(augment_pipeline), size=num_pipes, 
                                                        p=this_p, replace=False))

                        augs_idx = np.array(augs_idx)
                        sample_idxs = [np.where(np.sum(augs_idx == i, axis=1) >= 1)[0] for i in range(len(augment_pipeline))]
                        augs_idx = [i for i in range(len(augment_pipeline))]
                    else: 
                        this_lr = optimizer.param_groups[0]['lr']
                        augs_idx = config_args['augment_prob'](ratio=this_lr)
                        sample_idxs = [np.arange(len(data))] * len(augs_idx)                    

                augs = [augment_pipeline[i] for i in augs_idx]
            
                for data_idx, augment in zip(sample_idxs, augs):
                    # Apply augment
                    wavs_aug = augment(wavs[data_idx], torch.tensor([1.0]*len(wavs[[data_idx]])).cuda())
                    # Managing speed change
                    if wavs_aug.shape[1] > wavs[data_idx].shape[1]:
                        wavs_aug = wavs_aug[:, 0 : wavs[data_idx].shape[1]]
                    else:
                        zero_sig = torch.zeros_like(wavs[data_idx])
                        zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                        wavs_aug = zero_sig

                    label_offset = 0
                    if isinstance(augment, TimeDomainSpecAugment):
                        if len(augment.speed_perturb.speeds) == 1:
                            if augment.speed_perturb.speeds[0] < 95:
                                label_offset = int(1 * config_args['num_real_classes'])
                            elif augment.speed_perturb.speeds[0] > 105:
                                label_offset = int(2 * config_args['num_real_classes'])

                    if 'concat_augment' in config_args and config_args['concat_augment']:
                        wavs_aug_tot.append(wavs_aug.unsqueeze(1).unsqueeze(1))
                        labels_aug_tot.append(wav_label[data_idx] + label_offset)
                    else:
                        wavs = wavs_aug
                        wavs_aug_tot[0] = wavs_aug.unsqueeze(1).unsqueeze(1)
                        labels_aug_tot[0] = wav_label[data_idx] + label_offset

                #Todo: shuffle the original data may be needed here.
                data = torch.cat(wavs_aug_tot, dim=0)
                label = torch.cat(labels_aug_tot)
          
                if 'train_second_dir' in config_args:
                    data_shape = data.shape
                    data = data.reshape(2, -1, 1, wavs.shape[-1]).transpose(0,1)
                    data = data.reshape(data_shape)
                    label = label.reshape(2, -1).transpose(0,1).reshape(-1)
                    
                    if 'domain_mix' in config_args and config_args['domain_mix']:
                        label = torch.cat([label, label.reshape(2, -1)[0]])

        lamda_beta = np.random.beta(config_args['lamda_beta'], config_args['lamda_beta'])
        mixup_percent = 0.5 if 'batmix_ratio' not in config_args else config_args['batmix_ratio']
        half_data = int(len(data) * mixup_percent)

        if 'mix_ratio'in config_args and np.random.uniform(0, 1) <= config_args['mix_ratio'] and epoch <= mix_epoch:
            if config_args['mixup_type'].startswith('style'):
                perm = torch.arange(half_data-1, -1, -1)  # inverse index crossdomain mixup
                perm_b, perm_a = perm.chunk(2)
                perm_b = perm_b[torch.randperm(perm_b.shape[0])]
                perm_a = perm_a[torch.randperm(perm_a.shape[0])]
                rand_idx = torch.cat([perm_b, perm_a], 0)
                label = torch.cat([label, label[-half_data:][rand_idx]], dim=0)
            elif config_args['mixup_type'] != '':
                rand_idx = torch.randperm(half_data)
                label = torch.cat([label, label[-half_data:][rand_idx]], dim=0)
        else:
            lamda_beta = 0
            rand_idx = None
            half_data = 0

        if torch.cuda.is_available():
            label = label.cuda()
            data = data.cuda()

        data, label = Variable(data), Variable(label)
        classfier, feats = model(data, proser=rand_idx,
                                 lamda_beta=lamda_beta,
                                 mixup_alpha=config_args['mixup_layer'])

        loss = model.module.loss((classfier, feats), label,
                                             batch_weight=batch_weight, epoch=epoch,
                                             half_data=half_data, lamda_beta=lamda_beta)
        other_loss = 0

        predicted_labels = output_softmax(classfier.clone())
        predicted_one_labels = torch.max(predicted_labels, dim=1)[1]

        if half_data == len(data):
            minibatch_correct = lamda_beta * predicted_one_labels.eq(
                label[:half_data]).cpu().sum().float() + \
                (1 - lamda_beta) * predicted_one_labels.eq(
                label[-half_data:]).cpu().sum().float()
        elif half_data != 0:
            minibatch_correct = predicted_one_labels[:-half_data].eq(
                label[:int(-2*half_data)]).cpu().sum().float() + \
                lamda_beta * predicted_one_labels[-half_data:].eq(
                label[int(-2*half_data):-half_data]).cpu().sum().float() + \
                (1 - lamda_beta) * predicted_one_labels[-half_data:].eq(
                label[-half_data:]).cpu().sum().float()
        else:
            minibatch_correct = predicted_one_labels.eq(label[:len(predicted_one_labels)]).cpu().sum().float()
        
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

        if torch.distributed.get_rank() == 0 and (batch_idx + 1) % config_args['log_interval'] == 0:
            epoch_str = 'Train Epoch {} '.format(epoch)

            if len(config_args['random_chunk']) == 2 and config_args['random_chunk'][0] <= \
                    config_args['random_chunk'][
                        1]:
                batch_length = data.shape[-1] if config_args['feat_format'] == 'wav' and 'trans_fbank' not in config_args else data.shape[-2]

            pbar.set_description(epoch_str)
            batch_dict = OrderedDict({'batch_length': batch_length,
                                      'accuracy': '{:>6.2f}%'.format(100. * minibatch_acc),
                                      'average_loss': '{:.4f}'.format(total_loss / (batch_idx + 1)),
                                      'lamda': '{:>4.2f}'.format(lamda_beta),
                                      })
            if total_other_loss != 0:
                batch_dict[config_args['second_loss']] = '{:.4f}'.format(total_other_loss / (batch_idx + 1))

            pbar.set_postfix(batch_dict)

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
    # parser.add_argument('--local_rank', default=-1, type=int,
    #                     help='node rank for distributed training')
    parser.add_argument('--train-config', default='', type=str,
                        help='node rank for distributed training')
    parser.add_argument('--seed', type=int, default=123456,
                        help='random seed (default: 0)')
    parser.add_argument('--lamda-beta', type=float, default=0.2,
                    help='random seed (default: 0)')

    args = parser.parse_args()

    all_seed(args.seed)
    torch.distributed.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # torch.multiprocessing.set_sharing_strategy('file_system')

    # load train config file args.train_config
    with open(args.train_config, 'r') as f:
        config_args = load_hyperpyyaml(f)

    # Create logger & Define visulaize SummaryWriter instance
    if isinstance(config_args['mixup_layer'], list):
        mixup_layer_str = ''.join([str(s) for s in config_args['mixup_layer']])
    else:
        mixup_layer_str = str(config_args['mixup_layer'])

    lambda_str = '_lamda' + str(args.lamda_beta)
    mixup_str = '_%s'%(config_args['mixup_type']) + mixup_layer_str + lambda_str
    if 'batmix_ratio' in config_args and config_args['batmix_ratio'] != 0.5:
        mixup_str += '_batmix{:.2f}'.format(config_args['batmix_ratio'])

    if 'mix_ratio' in config_args and config_args['mix_ratio'] != 0.5:
        mixup_str += '_mixrt{:.2f}'.format(config_args['mix_ratio'])

    config_args['lamda_beta'] = args.lamda_beta
    check_path = config_args['check_path'] + mixup_str + '/' + str(args.seed)
    
    if torch.distributed.get_rank() == 0:
        if not os.path.exists(check_path):
            print('Making checkpath...', check_path)
            os.makedirs(check_path)

        new_yaml_name = check_path + '/model.%s.yaml' % time.strftime("%Y.%m.%d", time.localtime())
        shutil.copy(args.train_config, new_yaml_name)
        with open(new_yaml_name, 'a') as f:
            import socket
            f.write('\nhostname: {}'.format(socket.gethostname()))
            f.write('\noriginal_yaml: {}'.format(args.train_config))
        
        model_yaml = config_args['check_path'] + f'{mixup_str}/model.yaml'
        if not os.path.isfile(model_yaml):
            shutil.copy(args.train_config, model_yaml)
            with open(model_yaml, 'a') as f:
                import socket
                f.write('\nhostname: {}'.format(socket.gethostname()))
                f.write('\noriginal_yaml: {}'.format(args.train_config))
        else:
            with open(new_yaml_name, 'rb') as f1, open(model_yaml, 'rb') as f2:
                hash1 = hashlib.md5(f1.read()).hexdigest()
                hash2 = hashlib.md5(f2.read()).hexdigest()
                if hash1 != hash2:
                    print('{} is not the same as model.yaml'.format(os.path.basename(new_yaml_name)))

        writer = SummaryWriter(logdir=check_path)
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
            print('#Speakers in training set is not equal to the classifier\'s #Speakers {}.\n'.format(config_args['num_classes']))

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
        if on_main():
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
            if on_main():
                print('Set the lr and weight_decay of filter layer to %f and %f' % (init_lr, init_wd))
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
    top_k_epoch = config_args['top_k_epoch'] if 'top_k_epoch' in config_args else 5
    if 'early_stopping' in config_args:
        early_stopping_scheduler = EarlyStopping(patience=config_args['early_patience'],
                                                min_delta=config_args['early_delta'],
                                                top_k_epoch=top_k_epoch)
    else:
        early_stopping_scheduler = None

    # Save model config txt
    if torch.distributed.get_rank() == 0:
        with open(os.path.join(check_path,
                               'model.%s.conf' % time.strftime("%Y.%m.%d", time.localtime())),
                  'w') as f:
            f.write('Model: ' + str(model) + '\n')
            f.write('Optimizer: ' + str(optimizer) + '\n')
            f.write('Scheduler: ' + str(scheduler) + '\n')

    if 'resume' in config_args:
        resume_checkpoint(model, scheduler, optimizer, config_args)

    start = 1 + start_epoch
    end = start + config_args['epochs']
    if torch.distributed.get_rank() == 0:
        print(' #Epochs,  {}  to  {} ...'.format(start, end))
        print(' #Params, Total: {} Trainable {}'.format(format(get_layer_param(model), ",d"),
                                                      format(get_trainable_param(model), ",d")))

    # if config_args['cuda']:
    if len(config_args['gpu_id']) > 1:
        print("Continue with gpu: %s ..." % str(local_rank))
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model.cuda(), device_ids=[local_rank])
    else:
        model = model.cuda()

    try:
        print('Dropout is {}.'.format(model.dropout_p))
    except:
        pass

    xvector_dir = check_path.replace('checkpoint', 'xvector')
    start_time = time.time()

    all_lr, valid_test_result = [],{}

    try:
        for epoch in range(start, end):

            # if torch. is_distributed():
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)
            train_extract_sampler.set_epoch(epoch)

            # pdb.set_trace()
            this_lr = [ param_group['lr'] for param_group in optimizer.param_groups]
            all_lr.append(max(this_lr))
            if torch.distributed.get_rank() == 0:
                lr_string = '\33[1;34m \'{}\' learning rate: '.format(config_args['optimizer'])
                lr_string += " ".join(['{:.8f} '.format(i) for i in this_lr])
                # lr_string += snr_str
                print('%s \33[0m' % lr_string)
                writer.add_scalar('Train/lr', this_lr[0], epoch)
            torch.distributed.barrier()
            if not torch.distributed.is_initialized():
                break

            train_mix(train_loader, model, optimizer,
                  epoch, scheduler, config_args, writer)

            valid_loss = valid_class(
                valid_loader, model, epoch, config_args, writer)

            if early_stopping_scheduler != None or epoch in [start, end-1]:
                valid_test_dict = valid_test(
                    train_extract_loader, model, epoch, xvector_dir, config_args, writer)
            else:
                valid_test_dict = {}

            valid_test_dict['Valid_Loss'] = valid_loss
            valid_test_result[epoch] = valid_test_dict

            if torch.distributed.get_rank() == 0 and early_stopping_scheduler != None:
                early_stopping_scheduler(
                    valid_test_dict[config_args['early_meta']], epoch)

                if early_stopping_scheduler.best_epoch + early_stopping_scheduler.patience >= end and this_lr[0] <= 0.1 ** 3 * config_args['lr']:
                    if config_args['scheduler'] != 'cyclic' or ('cyclic_epoch' in config_args and epoch - start >= 6*config_args['cyclic_epoch']):
                        early_stopping_scheduler.early_stop = True

                if config_args['scheduler'] != 'cyclic' and this_lr[0] <= 0.1 ** 3 * config_args['lr']:
                    if len(all_lr) > 5:
                        if all_lr[-5] >= this_lr[0]:
                            early_stopping_scheduler.early_stop = True

                        if this_lr[0] <= 0.1 ** 3 * config_args['lr'] and all_lr[-5] > this_lr[0]:
                            early_stopping_scheduler.early_stop = False

                top_k = early_stopping_scheduler.top_k()
            else:
                current_results = [[valid_test_result[e]['Valid_Loss'], e] for e in valid_test_result]
                tops = torch.tensor(current_results)
                top_k = tops[torch.argsort(tops[:, 0])][:top_k_epoch, 1].long().tolist()

            if on_main() and (epoch % config_args['test_interval'] == 0 or epoch in config_args['milestones'] or epoch >= (
                    end - 4) or epoch in top_k):
                save_checkpoint(model, optimizer, scheduler,
                    check_path, epoch)
                
            if early_stopping_scheduler != None:
                break_training = check_earlystop_break(early_stopping_scheduler,
                    start, end, epoch, check_path, valid_test_result)
            else:
                break_training = False
                if epoch == end - 1 and torch.distributed.get_rank() == 0:
                    best_str = test_results(best_res=valid_test_dict)
                    with open(os.path.join(check_path,
                                        'result.%s.txt' % time.strftime("%Y.%m.%d", time.localtime())), 'a+') as f:
                        f.write(best_str + '\n')

            if break_training:
                break

            if config_args['scheduler'] == 'rop':
                scheduler.step(valid_loss)
            elif config_args['scheduler'] == 'cyclic':
                continue
            else:
                scheduler.step()

            if torch.distributed.get_rank() == 0 and epoch == start:
                print("INFO: Epoch time: {:.4f} minutes.".format(float(time.time() - start_time) / 60))

    except KeyboardInterrupt:
        end = epoch

    stop_time = time.time()
    t = float(stop_time - start_time)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        writer.close()
        print("Running %.4f minutes for each epoch.\n" %
              (t / 60 / (max(end - start, 1))))
        
    time.sleep(3)
    # os.kill(os.getpid(), signal.SIGKILL)
    exit(0)


if __name__ == '__main__':
    main()
