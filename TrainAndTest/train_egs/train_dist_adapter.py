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
import copy
from Define_Model.ParallelBlocks import Adapter, gumbel_softmax
from Define_Model.model import create_classifier
from Light.dataset import Sampler_Loaders, SubScriptDatasets
from Light.model import SpeakerLoss
from TrainAndTest.train_egs.train_egs import select_samples
import torch._utils

import argparse
import hashlib
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
from TrainAndTest.common_func import load_checkpoint, on_main, resume_checkpoint, verification_test, verification_extract
#, args_parse
# from Define_Model.model import create_model, args_model, save_model_args 
from Define_Model.Optimizer import create_optimizer, create_scheduler
from logger import NewLogger

from TrainAndTest.train_egs.train_dist import all_seed, test_results, save_checkpoint, check_earlystop_break, valid_test, valid_class #, train
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

def train(train_loader, model, optimizer, epoch, scheduler, config_args, writer):
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
    # lambda_ = (epoch / config_args['epochs']) ** 2

    if 'augment_pipeline' in config_args:
        num_pipes = config_args['num_pipes'] if 'num_pipes' in config_args else 1
        augment_pipeline = []
        for _, augment in enumerate(config_args['augment_pipeline']):
            augment_pipeline.append(augment.cuda())

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

        # print(data.shape)
        # pdb.set_trace()
        if 'augment_pipeline' in config_args:
            with torch.no_grad():
                wavs_aug_tot = []
                wavs_aug_tot.append(data.cuda()) # data_shape [batch, 1,1,time]
                wavs = data.squeeze().cuda()

                # augment = np.random.choice(augment_pipeline)
                # for count, augment in enumerate(augment_pipeline):
                for augment in np.random.choice(augment_pipeline, size=num_pipes, replace=False):
                    # Apply augment
                    wavs_aug = augment(wavs, torch.tensor([1.0]*len(data)).cuda())
                    # Managing speed change
                    if wavs_aug.shape[1] > wavs.shape[1]:
                        wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
                    else:
                        zero_sig = torch.zeros_like(wavs)
                        zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                        wavs_aug = zero_sig

                    if 'concat_augment' in config_args and config_args['concat_augment']:
                        wavs_aug_tot.append(wavs_aug.unsqueeze(1).unsqueeze(1))
                    else:
                        wavs = wavs_aug
                        wavs_aug_tot[0] = wavs.unsqueeze(1).unsqueeze(1)
                
                data = torch.cat(wavs_aug_tot, dim=0)
                n_augment = len(wavs_aug_tot)
                label = torch.cat([label] * n_augment)

        if torch.cuda.is_available():
            label = label.cuda().long()
            data = data.cuda()

        data, label = Variable(data), Variable(label)
        # pdb.set_trace()
        # (classfier, other_loss), feats = model(data)
        classfier, feats = model(data)
        # print('label', label)
        loss, other_loss = model.module.loss((classfier, feats), label, other=True)
        # loss, other_loss = model.module.loss(classfier, feats, label,
        #                                      batch_weight=batch_weight, epoch=epoch)
        # loss = loss + other_loss

        predicted_labels = output_softmax(classfier.clone())
        predicted_one_labels = torch.max(predicted_labels, dim=1)[1]

        minibatch_correct = float(
            (predicted_one_labels.cpu() == label.cpu()).sum().item())
        minibatch_acc = minibatch_correct / len(predicted_one_labels)
        correct += minibatch_correct

        total_datasize += len(predicted_one_labels)
        # print(loss.shape)
        total_loss += float(loss.item())
        total_other_loss +=  float(other_loss)

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

            pbar.set_description(epoch_str)
            batch_dict = OrderedDict({'batch_length': batch_length, 'accuracy': '{:>6.2f}%'.format(
                100. * minibatch_acc), 'average_loss': '{:.4f}'.format(total_loss / (batch_idx + 1))})
            if total_other_loss != 0:
                batch_dict[config_args['second_loss']] = '{:.4f}'.format(total_other_loss / (batch_idx + 1))

            pbar.set_postfix(batch_dict)

    this_epoch_str = 'Epoch {:>2d}: \33[91mTrain Accuracy: {:.6f}%, Avg loss: {:6f}'.format(epoch, 100 * float(
        correct) / total_datasize, total_loss / len(train_loader))

    if total_other_loss != 0:
        this_epoch_str += ' {} Loss: {:6f}'.format(
            config_args['second_loss'], total_other_loss / len(train_loader))

    this_epoch_str += '.\33[0m'

    if torch.distributed.get_rank() == 0:
        print(this_epoch_str)
        writer.add_scalar('Train/Accuracy', correct / total_datasize, epoch)
        writer.add_scalar('Train/Loss', total_loss / len(train_loader), epoch)

    torch.cuda.empty_cache()


# Training settings
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
    torch.multiprocessing.set_sharing_strategy('file_system')

    # load train config file args.train_config
    with open(args.train_config, 'r') as f:
        config_args = load_hyperpyyaml(f)

    # Create logger & Define visulaize SummaryWriter instance
    check_path = config_args['check_path'] + '/' + str(args.seed)
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
        
        model_yaml = config_args['check_path'] + '/model.yaml'
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
    model = config_args['adapter']
    
    # Adapter(model, scale=config_args['scale'],
    #                 layers=config_args['layers'],
    #                 adapter_type=config_args['adapter_type'])
    # if 'embedding_model' in config_args:
    #     model = config_args['embedding_model']

    # if 'classifier' in config_args:
    #     model.classifier = config_args['classifier']
    # else:
    #     create_classifier(model, **config_args)

    if 'agent_model' in config_args:
        agent_model = config_args['agent_model']
    else:
        agent_model = None

    start_epoch = 0
    check_path = config_args['check_path'] + '/' + str(args.seed)
    if 'finetune' not in config_args or not config_args['finetune']:
        this_check_path = '{}/checkpoint_{}_{}.pth'.format(check_path, start_epoch,
                                                           time.strftime('%Y_%b_%d_%H:%M', time.localtime()))
        if not os.path.exists(this_check_path):
            torch.save({'state_dict': model.state_dict()}, this_check_path)

    # Load checkpoint
    if 'fintune' in config_args:
        start_epoch += load_checkpoint(model.model, config_args)
    
    model.loss = SpeakerLoss(config_args)

    model_para = [{'params': model.parameters()}]
    if config_args['loss_type'] in ['center', 'variance', 'mulcenter', 'gaussian', 'coscenter', 'ring']:
        assert config_args['lr_ratio'] > 0
        model_para.append({'params': model.loss.xe_criterion.parameters(
        ), 'lr': config_args['lr'] * config_args['lr_ratio']})

    if 'multi_lr' in config_args:
        name2lr = config_args['multi_lr']
        lr2ps = {}
        for k in set(name2lr.keys()):
            if name2lr[k] != 0:
                lr2ps[name2lr[k]] = []
                
        lr_list = list(lr2ps.keys())
        lr_list.sort()

        default_lr = config_args['lr']
        model_para = []

        if 'second_wd' in config_args:
            init_wd = config_args['second_wd'] if config_args['second_wd'] > 0 else config_args['weight_decay']
            classifier_params = {'params': [], 'weight_decay': init_wd}

        for n,p in model.named_parameters():
            
            this_key = default_lr
            for key in name2lr:
                if key in n:
                    this_key = name2lr[key]
            
            if this_key == 0:
                continue
            
            if 'second_wd' in config_args and 'classifier' in n:
                classifier_params['params'].append(p)
                classifier_params['lr'] = this_key
            else:
                lr2ps[this_key].append(p)
            
        model_para.extend([{'params': lr2ps[lr], 'lr': lr} for lr in lr_list])
        if 'second_wd' in config_args and 'lr' in classifier_params:
            model_para.append(classifier_params)
            # if 'classifier' not in set(name2lr.keys()):
            lr_list.append(classifier_params['lr'])

        config_args['lr_list'] = lr_list
        if torch.distributed.get_rank() == 0:
            print('learning rate lst: ', lr_list)

    elif 'second_wd' in config_args and config_args['second_wd'] > 0:
        # if config_args['loss_type in ['asoft', 'amsoft']:
        classifier_params = list(map(id, model.classifier.parameters()))
        rest_params = filter(lambda p: id(
            p) not in classifier_params and p.requires_grad, model.parameters())

        init_lr = config_args['lr'] * \
            config_args['lr_ratio'] if config_args['lr_ratio'] > 0 else config_args['lr']
        init_wd = config_args['second_wd'] if config_args['second_wd'] > 0 else config_args['weight_decay']
        if torch.distributed.get_rank() == 0:
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
        if torch.distributed.get_rank() == 0:
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
    scheduler = create_scheduler(optimizer, config_args, train_loader)
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
            f.write('Model:     ' + str(model) + '\n')
            f.write('Optimizer: ' + str(optimizer) + '\n')
            f.write('Scheduler: ' + str(scheduler) + '\n')

    if 'resume' in config_args and 'fintune' not in config_args:
        resume_checkpoint(model, scheduler, optimizer, config_args)

    start = 1 + start_epoch
    if torch.distributed.get_rank() == 0:
        print('Start epoch is : ' + str(start))
    end = start + config_args['epochs']

    if torch.distributed.get_world_size() > 1:
        print("Continue with gpu: %s ..." % str(local_rank))
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        find_unused_parameters = False if 'agent_model' in config_args else True
        model = DistributedDataParallel(
            model.cuda(), device_ids=[local_rank], find_unused_parameters=find_unused_parameters)
        # agent_model = DistributedDataParallel(
        #     agent_model.cuda(), device_ids=[local_rank])
    else:
        model = model.cuda()
        # agent_model = agent_model.cuda()

    try:
        print('Dropout is {}.'.format(model.dropout_p))
    except:
        pass

    xvector_dir = check_path.replace('checkpoint', 'xvector')
    start_time = time.time()

    all_lr, valid_test_result = [],{}

    try:
        for epoch in range(start, end):
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)
            train_extract_sampler.set_epoch(epoch)

            # if torch.distributed.get_rank() == 0:
            this_lr = [ param_group['lr'] for param_group in optimizer.param_groups]
            all_lr.append(max(this_lr))
            if torch.distributed.get_rank() == 0:
                lr_string = '\33[1;34m \'{}\' learning rate: '.format(config_args['optimizer'])
                lr_string += " ".join(['{:.8f} '.format(i) for i in this_lr])
                print('%s \33[0m' % lr_string)
                writer.add_scalar('Train/lr', this_lr[0], epoch)

            torch.distributed.barrier()

            train(train_loader, model, optimizer,
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
                pass
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

    time.sleep(5)
    exit(0)

if __name__ == '__main__':
    main()
