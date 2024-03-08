#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: train_egs_dist.py
@Time: 2024/1/09 11:52
@Overview:
"""
from __future__ import print_function
import pdb
from Define_Model.TDNN.ECAPA_brain import Classifier
from Light.dataset import Sampler_Loaders, SubScriptDatasets
from Light.model import SpeakerLoss
from Process_Data.Datasets.SelectDataset import Forgetting, GraNd, LossSelect, OTSelect, RandomSelect
import torch._utils

import argparse
import os
import shutil
import sys
import time
# Version conflict
import warnings
from collections import OrderedDict

import numpy as np
import torch

from hyperpyyaml import load_hyperpyyaml
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
# from torch.optim import lr_scheduler
import torch.distributed as dist

from Define_Model.Optimizer import EarlyStopping
from TrainAndTest.common_func import create_classifier, create_optimizer, create_scheduler
from logger import NewLogger
# import pytorch_lightning as pl
from TrainAndTest.train_egs.train_dist import all_seed, train, valid_class, valid_test, test_results

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch ( Distributed ) Speaker Recognition: Classification')
    # parser.add_argument('--local_rank', default=-1, type=int,
    #                     help='node rank for distributed training')
    parser.add_argument('--train-config', default='', type=str,
                        help='node rank for distributed training')
    parser.add_argument('--seed', type=int, default=123456,
                        help='random seed (default: 0)')
    parser.add_argument('--sample-ratio', type=int,
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
    check_path = config_args['check_path'] 
    if args.sample_ratio < 100:
        check_path += '_{}{}'.format(config_args['select_method'], args.sample_ratio) 
    
    check_path += '/' + str(args.seed)

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

        writer = SummaryWriter(logdir=check_path, filename_suffix='SV')
        sys.stdout = NewLogger(
            os.path.join(check_path, 'log.%s.txt' % time.strftime("%Y.%m.%d", time.localtime())))
    else:
        writer = None

    # Full Dataset  
    train_dir, valid_dir, train_extract_dir = SubScriptDatasets(config_args)

    # model = create_model(config_args['model'], **model_kwargs)
    if 'embedding_model' in config_args:
        model = config_args['embedding_model']

    if 'classifier' in config_args:
        model.classifier = config_args['classifier']
    else:
        if not isinstance(model.classifier, Classifier):
            create_classifier(model, **config_args)

    start_epoch = 0
    if 'finetune' not in config_args or not config_args['finetune']:
        this_check_path = '{}/checkpoint_{}_{}.pth'.format(check_path, start_epoch,
                                                           time.strftime('%Y_%b_%d_%H:%M', time.localtime()))
        if not os.path.exists(this_check_path):
            torch.save({'state_dict': model.state_dict()}, this_check_path)

    # Load checkpoint
    if 'fintune' in config_args:
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
            if list(filtered.keys())[0].startswith('module'):
                new_state_dict = OrderedDict()
                for k, v in filtered.items():
                    # remove `module.`，表面从第7个key值字符取到最后一个字符，去掉module.
                    new_state_dict[k[7:]] = v  # 新字典的key值对应的value为一一对应的值。

                model.load_state_dict(new_state_dict)
                del new_state_dict
            else:
                model_dict = model.state_dict()
                model_dict.update(filtered)
                model.load_state_dict(model_dict)
                del model_dict

            del checkpoint_state_dict, filtered, checkpoint
            # model.dropout.p = args.dropout_p
        else:
            print('=> no checkpoint found at {}'.format(config_args['resume']))

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
            p) not in classifier_params, model.parameters())

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
    scheduler = create_scheduler(optimizer, config_args, train_dir)

    top_k_epoch = config_args['top_k_epoch'] if 'top_k_epoch' in config_args else 5
    if 'early_stopping' in config_args and config_args['early_stopping']:
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

    if 'resume' in config_args:
        if 'fintune' in config_args:
            pass
        else:
            if os.path.isfile(config_args['resume']):
                if torch.distributed.get_rank() == 0:
                    print('=> loading  model   check: {}'.format(config_args['resume']))
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

            if 'resume_optim' in config_args and os.path.isfile(config_args['resume_optim']):
                if torch.distributed.get_rank() == 0:
                    print('=> loading optmizer check {}'.format(config_args['resume_optim']))
                optm_check = torch.load(config_args['resume_optim'])
                if 'scheduler' in checkpoint:
                    scheduler.load_state_dict(optm_check['scheduler'])
                if 'optimizer' in checkpoint:
                    optimizer.load_state_dict(optm_check['optimizer'])
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
            model.cuda(), device_ids=[local_rank])
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
    
    if config_args['select_method'] == 'grand':
        select_method = GraNd

    elif config_args['select_method'] == 'random':
        select_method = RandomSelect
    elif config_args['select_method'] == 'loss':
        select_method = LossSelect
    elif config_args['select_method'] == 'optimal':
        select_method = OTSelect
    elif config_args['select_method'] == 'forget':
        select_method = Forgetting

    if torch.distributed.get_rank() == 0:
        print('\nCurrent time is \33[91m{}\33[0m.'.format(str(time.asctime())))
        print('Number of Speakers: {}.\n'.format(train_dir.num_spks))
        if train_dir.num_spks != config_args['num_classes']:
            print('Number of Speakers in training set is not equal to the asigned number.\n'.format(
                train_dir.num_spks))

        print('Testing with %s distance, ' %
            ('cos' if config_args['cos_sim'] else 'l2'))
        
    select_repeat = config_args['select_repeat'] if 'select_repeat' in config_args else 4
    spk_balance = False if 'select_balance' not in config_args else config_args['select_balance']
    select_sample = 'top' if 'select_sample' not in config_args else config_args['select_sample']
    stratas = 50 if 'stratas' not in config_args else config_args['stratas']
    stratas_select = 'random' if 'stratas_select' not in config_args else config_args['stratas_select']
    
    sample_ratio = args.sample_ratio / 100
    selector = select_method(train_dir,
                             args=config_args, repeat=select_repeat,
                             random_seed=args.seed,
                             balance=spk_balance,
                             save_dir=check_path,
                             select_sample=select_sample, stratas=stratas,
                             stratas_select=stratas_select,
                             fraction=sample_ratio)
    
    if config_args['select_method'] == 'forget':
        config_args['forgetting_events'] = selector
    
    total_size = int(np.ceil(scheduler.total_size * sample_ratio))
    train_loader = None
    
    try:
        for epoch in range(start, end):          
            # select coreset
            if 'full_epoch' in config_args and epoch <= config_args['full_epoch']:
                subtrain_dir = train_dir
                if train_loader == None:
                    train_loader, train_sampler, valid_loader, valid_sampler, train_extract_loader, train_extract_sampler = Sampler_Loaders(
                    subtrain_dir, valid_dir, train_extract_dir, config_args, verbose=0)

            elif ('full_epoch' in config_args and epoch == config_args['full_epoch']+1) or \
                (epoch % config_args['select_epoch'] == 1):
                
                last_epoch = scheduler.last_epoch
                scheduler.last_epoch = int(last_epoch / scheduler.total_size * total_size)
                scheduler.total_size = total_size

                subtrain_dir = selector.select(model=model)
                
                # print('subset length:', len(subtrain_dir))
                train_loader, train_sampler, valid_loader, valid_sampler, train_extract_loader, train_extract_sampler = Sampler_Loaders(
                    subtrain_dir, valid_dir, train_extract_dir, config_args, verbose=0)

            torch.distributed.barrier()
            # print('train_loader length:', len(train_loader))
        
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)
            train_extract_sampler.set_epoch(epoch)

            if 'linear_snr' in config_args:
                total_snr = np.linspace(config_args['snr_start'], config_args['snr_stop'],
                                       config_args['snr_num'])
                snr_idx = min(epoch-1, config_args['snr_num']-1)
                this_snr = total_snr[snr_idx]
                
                for aug in config_args['augment_pipeline']:
                    if hasattr(aug, 'add_noise'):
                        aug.add_noise.snr_high = this_snr
                
                snr_str = ' snr: {:.2f}'.format(this_snr)
            else:
                snr_str = ''

            # if torch.distributed.get_rank() == 0:
            this_lr = [ param_group['lr'] for param_group in optimizer.param_groups]
            all_lr.append(max(this_lr)) 
            if torch.distributed.get_rank() == 0:
                lr_string = '\33[1;34m \'{}\' learning rate: '.format(config_args['optimizer'])
                lr_string += " ".join(['{:.8f} '.format(i) for i in this_lr])
                lr_string += snr_str
                print('%s \33[0m' % lr_string)
                writer.add_scalar('Train/lr', this_lr[0], epoch)

            torch.distributed.barrier()

            train(train_loader, model, optimizer,
                  epoch, scheduler, config_args, writer)

            valid_loss = valid_class(
                valid_loader, model, epoch, config_args, writer)
            if early_stopping_scheduler != None or epoch == end - 1:
                valid_test_dict = valid_test(
                    train_extract_loader, model, epoch, xvector_dir, config_args, writer)
            else:
                valid_test_dict = {}

            valid_test_dict['Valid_Loss'] = valid_loss

            if torch.distributed.get_rank() == 0 and early_stopping_scheduler != None:
                valid_test_result.append(valid_test_dict)

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

            elif early_stopping_scheduler == None:
                valid_test_result.append(valid_test_dict)

            if torch.distributed.get_rank() == 0 and early_stopping_scheduler != None:
                top_k = early_stopping_scheduler.top_k()
            else:
                top_k = []

            if torch.distributed.get_rank() == 0 and (epoch % config_args['test_interval'] == 0 or epoch in config_args['milestones'] or epoch == (
                    end - 1) or epoch in top_k):

                model.eval()
                this_check_path = '{}/checkpoint_{}.pth'.format(
                    check_path, epoch)
                model_state_dict = model.module.state_dict() \
                    if isinstance(model, DistributedDataParallel) else model.state_dict()
                    
                torch.save({'epoch': epoch, 'state_dict': model_state_dict},
                           this_check_path)

                this_optim_path = '{}/optim_{}.pth'.format(
                    check_path, epoch)
                torch.save({'epoch': epoch,
                            'scheduler': scheduler.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            }, this_optim_path)

            if early_stopping_scheduler != None:
                check_stop = torch.tensor(
                    int(early_stopping_scheduler.early_stop)).cuda()
                dist.all_reduce(check_stop, op=dist.ReduceOp.SUM)
            else:
                check_stop = False

            if check_stop or epoch == end - 1:
                end = epoch
                if torch.distributed.get_rank() == 0:
                    if early_stopping_scheduler != None:
                        print('Best Epochs : ', top_k)
                        best_epoch = early_stopping_scheduler.best_epoch
                        best_res = valid_test_result[int(best_epoch - start)]
                    else:
                        valid_losses = [v['Valid_Loss'] for v in valid_test_result]
                        tops = torch.tensor(valid_losses)
                        top_k = torch.argsort(tops)[:top_k_epoch]
        
                        print('Best Epochs : ', top_k)
                        best_res = valid_test_result[-1]
                    
                    best_str = test_results(best_res)

                    with open(os.path.join(check_path, 'result.%s.txt' % time.strftime("%Y.%m.%d", time.localtime())), 'a+') as f:
                        f.write(best_str + '\n')

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
        # print("Running %.4f minutes for each epoch.\n" %
        #       (t / 60 / (max(end - start, 1))))

    time.sleep(5)
    exit(0)


if __name__ == '__main__':
    main()
