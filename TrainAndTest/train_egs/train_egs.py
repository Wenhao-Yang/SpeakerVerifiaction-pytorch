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
import torch._utils

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
from hyperpyyaml import load_hyperpyyaml
from Define_Model.model import AttrDict
from Light.model import SpeakerLoss
import kaldiio
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from kaldi_io import read_mat, read_vec_flt
from kaldiio import load_mat
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel
from torch.optim import lr_scheduler
from tqdm import tqdm

from Light.dataset import SubDatasets, SubLoaders
from Define_Model.Optimizer import EarlyStopping
from Process_Data.Datasets.KaldiDataset import KaldiExtractDataset, \
    ScriptVerifyDataset
from TrainAndTest.common_func import create_classifier, create_optimizer, create_model, create_scheduler, verification_test, verification_extract, \
    args_parse, args_model, save_model_args
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


def train(train_loader, model, optimizer, epoch, scheduler, args, writer):
    # switch to train mode
    model.train()

    correct = 0.
    total_datasize = 0.
    total_loss = 0.
    total_other_loss = 0.

    pbar = tqdm(enumerate(train_loader))
    output_softmax = nn.Softmax(dim=1)
    # lambda_ = (epoch / args.epochs) ** 2
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

        if torch.distributed.is_initialized():
            loss, other_loss = model.module.loss(classfier, feats, label, epoch=epoch)
        else:
            loss, other_loss = model.loss(classfier, feats, label, epoch=epoch)

        predicted_labels = output_softmax(classfier.clone())
        predicted_one_labels = torch.max(predicted_labels, dim=1)[1]

        minibatch_correct = float((predicted_one_labels.cpu() == label.cpu()).sum().item())
        minibatch_acc=minibatch_correct / len(predicted_one_labels)

        correct += minibatch_correct
        total_datasize += len(predicted_one_labels)
        total_loss += float(loss.item())
        total_other_loss += other_loss

        if np.isnan(loss.item()):
            pdb.set_trace()
            raise ValueError('Loss value is NaN!')

        # compute gradient and update weights
        loss.backward()

        # gradient clip
        if args.grad_clip > 0:
            this_lr=args.lr
            for param_group in optimizer.param_groups:
                this_lr=min(param_group['lr'], this_lr)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        #
        if args.loss_ratio != 0:
            if args.loss_type in ['center', 'mulcenter', 'gaussian', 'coscenter']:
                for param in model.module.loss.xe_criterion.parameters():
                    param.grad.data *= (1. / args.loss_ratio)

        writer.add_scalar('Train/Total_Loss', float(loss.item()),
                          int((epoch - 1) * len(train_loader) + batch_idx + 1))

        if ((batch_idx + 1) % args.accu_steps) == 0:
            # optimizer the net
            optimizer.step()  # update parameters of net
            optimizer.zero_grad()  # reset gradient

            if args.model == 'FTDNN' and ((batch_idx + 1) % 4) == 0:
                if isinstance(model, DistributedDataParallel):
                    # The key method to constrain the first two convolutions, perform after every SGD step
                    model.module.step_ftdnn_layers()
                    orth_err += model.module.get_orth_errors()
                else:
                    # The key method to constrain the first two convolutions, perform after every SGD step
                    model.step_ftdnn_layers()
                    orth_err += model.get_orth_errors()

        # optimizer.step()
        if args.scheduler == 'cyclic':
            scheduler.step()

        if (batch_idx + 1) % args.log_interval == 0:
            
            epoch_str='Train Epoch {}: [ {:>5.1f}% ]'.format(
                epoch, 100. * batch_idx / len(train_loader))

            if len(args.random_chunk) == 2 and args.random_chunk[0] <= args.random_chunk[1]:
                batch_length = data.shape[-1] if args.feat_format == 'wav' else data.shape[-2]
                epoch_str += ' Batch Len: {:>3d} '.format(batch_length)

            epoch_str += ' Accuracy(%): {:>6.2f}%'.format(100. * minibatch_acc)

            if other_loss != 0:
                epoch_str += ' Other Loss: {:.4f}'.format(other_loss)
            
            epoch_str += ' Avg Loss: {:.4f}'.format(total_loss / (batch_idx + 1))

            pbar.set_description(epoch_str)
            # break

    this_epoch_str = 'Epoch {:>2d}: \33[91mTrain Accuracy: {:6.2f}%, Avg loss: {:7.4f}'.format(epoch, 100 * float(
        correct) / total_datasize, total_loss / len(train_loader))

    if other_loss != 0:
        this_epoch_str += ' {} Loss: {:>7.4f}'.format(
            args.loss_type, other_loss / len(train_loader))

    this_epoch_str += '.\33[0m'
    print(this_epoch_str)

    writer.add_scalar('Train/Accuracy', correct / total_datasize, epoch)
    writer.add_scalar('Train/Loss', total_loss / len(train_loader), epoch)

    torch.cuda.empty_cache()


def valid_class(valid_loader, model, epoch, args, writer):
    # switch to evaluate mode
    model.eval()

    total_loss = 0.
    total_other_loss = 0.
    correct = 0.
    total_datasize = 0.
    # ce_criterion, xe_criterion = ce
    softmax = nn.Softmax(dim=1)

    # lambda_ = (epoch / args.epochs) ** 2
    # 2. / (1 + np.exp(-10. * epoch / args.epochs)) - 1.

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(valid_loader):
            data = data.cuda()
            label = label.cuda()

            # compute output
            classfier, feats = model(data)
            if torch.distributed.is_initialized():
                loss, other_loss = model.module.loss(classfier, feats, label)
            else:
                loss, other_loss = model.loss(classfier, feats, label)

            total_loss += float(loss.item())
            total_other_loss += other_loss

            predicted_one_labels = softmax(classfier)
            predicted_one_labels = torch.max(predicted_one_labels, dim=1)[1]

            batch_correct = (predicted_one_labels.cuda() == label).sum().item()
            correct += batch_correct
            total_datasize += len(predicted_one_labels)

    valid_loss = total_loss / len(valid_loader)
    valid_accuracy = 100. * correct / total_datasize

    writer.add_scalar('Valid/Loss', valid_loss, epoch)
    writer.add_scalar('Valid/Accuracy', valid_accuracy, epoch)
    torch.cuda.empty_cache()

    this_epoch_str = '          \33[91mValid Accuracy: {:6.2f}%, Avg loss: {:>7.4f}'.format(
        valid_accuracy, valid_loss)

    if total_other_loss != 0:
        this_epoch_str += ' {} Loss: {:6f}'.format(
            args.loss_type, total_other_loss / len(valid_loader))
        
    this_epoch_str += '.\33[0m'
    print(this_epoch_str)

    return valid_loss


def valid_test(train_extract_loader, model, epoch, xvector_dir, args, writer):
    # switch to evaluate mode
    model.eval()

    this_xvector_dir = "%s/train/epoch_%s" % (xvector_dir, epoch)
    verification_extract(train_extract_loader, model,
                         this_xvector_dir, epoch, test_input=args.test_input)

    verify_dir = ScriptVerifyDataset(dir=args.train_test_dir, trials_file=args.train_trials,
                                     xvectors_dir=this_xvector_dir,
                                     loader=read_vec_flt)
    
    kwargs = {'num_workers': args.nj, 'pin_memory': False} 
    verify_loader = torch.utils.data.DataLoader(
        verify_dir, batch_size=128, shuffle=False, **kwargs)
    eer, eer_threshold, mindcf_01, mindcf_001 = verification_test(test_loader=verify_loader,
                                                                  dist_type=(
                                                                      'cos' if args.cos_sim else 'l2'),
                                                                  log_interval=args.log_interval,
                                                                  xvector_dir=this_xvector_dir,
                                                                  epoch=epoch)
    mix3 = 100. * eer * mindcf_01 * mindcf_001
    mix2 = 100. * eer * mindcf_001

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


def select_samples(train_dir, train_loader, model, args, select_score='loss'):
    model.eval()
    if torch.distributed.is_initialized():
        model.module.loss.xe_criterion.ce.reduction = 'none'
    else:
        model.loss.xe_criterion.ce.reduction = 'none'

    if isinstance(args, dict):
        args = AttrDict(args)

    score_dict = {}
    for i in range(train_dir.num_spks):
        score_dict[i] = []

    if len(train_dir.rest_dataset) > 0:
        train_dir.dataset = np.concatenate(
            [train_dir.dataset, train_dir.rest_dataset])

    all_loss = []

    with torch.no_grad():
        pbar = tqdm(enumerate(train_loader))
        for batch_idx, (data, label) in pbar:

            if 'loss' in select_score:
                if torch.cuda.is_available():
                    data = data.cuda()
                    label = label.cuda()

                classfier, feats = model(data)
                if torch.distributed.is_initialized():
                    loss, other_loss = model.module.loss(classfier, feats, label)
                else:
                    loss, other_loss = model.loss(classfier, feats, label)

            elif select_score == 'random':
                loss = torch.zeros_like(label)

            idx_labels = batch_idx * len(data) + np.arange(args.batch_size)
            for i, (l, sample_loss) in enumerate(zip(label, loss)):
                score_dict[int(l)].append([float(sample_loss), idx_labels[i]])
                all_loss.append(float(sample_loss))

    if torch.distributed.is_initialized():
        model.module.loss.xe_criterion.ce.reduction = 'mean'
    else:
        model.loss.xe_criterion.ce.reduction = 'mean'
    train_dataset = train_dir.dataset

    dataset = []
    rest_dataset = []

    all_loss = np.sort(all_loss)

    if select_score == 'loss_mean':
        threshold = np.mean(all_loss)
    elif select_score == 'loss_part':
        number_samples = int(len(all_loss)*args.coreset_percent)
        number_samples = int(
            np.ceil(number_samples / args.batch_size) * args.batch_size)
        threshold = all_loss[-number_samples]

    for i in score_dict:
        sort_np = np.array(score_dict[i])
        if select_score in ['loss', 'random']:
            idx = np.argsort(sort_np, axis=0)
            sort_np = sort_np[idx[:, 0]]
            sort_np_len = len(sort_np)

            # pdb.set_trace()
            for _, j in sort_np[-int(sort_np_len*args.coreset_percent):]:
                dataset.append(train_dataset[int(j)])

            for _, k in sort_np[:-int(sort_np_len*args.coreset_percent)]:
                rest_dataset.append(train_dataset[int(k)])
        else:
            for l, j in sort_np:
                if l >= threshold:
                    dataset.append(train_dataset[int(j)])
                else:
                    rest_dataset.append(train_dataset[int(j)])

    # pdb.set_trace()

    dataset = np.array(dataset)
    dataset = dataset[np.argsort(dataset, axis=0)[:, 2]]
    if len(dataset) % args.batch_size > 0:
        dataset = dataset[:-(len(dataset) % args.batch_size)]

    np.random.shuffle(dataset)

    train_dir.dataset = dataset
    train_dir.rest_dataset = rest_dataset



def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive
    # test_display_triplet_distance = False
    # print the experiment configuration
    print('\nCurrent time is \33[91m{}\33[0m.'.format(str(time.asctime())))

    # Training settings
    args = args_parse('PyTorch Speaker Recognition: Classification')
    
    if os.path.exists(args.train_config):
        with open(args.train_config, 'r') as f:
            config_args = load_hyperpyyaml(f)
    else:
        config_args = vars(args)
    
    model_str = 'baseline'
    if 'mix_type' in config_args and len(config_args['mixup_layer'])>0:
        if isinstance(config_args['mixup_layer'], list):
            mixup_layer_str = ''.join([str(s) for s in config_args['mixup_layer']])
        else:
            mixup_layer_str = str(config_args['mixup_layer'])

        lambda_str = '_lamda' + str(args.lamda_beta)
        model_str = '/mani' + mixup_layer_str + lambda_str

    check_path = config_args['check_path'] + model_str + '/' + str(args.seed)

    # create logger & Define visulaize SummaryWriter instance
    args_object = AttrDict(config_args)
    if not os.path.exists(check_path):
        print('Making checkpath...')
        os.makedirs(check_path)

    writer = SummaryWriter(logdir=check_path)
    sys.stdout = NewLogger(osp.join(check_path, 'log.%s.txt' %
                                    time.strftime("%Y.%m.%d", time.localtime())))
    
    keys = list(config_args.keys())
    keys.sort()
    options = ["\'%s\': \'%s\'" % (
        str(k), str(config_args[k])) for k in keys]
    print('Parsed options: \n{ %s }' % (', '.join(options)))

    # Set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
    # order to prevent any memory allocation on unused GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '29555'

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    torch.multiprocessing.set_sharing_strategy('file_system')
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.benchmark = True

    # Datasets
    train_dir, valid_dir, train_extract_dir = SubDatasets(config_args)
    train_loader, valid_loader, train_extract_loader = SubLoaders(
        train_dir, valid_dir, train_extract_dir, config_args)
    
    print('Number of Speakers: {}.\n'.format(train_dir.num_spks))

    # instantiate model and initialize weights
    if 'embedding_model' in config_args:
        model = config_args['embedding_model']
    else:
        model_kwargs = args_model(args, train_dir)

        keys = list(model_kwargs.keys())
        keys.sort()
        model_options = ["\'%s\': \'%s\'" % (
            str(k), str(model_kwargs[k])) for k in keys]
        print('Model options: \n{ %s }' % (', '.join(model_options)))
        print('Testing with %s distance, ' % ('cos' if args.cos_sim else 'l2'))

        model = create_model(args.model, **model_kwargs)
        model_yaml_path = os.path.join(
            check_path, 'model.%s.yaml' % time.strftime("%Y.%m.%d", time.localtime()))
        save_model_args(model_kwargs, model_yaml_path)

    model.loss = SpeakerLoss(config_args)
    
    if 'classifier' in config_args:
        model.classifier = config_args['classifier']
    else:
        create_classifier(model, **config_args)    
    
    start_epoch = 0
    if args.save_init and not args.finetune:
        checkpoint_path = '{}/checkpoint_{}_{}.pth'.format(check_path, start_epoch,
                                                      time.strftime('%Y_%b_%d_%H:%M', time.localtime()))
        if not os.path.exists(checkpoint_path):
            torch.save({'state_dict': model.state_dict()}, checkpoint_path)

    # Load checkpoint
    iteration = 0  # if args.resume else 0
    if args.finetune and args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
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
            print('=> no checkpoint found at {}'.format(args.resume))
    
    model_para = [{'params': model.parameters()}]
    if args.loss_type in ['center', 'variance', 'mulcenter', 'gaussian', 'coscenter', 'ring']:
        assert args.lr_ratio > 0
        model_para.append(
            {'params': model.loss.xe_criterion.parameters(), 'lr': args.lr * args.lr_ratio})

    if args.finetune or args.second_wd > 0:
        # if args.loss_type in ['asoft', 'amsoft']:
        classifier_params = list(map(id, model.classifier.parameters()))
        rest_params = filter(lambda p: id(
            p) not in classifier_params, model.parameters())
        init_lr = args.lr * args.lr_ratio if args.lr_ratio > 0 else args.lr
        init_wd = args.second_wd if args.second_wd > 0 else args.weight_decay
        print('Set the lr and weight_decay of classifier to %f and %f' %
              (init_lr, init_wd))
        model_para = [{'params': rest_params},
                      {'params': model.classifier.parameters(), 'lr': init_lr, 'weight_decay': init_wd}]

    if args.filter in ['fDLR', 'fBLayer', 'fLLayer', 'fBPLayer']:
        filter_params = list(map(id, model.filter_layer.parameters()))
        rest_params = filter(lambda p: id(
            p) not in filter_params, model_para[0]['params'])
        init_wd = args.filter_wd if args.filter_wd > 0 else args.weight_decay
        init_lr = args.lr * args.lr_ratio if args.lr_ratio > 0 else args.lr
        print('Set the lr and weight_decay of filter layer to %f and %f' %
              (init_lr, init_wd))
        model_para[0]['params'] = rest_params
        model_para.append({'params': model.filter_layer.parameters(), 'lr': init_lr,
                           'weight_decay': init_wd})

    opt_kwargs = {'lr': config_args['lr'],
                    'lr_decay': config_args['lr_decay'],
                    'weight_decay': config_args['weight_decay'],
                    'dampening': config_args['dampening'],
                    'momentum': config_args['momentum'],
                    'nesterov': config_args['nesterov']}

    optimizer = create_optimizer(model_para, config_args['optimizer'], **opt_kwargs)
    scheduler = create_scheduler(optimizer, config_args)
    early_stopping_scheduler = EarlyStopping(patience=config_args['early_patience'],
                                             min_delta=config_args['early_delta'])
    
    if not args.finetune and args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
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
                    name = k[7:]
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
            print('=> no checkpoint found at {}'.format(args.resume))

    # Save model config txt
    with open(osp.join(check_path, 'model.%s.conf' % time.strftime("%Y.%m.%d", time.localtime())), 'w') as f:
        f.write('model: ' + str(model) + '\n')
        f.write('Optimizer: ' + str(optimizer) + '\n')

    start = args.start_epoch + start_epoch
    print('Start epoch is : ' + str(start))
    # start = 0
    end = start + args.epochs

    if args.cuda:
        if len(args.gpu_id) > 1:
            print("Continue with gpu: %s ..." % str(args.gpu_id))
            # torch.distributed.init_process_group(backend="nccl",
            #                                      init_method='file:///home/ssd2020/yangwenhao/lstm_speaker_verification/data/sharedfile',
            #                                      rank=0,
            #                                      world_size=1)
            
            torch.distributed.init_process_group(backend="nccl", init_method='tcp://localhost:32456', rank=0,
                                                 world_size=1)
            # if args.gain
            # model = DistributedDataParallel(model.cuda(), find_unused_parameters=True)
            model = DistributedDataParallel(model.cuda())

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
            # pdb.set_trace()
            lr_string = '\n\33[1;34m Current \'{}\' learning rate is '.format(
                args.optimizer)
            this_lr = []
            for param_group in optimizer.param_groups:
                this_lr.append(param_group['lr'])
                lr_string += '{:.10f} '.format(param_group['lr'])
            print('%s \33[0m' % lr_string)
            all_lr.append(this_lr[0])
            writer.add_scalar('Train/lr', this_lr[0], epoch)

            if args.coreset_percent > 0 and epoch % args.select_interval == 1:
                select_samples(train_dir, train_loader, model, args, args.select_score)

            train(train_loader, model, optimizer, epoch, scheduler, args, writer)
            if config_args['batch_shuffle']:
                train_dir.__shuffle__()
                
            valid_loss = valid_class(valid_loader, model, epoch, args, writer)
            if args.early_stopping or (epoch % args.test_interval == 1 or epoch in milestones or epoch == (
                    end - 1)):
                valid_test_dict = valid_test(
                    train_extract_loader, model, epoch, xvector_dir, args, writer)
            else:
                valid_test_dict = {}

            valid_test_dict['Valid_Loss'] = valid_loss
            valid_test_result.append(valid_test_dict)

            if args.early_stopping:
                early_stopping_scheduler(
                    valid_test_dict[args.early_meta], epoch)
                if early_stopping_scheduler.best_epoch + early_stopping_scheduler.patience >= end:
                    early_stopping_scheduler.early_stop = True

                if args.scheduler != 'cyclic' and this_lr[0] <= 0.1 ** 3 * args.lr:
                    if len(all_lr) > 5 and all_lr[-5] == this_lr[0]:
                        early_stopping_scheduler.early_stop = True

            if epoch % args.test_interval == (args.test_interval-1) or epoch in milestones or epoch == (
                    end - 1) or early_stopping_scheduler.best_epoch == epoch:
                model.eval()
                check_path = '{}/checkpoint_{}.pth'.format(
                    check_path, epoch)
                model_state_dict = model.module.state_dict() \
                    if isinstance(model, DistributedDataParallel) else model.state_dict()
                torch.save({'epoch': epoch, 'state_dict': model_state_dict,
                            'scheduler': scheduler.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            }, check_path)

                if args.early_stopping:
                    pass
                # elif early_stopping_scheduler.best_epoch == epoch or (
                #         args.early_stopping == False and epoch % args.test_interval == 1):
                # elif epoch % args.test_interval == 1 or epoch == (end - 1):
                #     test(model, epoch, writer, xvector_dir)

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

                try:
                    shutil.copy('{}/checkpoint_{}.pth'.format(check_path, early_stopping_scheduler.best_epoch),
                                '{}/best.pth'.format(check_path))
                except Exception as e:
                    print(e)
                end = epoch
                break

            if args.scheduler == 'rop':
                scheduler.step(valid_loss)
            elif args.scheduler == 'cyclic':
                continue
            else:
                scheduler.step()

    except KeyboardInterrupt:
        end = epoch

    writer.close()
    stop_time = time.time()
    t = float(stop_time - start_time)
    print("Running %.4f minutes for each epoch.\n" %
          (t / 60 / (max(end - start, 1))))
    exit(0)


if __name__ == '__main__':
    main()
