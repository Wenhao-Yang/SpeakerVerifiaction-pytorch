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
from kaldi_io import read_mat, read_vec_flt
from kaldiio import load_mat
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel
from torch.optim import lr_scheduler
from tqdm import tqdm

from Define_Model.Loss.LossFunction import CenterLoss, Wasserstein_Loss, MultiCenterLoss, CenterCosLoss, RingLoss, \
    VarianceLoss, DistributeLoss, MMD_Loss, aDCFLoss
from Define_Model.Loss.SoftmaxLoss import AngleSoftmaxLoss, AngleLinear, AdditiveMarginLinear, AMSoftmaxLoss, \
    ArcSoftmaxLoss, \
    GaussianLoss, MinArcSoftmaxLoss, MinArcSoftmaxLoss_v2, SubArcSoftmaxLoss
from Define_Model.Optimizer import EarlyStopping
from Process_Data.Datasets.KaldiDataset import KaldiExtractDataset, \
    ScriptVerifyDataset
from Process_Data.Datasets.LmdbDataset import EgsDataset
import Process_Data.constants as C
from Define_Model.TDNN.Slimmable import FLAGS
from Process_Data.audio_processing import ConcateVarInput, tolog, ConcateOrgInput, PadCollate
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
args = args_parse('PyTorch Speaker Recognition: Classification')

# Set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '29555'

args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

# torch.multiprocessing.set_sharing_strategy('file_system')
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

# create logger
# Define visulaize SummaryWriter instance
writer = SummaryWriter(logdir=args.check_path, filename_suffix='_first')
sys.stdout = NewLogger(osp.join(args.check_path, 'log.%s.txt' % time.strftime("%Y.%m.%d", time.localtime())))

kwargs = {'num_workers': args.nj, 'pin_memory': False} if args.cuda else {}
extract_kwargs = {'num_workers': 4, 'pin_memory': False} if args.cuda else {}

if not os.path.exists(args.check_path):
    print('Making checkpath...')
    os.makedirs(args.check_path)

opt_kwargs = {'lr': args.lr, 'lr_decay': args.lr_decay, 'weight_decay': args.weight_decay, 'dampening': args.dampening,
              'momentum': args.momentum, 'nesterov': args.nesterov}

l2_dist = nn.CosineSimilarity(dim=1, eps=1e-12) if args.cos_sim else nn.PairwiseDistance(p=2)

transform = transforms.Compose([
    totensor()
])

if args.test_input == 'var':
    transform_V = transforms.Compose([
        ConcateOrgInput(remove_vad=args.remove_vad, feat_type=args.feat_format),
    ])
elif args.test_input == 'fix':
    transform_V = transforms.Compose([
        ConcateVarInput(remove_vad=args.remove_vad, num_frames=args.chunk_size, frame_shift=args.chunk_size,
                        feat_type=args.feat_format),
    ])

if args.log_scale:
    transform.transforms.append(tolog())
    transform_V.transforms.append(tolog())

# pdb.set_trace()
if args.feat_format in ['kaldi', 'wav']:
    # file_loader = read_mat
    file_loader = load_mat
elif args.feat_format == 'npy':
    file_loader = np.load

# torch.multiprocessing.set_sharing_strategy('file_system')

train_dir = EgsDataset(dir=args.train_dir, feat_dim=args.input_dim, loader=file_loader, transform=transform,
                       batch_size=args.batch_size, random_chunk=args.random_chunk, shuffle=args.batch_shuffle)

train_extract_dir = KaldiExtractDataset(dir=args.train_test_dir,
                                        transform=transform_V,
                                        filer_loader=file_loader,
                                        trials_file=args.train_trials)

extract_dir = KaldiExtractDataset(dir=args.test_dir, transform=transform_V,
                                  trials_file=args.trials, filer_loader=file_loader)

valid_dir = EgsDataset(dir=args.valid_dir, feat_dim=args.input_dim, loader=file_loader, transform=transform)


def train(train_loader, model, ce, optimizer, epoch, scheduler):
    # switch to evaluate mode
    model.train()

    correct = {'width%s' % i: 0. for i in FLAGS.width_mult_list}
    total_datasize = {'width%s' % i: 0. for i in FLAGS.width_mult_list}
    total_loss = {'width%s' % i: 0. for i in FLAGS.width_mult_list}
    orth_err = {'width%s' % i: 0. for i in FLAGS.width_mult_list}
    other_loss = {'width%s' % i: 0. for i in FLAGS.width_mult_list}

    ce_criterion, xe_criterion = ce
    pbar = tqdm(enumerate(train_loader))  # , total=len(train_loader), ncols=300)
    output_softmax = nn.Softmax(dim=1)
    lambda_ = (epoch / args.epochs) ** 2

    # start_time = time.time()
    # pdb.set_trace()
    # global FLAGS
    for batch_idx, (data, label) in pbar:
        if args.cuda:
            # label = label.cuda(non_blocking=True)
            # data = data.cuda(non_blocking=True)
            label = label.cuda()
            data = data.cuda()

        data, label = Variable(data), Variable(label)
        # pdb.set_trace()
        batch_accs = []
        for width_mult in FLAGS.width_mult_list:
            # FLAGS.width_mult = width_mult
            model.apply(lambda m: setattr(m, 'width_mult', width_mult))

            classfier, feats = model(data)
            classfier_label = classfier.clone()
            # print('max logit is ', classfier_label.max())

            if args.loss_type == 'soft':
                loss = ce_criterion(classfier, label)
            elif args.loss_type == 'asoft':
                classfier_label, _ = classfier
                loss = xe_criterion(classfier, label)
            elif args.loss_type in ['center', 'mulcenter', 'gaussian', 'coscenter', 'variance']:
                loss_cent = ce_criterion(classfier, label)
                loss_xent = args.loss_ratio * xe_criterion(feats, label)
                other_loss += loss_xent

                loss = loss_xent + loss_cent
            elif args.loss_type == 'ring':
                loss_cent = ce_criterion(classfier, label)
                loss_xent = args.loss_ratio * xe_criterion(feats)

                other_loss += loss_xent
                loss = loss_xent + loss_cent
            elif args.loss_type in ['amsoft', 'arcsoft', 'minarcsoft', 'minarcsoft2', 'subarc', 'aDCF']:
                loss = xe_criterion(classfier, label)
            elif 'arcdist' in args.loss_type:
                # pdb.set_trace()
                loss_cent = args.loss_ratio * ce_criterion(classfier, label)
                if args.loss_lambda:
                    loss_cent = loss_cent * lambda_

                loss_xent = xe_criterion(classfier, label)

                other_loss += loss_cent
                loss = loss_xent + loss_cent

            if args.loss_type in ['subarc']:
                classfier_label = classfier_label.max(dim=-1)[0]

            predicted_labels = output_softmax(classfier_label)
            predicted_one_labels = torch.max(predicted_labels, dim=1)[1]

            if args.lncl:
                if args.loss_type in ['amsoft', 'arcsoft', 'minarcsoft', 'minarcsoft2', 'subarc', 'arcdist', 'aDCF']:
                    predict_loss = xe_criterion(classfier, predicted_one_labels)
                else:
                    predict_loss = ce_criterion(classfier, predicted_one_labels)

                alpha_t = np.clip(args.alpha_t * (epoch / args.epochs) ** 2, a_min=0, a_max=1)
                mp = predicted_labels.mean(dim=0) * predicted_labels.shape[1]

                loss = (1 - alpha_t) * loss + alpha_t * predict_loss + args.beta * torch.mean(-torch.log(mp))

            minibatch_correct = float((predicted_one_labels.cpu() == label.cpu()).sum().item())
            minibatch_acc = minibatch_correct / len(predicted_one_labels)
            batch_accs.append(minibatch_acc)

            correct['width%s' % width_mult] += minibatch_correct
            total_datasize['width%s' % width_mult] += len(predicted_one_labels)
            total_loss['width%s' % width_mult] += float(loss.item())

            wid_str = '' if width_mult == 1.0 else '_%s' % width_mult
            writer.add_scalar('Train/Loss' + wid_str, float(loss.item()),
                              int((epoch - 1) * len(train_loader) + batch_idx + 1))

            if np.isnan(loss.item()):
                raise ValueError('Loss value is NaN!')

            # compute gradient and update weights
            loss.backward()

            if args.grad_clip > 0:
                this_lr = args.lr
                for param_group in optimizer.param_groups:
                    this_lr = min(param_group['lr'], this_lr)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                if args.model == 'FTDNN' and ((batch_idx + 1) % 4) == 0:
                    if isinstance(model, DistributedDataParallel):
                        model.module.step_ftdnn_layers()  # The key method to constrain the first two convolutions, perform after every SGD step
                        orth_err['width%s' % width_mult] += model.module.get_orth_errors()
                    else:
                        model.step_ftdnn_layers()  # The key method to constrain the first two convolutions, perform after every SGD step
                        orth_err['width%s' % width_mult] += model.get_orth_errors()

            # optimizer.zero_grad()
            # loss.backward()

            if args.loss_ratio != 0:
                if args.loss_type in ['center', 'mulcenter', 'gaussian', 'coscenter']:
                    for param in xe_criterion.parameters():
                        param.grad.data *= (1. / args.loss_ratio)

        # optimizer.step()
        if (batch_idx + 1) % args.log_interval == 0:
            # if (batch_idx + 1) % 500 == 0:
            #     break
            epoch_str = 'Train Epoch {}: [ {:>5.1f}% ]'.format(epoch, 100. * batch_idx / len(train_loader))

            # epoch_str += ' Width: {:.2f}'.format(width_mult)
            if len(args.random_chunk) == 2 and args.random_chunk[0] <= args.random_chunk[1]:
                epoch_str += ' Batch Len: {:>3d} '.format(data.shape[-2])

            epoch_str += ' Accuracy(%): '

            for minibatch_acc in batch_accs:
                epoch_str += '{:>6.2f} '.format(100. * minibatch_acc)

            # if orth_err['width%s' % width_mult] > 0:
            #     epoch_str += ' Orth_err: {:>5d}'.format(int(orth_err['width%s' % width_mult]))
            #
            # if args.loss_type in ['center', 'variance', 'mulcenter', 'gaussian', 'coscenter']:
            #     epoch_str += ' Center Loss: {:.4f}'.format(loss_xent.float())
            #
            # if 'arcdist' in args.loss_type:
            #     epoch_str += ' Dist Loss: {:.4f}'.format(loss_cent.float())

            epoch_str += '   Loss: '
            for width_mult in FLAGS.width_mult_list:
                epoch_str += '{:>7.4f} '.format(total_loss['width%s' % width_mult] / (batch_idx + 1))

            pbar.set_description(epoch_str)

        if ((batch_idx + 1) % args.accu_steps) == 0:
            # optimizer the net
            optimizer.step()  # update parameters of net
            optimizer.zero_grad()  # reset gradient

            # if (batch_idx + 1) == 200:
            #     break

        if args.scheduler == 'cyclic':
            scheduler.step()

    if args.batch_shuffle:
        train_dir.__shuffle__()

    this_epoch_str = 'Epoch {:>2d}: \33[91m'.format(epoch)
    # if len(FLAGS.width_mult_list) > 1:
    #     this_epoch_str += ' Width: ' + ' '.join([str(w) for w in FLAGS.width_mult_list]) + ' '
    # if len(FLAGS.width_mult_list) > 1:
    this_epoch_str += 'Train '

    acc_str = 'Accuracy(%): '
    loss_str = '   Loss:      '
    other_str = '{} Loss: '.format(args.loss_type)
    add_other = False
    for width_mult in FLAGS.width_mult_list:  # .25%:
        acc_str += '{:>6.2f} '.format(
            100 * float(correct['width%s' % width_mult]) / total_datasize['width%s' % width_mult])
        loss_str += '{:>7.4f} '.format(total_loss['width%s' % width_mult] / len(train_loader))

        if other_loss['width%s' % width_mult] != 0:
            add_other = True
            other_str += '{:2.4f}'.format(other_loss['width%s' % width_mult] / len(train_loader))

    this_epoch_str += acc_str + loss_str
    if add_other:
        this_epoch_str += other_str

    this_epoch_str += '.\33[0m'
    print(this_epoch_str)
    for width_mult in FLAGS.width_mult_list:
        wid_str = '' if width_mult == 1.0 else '_%s' % width_mult
        writer.add_scalar('Train/Accuracy' + wid_str,
                          correct['width%s' % width_mult] / total_datasize['width%s' % width_mult], epoch)
        writer.add_scalar('Train/Loss' + wid_str, total_loss['width%s' % width_mult] / len(train_loader), epoch)

    torch.cuda.empty_cache()


def valid_class(valid_loader, model, ce, epoch):
    # switch to evaluate mode
    model.eval()

    total_loss = {'width%s' % i: 0. for i in FLAGS.width_mult_list}
    other_loss = {'width%s' % i: 0. for i in FLAGS.width_mult_list}
    correct = {'width%s' % i: 0. for i in FLAGS.width_mult_list}
    total_datasize = {'width%s' % i: 0. for i in FLAGS.width_mult_list}

    ce_criterion, xe_criterion = ce
    softmax = nn.Softmax(dim=1)

    lambda_ = (epoch / args.epochs) ** 2
    # global FLAGS
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(valid_loader):
            data = data.cuda()
            label = label.cuda()

            # compute output
            # pdb.set_trace()
            for width_mult in FLAGS.width_mult_list:
                # FLAGS.width_mult = width_mult
                model.apply(lambda m: setattr(m, 'width_mult', width_mult))

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
                elif args.loss_type in ['variance', 'center', 'mulcenter', 'gaussian', 'coscenter']:
                    loss_cent = ce_criterion(classfier, label)
                    loss_xent = args.loss_ratio * xe_criterion(feats, label)
                    other_loss += float(loss_xent.item())

                    loss = loss_xent + loss_cent
                elif args.loss_type in ['amsoft', 'arcsoft', 'minarcsoft', 'minarcsoft2', 'subarc', 'aDCF']:
                    loss = xe_criterion(classfier, label)
                elif 'arcdist' in args.loss_type:
                    loss_cent = args.loss_ratio * ce_criterion(classfier, label)
                    if args.loss_lambda:
                        loss_cent = loss_cent * lambda_

                    loss_xent = xe_criterion(classfier, label)

                    other_loss += float(loss_cent.item())
                    loss = loss_xent + loss_cent

                total_loss['width%s' % width_mult] += float(loss.item())
                # pdb.set_trace()
                if args.loss_type in ['subarc']:
                    predicted_labels = predicted_labels.max(dim=-1)[0]
                predicted_one_labels = softmax(predicted_labels)
                predicted_one_labels = torch.max(predicted_one_labels, dim=1)[1]

                batch_correct = (predicted_one_labels.cuda() == label).sum().item()
                correct['width%s' % width_mult] += batch_correct
                total_datasize['width%s' % width_mult] += len(predicted_one_labels)

    for width_mult in FLAGS.width_mult_list:
        wid_str = '' if width_mult == 1.0 else '_%s' % width_mult

        valid_loss = total_loss['width%s' % width_mult] / len(valid_loader)
        valid_accuracy = 100. * correct['width%s' % width_mult] / total_datasize['width%s' % width_mult]
        writer.add_scalar('Train/Valid_Loss' + wid_str, valid_loss, epoch)
        writer.add_scalar('Train/Valid_Accuracy' + wid_str, valid_accuracy, epoch)

    torch.cuda.empty_cache()

    this_epoch_str = '          \33[91mValid '
    acc_str = 'Accuracy(%): '
    loss_str = '   Loss:      '
    other_str = '{} Loss: '.format(args.loss_type)
    add_other = False
    for width_mult in FLAGS.width_mult_list:  # .25%:
        acc_str += '{:>6.2f} '.format(
            100 * float(correct['width%s' % width_mult]) / total_datasize['width%s' % width_mult])
        loss_str += '{:>7.4f} '.format(total_loss['width%s' % width_mult] / len(valid_loader))

        if other_loss['width%s' % width_mult] != 0:
            add_other = True
            other_str += '{:2.4f}'.format(other_loss['width%s' % width_mult] / len(valid_loader))

    this_epoch_str += acc_str + loss_str
    if add_other:
        this_epoch_str += other_str

    this_epoch_str += '.\33[0m'
    print(this_epoch_str)

    return valid_loss


def valid_test(train_extract_loader, model, epoch, xvector_dir):
    # switch to evaluate mode
    model.eval()

    eer_dict = {}
    eer_threshold_dict = {}
    mindcf_01_dict = {}
    mindcf_001_dict = {}
    mix_dict = {}

    test_str = '          \33[91mTest '
    eer_str = 'EER(%):       '
    threshold_str = '   Threshold: '
    mindcf_01_str = ' MinDcf-0.01: '
    mindcf_001_str = ' MinDcf-0.001: '
    mix_str = ' Mix3: '
    # global FLAGS

    for width_mult in FLAGS.width_mult_list:
        model.apply(lambda m: setattr(m, 'width_mult', width_mult))
        # FLAGS.width_mult = width_mult

        this_xvector_dir = "%s/train/epoch_%s_width%s" % (xvector_dir, epoch, width_mult)
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
        mix3 = 100. * eer * mindcf_01 * mindcf_001

        eer_dict['width%s' % width_mult] = eer
        eer_threshold_dict['width%s' % width_mult] = eer_threshold
        mindcf_01_dict['width%s' % width_mult] = mindcf_01
        mindcf_001_dict['width%s' % width_mult] = mindcf_001
        mix_dict['width%s' % width_mult] = mix3

        eer_str += '{:>6.2f} '.format(100. * eer)
        threshold_str += '{:>7.4f} '.format(eer_threshold)
        mindcf_01_str += '{:.4f} '.format(mindcf_01)
        mindcf_001_str += '{:.4f} '.format(mindcf_001)
        mix_str += '{:.4f} '.format(mix3)

        wid_str = '' if width_mult == 1.0 else '_%s' % width_mult
        writer.add_scalar('Train/EER' + wid_str, 100. * eer, epoch)
        writer.add_scalar('Train/Threshold' + wid_str, eer_threshold, epoch)
        writer.add_scalar('Train/mindcf-0.01' + wid_str, mindcf_01, epoch)
        writer.add_scalar('Train/mindcf-0.001' + wid_str, mindcf_001, epoch)
        writer.add_scalar('Train/mix3' + wid_str, mix3, epoch)

    test_str += eer_str + threshold_str + mindcf_01_str + mindcf_001_str + mix_str + '. \33[0m'
    print(test_str)
    torch.cuda.empty_cache()

    eer = np.min([eer_dict[i] for i in eer_dict])
    eer_threshold = np.max([eer_threshold_dict[i] for i in eer_threshold_dict])
    mindcf_01 = np.min([mindcf_01_dict[i] for i in mindcf_01_dict])
    mindcf_001 = np.min([mindcf_001_dict[i] for i in mindcf_001_dict])
    mix3 = np.min([mix_dict[i] for i in mix_dict])

    return {'EER': 100. * eer, 'Threshold': eer_threshold, 'MinDCF_01': mindcf_01,
            'MinDCF_001': mindcf_001, 'mix3': mix3}


def test(model, epoch, writer, xvector_dir):
    this_xvector_dir = "%s/test/epoch_%s" % (xvector_dir, epoch)

    extract_loader = torch.utils.data.DataLoader(extract_dir, batch_size=1, shuffle=False, **extract_kwargs)
    verification_extract(extract_loader, model, this_xvector_dir, epoch, test_input=args.test_input)

    verify_dir = ScriptVerifyDataset(dir=args.test_dir, trials_file=args.trials, xvectors_dir=this_xvector_dir,
                                     loader=read_vec_flt)
    verify_loader = torch.utils.data.DataLoader(verify_dir, batch_size=128, shuffle=False, **extract_kwargs)

    # pdb.set_trace()
    eer, eer_threshold, mindcf_01, mindcf_001 = verification_test(test_loader=verify_loader,
                                                                  dist_type=('cos' if args.cos_sim else 'l2'),
                                                                  log_interval=args.log_interval,
                                                                  xvector_dir=this_xvector_dir,
                                                                  epoch=epoch)
    print(
        '          \33[91mTest  EER: {:.4f}%, Threshold: {:.4f}, mindcf-0.01: {:.4f}, mindcf-0.001: {:.4f}.\33[0m\n'.format(
            100. * eer, eer_threshold, mindcf_01, mindcf_001))

    writer.add_scalar('Test/EER', 100. * eer, epoch)
    writer.add_scalar('Test/Threshold', eer_threshold, epoch)
    writer.add_scalar('Test/mindcf-0.01', mindcf_01, epoch)
    writer.add_scalar('Test/mindcf-0.001', mindcf_001, epoch)


def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive
    # test_display_triplet_distance = False
    # print the experiment configuration
    print('\nCurrent time is \33[91m{}\33[0m.'.format(str(time.asctime())))
    opts = vars(args)
    keys = list(opts.keys())
    keys.sort()
    options = ["\'%s\': \'%s\'" % (str(k), str(opts[k])) for k in keys]

    print('Parsed options: \n{ %s }' % (', '.join(options)))
    print('Number of Speakers: {}.\n'.format(train_dir.num_spks))

    # Simmable FLAGS
    global FLAGS
    if 'Slimmable' in args.model:
        width_mult_list = sorted([float(x) for x in args.width_mult_list.split(',')], reverse=True)
        FLAGS.width_mult_list = width_mult_list
        print('Slimmable width: ', width_mult_list)
    else:
        width_mult_list = [1]
        FLAGS.width_mult_list = width_mult_list

    # instantiate model and initialize weights
    model_kwargs = args_model(args, train_dir)

    keys = list(model_kwargs.keys())
    keys.sort()
    model_options = ["\'%s\': \'%s\'" % (str(k), str(model_kwargs[k])) for k in keys]
    print('Model options: \n{ %s }' % (', '.join(model_options)))
    print('Testing with %s distance, ' % ('cos' if args.cos_sim else 'l2'))

    model = create_model(args.model, **model_kwargs)
    model_yaml_path = os.path.join(args.check_path, 'model.%s.yaml' % time.strftime("%Y.%m.%d", time.localtime()))
    save_model_args(model_kwargs, model_yaml_path)

    start_epoch = 0
    if args.save_init and not args.finetune:
        check_path = '{}/checkpoint_{}_{}.pth'.format(args.check_path, start_epoch,
                                                      time.strftime('%Y_%b_%d_%H:%M', time.localtime()))
        if not os.path.exists(check_path):
            torch.save({'state_dict': model.state_dict()}, check_path)

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
        xe_criterion = AngleSoftmaxLoss(lambda_min=args.lambda_min, lambda_max=args.lambda_max)
    elif args.loss_type == 'center':
        xe_criterion = CenterLoss(num_classes=train_dir.num_spks, feat_dim=args.embedding_size)
    elif args.loss_type == 'variance':
        xe_criterion = VarianceLoss(num_classes=train_dir.num_spks, feat_dim=args.embedding_size)
    elif args.loss_type == 'gaussian':
        xe_criterion = GaussianLoss(num_classes=train_dir.num_spks, feat_dim=args.embedding_size)
    elif args.loss_type == 'coscenter':
        xe_criterion = CenterCosLoss(num_classes=train_dir.num_spks, feat_dim=args.embedding_size)
    elif args.loss_type == 'mulcenter':
        xe_criterion = MultiCenterLoss(num_classes=train_dir.num_spks, feat_dim=args.embedding_size,
                                       num_center=args.num_center)
    elif args.loss_type in ['amsoft', 'subam']:
        ce_criterion = None
        xe_criterion = AMSoftmaxLoss(margin=args.margin, s=args.s)
    elif args.loss_type in ['arcsoft']:
        ce_criterion = None
        if args.class_weight == 'cnc1':
            class_weight = torch.tensor(C.CNC1_WEIGHT)
            if len(class_weight) != train_dir.num_spks:
                class_weight = None
        else:
            class_weight = None
        xe_criterion = ArcSoftmaxLoss(margin=args.margin, s=args.s, iteraion=iteration,
                                      all_iteraion=args.all_iteraion, smooth_ratio=args.smooth_ratio,
                                      class_weight=class_weight)
    elif args.loss_type in ['subarc']:
        ce_criterion = None
        if args.class_weight == 'cnc1':
            class_weight = torch.tensor(C.CNC1_WEIGHT)
            if len(class_weight) != train_dir.num_spks:
                class_weight = None
        else:
            class_weight = None
        xe_criterion = SubArcSoftmaxLoss(margin=args.margin, s=args.s, iteraion=iteration,
                                         all_iteraion=args.all_iteraion, smooth_ratio=args.smooth_ratio,
                                         class_weight=class_weight)

    elif args.loss_type in ['aDCF']:
        ce_criterion = None
        xe_criterion = aDCFLoss(alpha=args.s, beta=(1 - args.smooth_ratio),
                                gamma=args.smooth_ratio, omega=args.margin)

    elif args.loss_type == 'minarcsoft':
        ce_criterion = None
        xe_criterion = MinArcSoftmaxLoss(margin=args.margin, s=args.s, iteraion=iteration,
                                         all_iteraion=args.all_iteraion)
    elif args.loss_type == 'minarcsoft2':
        ce_criterion = None
        xe_criterion = MinArcSoftmaxLoss_v2(margin=args.margin, s=args.s, iteraion=iteration,
                                            all_iteraion=args.all_iteraion)
    elif args.loss_type == 'wasse':
        xe_criterion = Wasserstein_Loss(source_cls=args.source_cls)
    elif args.loss_type == 'mmd':
        xe_criterion = MMD_Loss()
    elif args.loss_type == 'ring':
        xe_criterion = RingLoss(ring=args.ring)
        args.alpha = 0.0
    elif 'arcdist' in args.loss_type:
        ce_criterion = DistributeLoss(stat_type=args.stat_type, margin=args.m)
        xe_criterion = ArcSoftmaxLoss(margin=args.margin, s=args.s, iteraion=iteration, all_iteraion=args.all_iteraion)

    model_para = [{'params': model.parameters()}]
    if args.loss_type in ['center', 'variance', 'mulcenter', 'gaussian', 'coscenter', 'ring']:
        assert args.lr_ratio > 0
        model_para.append({'params': xe_criterion.parameters(), 'lr': args.lr * args.lr_ratio})

    if args.finetune or args.second_wd > 0:
        # if args.loss_type in ['asoft', 'amsoft']:
        classifier_params = list(map(id, model.classifier.parameters()))
        rest_params = filter(lambda p: id(p) not in classifier_params, model.parameters())
        init_lr = args.lr * args.lr_ratio if args.lr_ratio > 0 else args.lr
        init_wd = args.second_wd if args.second_wd > 0 else args.weight_decay
        print('Set the lr and weight_decay of classifier to %f and %f' % (init_lr, init_wd))
        model_para = [{'params': rest_params},
                      {'params': model.classifier.parameters(), 'lr': init_lr, 'weight_decay': init_wd}]

    if hasattr(model, 'filter_layer') and model.filter_layer != None:
        # args.filter in ['fDLR', 'fBLayer', 'fLLayer', 'fBPLayer', 'sinc2down', 'wav2down']:
        filter_params = list(map(id, model.filter_layer.parameters()))
        rest_params = filter(lambda p: id(p) not in filter_params, model_para[0]['params'])
        init_wd = args.filter_wd if args.filter_wd > 0 else args.weight_decay
        init_lr = args.lr * args.lr_ratio if args.lr_ratio > 0 else args.lr
        print('Set the lr and weight_decay of filter layer to %f and %f' % (init_lr, init_wd))
        model_para[0]['params'] = rest_params
        model_para.append({'params': model.filter_layer.parameters(), 'lr': init_lr,
                           'weight_decay': init_wd})

    optimizer = create_optimizer(model_para, args.optimizer, **opt_kwargs)
    early_stopping_scheduler = EarlyStopping(patience=args.early_patience,
                                             min_delta=args.early_delta)

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
        gamma = np.power(args.base_lr / args.lr, 1 / args.epochs) if args.gamma == 0 else args.gamma
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif args.scheduler == 'rop':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, min_lr=1e-5)
    elif args.scheduler == 'cyclic':
        cycle_momentum = False if args.optimizer == 'adam' else True
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=args.base_lr, max_lr=args.lr,
                                          step_size_up=args.cyclic_epoch * int(
                                              np.ceil(len(train_dir) / args.batch_size)),
                                          cycle_momentum=cycle_momentum,
                                          mode='triangular2')
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    ce = [ce_criterion, xe_criterion]

    start = args.start_epoch + start_epoch
    print('Start epoch is : ' + str(start))
    # start = 0
    end = start + args.epochs

    if len(args.random_chunk) == 2 and args.random_chunk[0] <= args.random_chunk[1]:
        min_chunk_size = int(args.random_chunk[0])
        max_chunk_size = int(args.random_chunk[1])
        pad_dim = 2 if args.feat_format == 'kaldi' else 3

        train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size,
                                                   collate_fn=PadCollate(dim=pad_dim,
                                                                         num_batch=int(
                                                                             np.ceil(len(train_dir) / args.batch_size)),
                                                                         min_chunk_size=min_chunk_size,
                                                                         max_chunk_size=max_chunk_size,
                                                                         chisquare=args.chisquare),
                                                   shuffle=args.shuffle, **kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_dir, batch_size=int(args.batch_size / 2),
                                                   collate_fn=PadCollate(dim=pad_dim, fix_len=True,
                                                                         min_chunk_size=min_chunk_size,
                                                                         max_chunk_size=max_chunk_size),
                                                   shuffle=False, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=args.shuffle,
                                                   **kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_dir, batch_size=int(args.batch_size / 2), shuffle=False,
                                                   **kwargs)
    train_extract_loader = torch.utils.data.DataLoader(train_extract_dir, batch_size=1, shuffle=False,
                                                       **extract_kwargs)

    if args.cuda:
        if len(args.gpu_id) > 1:
            print("Continue with gpu: %s ..." % str(args.gpu_id))
            # torch.distributed.init_process_group(backend="nccl",
            #                                      init_method='file:///home/yangwenhao/lstm_speaker_verification/data/sharedfile',
            #                                      rank=0,
            #                                      world_size=1)
            try:
                torch.distributed.init_process_group(backend="nccl", init_method='tcp://localhost:32457', rank=0,
                                                     world_size=1)
            except RuntimeError as r:
                torch.distributed.init_process_group(backend="nccl", init_method='tcp://localhost:32458', rank=0,
                                                     world_size=1)
            # if args.gain
            if 'Slimmable' in args.model:
                model = DistributedDataParallel(model.cuda(), find_unused_parameters=True)
            else:
                model = DistributedDataParallel(model.cuda())

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

    all_lr = []
    valid_test_result = []
    try:
        for epoch in range(start, end):
            # pdb.set_trace()
            lr_string = '\n\33[1;34m Current \'{}\' learning rate is '.format(args.optimizer)
            this_lr = []
            for param_group in optimizer.param_groups:
                this_lr.append(param_group['lr'])
                lr_string += '{:.10f} '.format(param_group['lr'])
            print('%s \33[0m' % lr_string)
            all_lr.append(this_lr[0])
            writer.add_scalar('Train/lr', this_lr[0], epoch)

            train(train_loader, model, ce, optimizer, epoch, scheduler)
            valid_loss = valid_class(valid_loader, model, ce, epoch)
            if args.early_stopping or (epoch % args.test_interval == 1 or epoch in milestones or epoch == (
                    end - 1)):
                valid_test_dict = valid_test(train_extract_loader, model, epoch, xvector_dir)
            else:
                valid_test_dict = {}

            valid_test_dict['Valid_Loss'] = valid_loss
            valid_test_result.append(valid_test_dict)

            if args.early_stopping:
                early_stopping_scheduler(valid_test_dict[args.early_meta], epoch)
                if early_stopping_scheduler.best_epoch + early_stopping_scheduler.patience >= end:
                    early_stopping_scheduler.early_stop = True

                if args.scheduler != 'cyclic' and this_lr[0] <= 0.1 ** 3 * args.lr:
                    if len(all_lr) > 10 and all_lr[-10] == this_lr[0]:
                        early_stopping_scheduler.early_stop = True

            if epoch % args.test_interval == 1 or epoch in milestones or epoch == (
                    end - 1) or early_stopping_scheduler.best_epoch == epoch:
                model.eval()
                check_path = '{}/checkpoint_{}.pth'.format(args.check_path, epoch)
                model_state_dict = model.module.state_dict() \
                    if isinstance(model, DistributedDataParallel) else model.state_dict()
                torch.save({'epoch': epoch,
                            'state_dict': model_state_dict,
                            'criterion': ce}, check_path)

                if args.early_stopping:
                    pass
                elif epoch % args.test_interval == 1:
                    test(model, epoch, writer, xvector_dir)

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
                    shutil.copy('{}/checkpoint_{}.pth'.format(args.check_path, early_stopping_scheduler.best_epoch),
                                '{}/best.pth'.format(args.check_path))
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
    print("Running %.4f minutes for each epoch.\n" % (t / 60 / (max(end - start, 1))))
    exit(0)


if __name__ == '__main__':
    main()