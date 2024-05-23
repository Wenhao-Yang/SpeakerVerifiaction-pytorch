#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: train_egs_hard.py
@Time: 2022/5/7 16:52
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

from Define_Model.Loss.LossFunction import CenterLoss, Wasserstein_Loss, MultiCenterLoss, CenterCosLoss, RingLoss, \
    VarianceLoss, DistributeLoss, pAUCLoss, aDCFLoss
from Define_Model.Loss.SoftmaxLoss import AngleSoftmaxLoss, AngleLinear, AdditiveMarginLinear, AMSoftmaxLoss, \
    ArcSoftmaxLoss, \
    GaussianLoss, MinArcSoftmaxLoss, MinArcSoftmaxLoss_v2, DAMSoftmaxLoss
from Eval.eval_metrics import evaluate_kaldi_eer
from Process_Data.Datasets.KaldiDataset import KaldiExtractDataset, \
    ScriptVerifyDataset, PairTrainDataset
from Process_Data.Datasets.LmdbDataset import EgsDataset
import Process_Data.constants as C
from Process_Data.audio_processing import ConcateVarInput, tolog, ConcateOrgInput, PadCollate, PadCollate3d, \
    ConcateInput
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
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

# create logger
# Define visulaize SummaryWriter instance
writer = SummaryWriter(logdir=args.check_path, filename_suffix='_first')
sys.stdout = NewLogger(osp.join(args.check_path, 'log.%s.txt' % time.strftime("%Y.%m.%d", time.localtime())))

kwargs = {'num_workers': args.nj, 'pin_memory': False} if args.cuda else {}
extract_kwargs = {'num_workers': 0,
                  'pin_memory': False} if args.cuda else {}

if not os.path.exists(args.check_path):
    print('Making checkpath...')
    os.makedirs(args.check_path)

opt_kwargs = {'lr': args.lr, 'lr_decay': args.lr_decay, 'weight_decay': args.weight_decay, 'dampening': args.dampening,
              'momentum': args.momentum}

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

transform_H = transforms.Compose([
    ConcateInput(remove_vad=args.remove_vad, num_frames=args.chunk_size),
    totensor()
])

if args.log_scale:
    transform.transforms.append(tolog())
    transform_V.transforms.append(tolog())

# pdb.set_trace()
if args.feat_format in ['kaldi', 'wav']:
    file_loader = read_mat
elif args.feat_format == 'npy':
    file_loader = np.load

torch.multiprocessing.set_sharing_strategy('file_system')

train_dir = EgsDataset(dir=args.train_dir, feat_dim=args.input_dim, loader=file_loader, transform=transform,
                       batch_size=args.batch_size, random_chunk=args.random_chunk)

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

    correct = 0.
    total_datasize = 0.
    total_loss = 0.
    orth_err = 0
    other_loss = 0.

    ce_criterion, xe_criterion = ce
    pbar = tqdm(enumerate(train_loader))
    output_softmax = nn.Softmax(dim=1)
    lambda_ = (epoch / args.epochs) ** 2

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
        # cos_theta, phi_theta = classfier
        classfier_label = classfier
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
        elif args.loss_type in ['amsoft', 'damsoft', 'arcsoft', 'minarcsoft', 'minarcsoft2', 'subarc', 'subam']:
            loss = xe_criterion(classfier, label)
        elif args.loss_type == 'arcdist':
            # pdb.set_trace()
            loss_cent = args.loss_ratio * ce_criterion(classfier, label)
            if args.loss_lambda:
                loss_cent = loss_cent * lambda_

            loss_xent = xe_criterion(classfier, label)

            other_loss += loss_cent
            loss = loss_xent + loss_cent

        predicted_labels = output_softmax(classfier_label)
        predicted_one_labels = torch.max(predicted_labels, dim=1)[1]

        if args.lncl:
            if args.loss_type in ['amsoft', 'damsoft', 'arcsoft', 'minarcsoft', 'minarcsoft2', 'subarc', 'arcdist']:
                predict_loss = xe_criterion(classfier, predicted_one_labels)
            else:
                predict_loss = ce_criterion(classfier, predicted_one_labels)

            alpha_t = np.clip(args.alpha_t * (epoch / args.epochs) ** 2, a_min=0, a_max=1)
            mp = predicted_labels.mean(dim=0) * predicted_labels.shape[1]

            loss = (1 - alpha_t) * loss + alpha_t * predict_loss + args.beta * torch.mean(-torch.log(mp))

        minibatch_correct = float((predicted_one_labels.cpu() == label.cpu()).sum().item())
        minibatch_acc = minibatch_correct / len(predicted_one_labels)
        correct += minibatch_correct

        total_datasize += len(predicted_one_labels)
        total_loss += float(loss.item())
        writer.add_scalar('Train/All_Loss', float(loss.item()), int((epoch - 1) * len(train_loader) + batch_idx + 1))

        if np.isnan(loss.item()):
            pdb.set_trace()
            raise ValueError('Loss value is NaN!')

        # compute gradient and update weights
        loss.backward()

        if args.grad_clip > 0:
            this_lr = args.lr
            for param_group in optimizer.param_groups:
                this_lr = min(param_group['lr'], this_lr)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if ((batch_idx + 1) % args.accu_steps) == 0:
            # optimizer the net
            optimizer.step()  # update parameters of net
            optimizer.zero_grad()  # reset gradient

            if args.model == 'FTDNN' and ((batch_idx + 1) % 4) == 0:
                if isinstance(model, DistributedDataParallel):
                    model.module.step_ftdnn_layers()  # The key method to constrain the first two convolutions, perform after every SGD step
                    orth_err += model.module.get_orth_errors()
                else:
                    model.step_ftdnn_layers()  # The key method to constrain the first two convolutions, perform after every SGD step
                    orth_err += model.get_orth_errors()

        # optimizer.zero_grad()
        # loss.backward()
        if args.loss_ratio != 0:
            if args.loss_type in ['center', 'mulcenter', 'gaussian', 'coscenter']:
                for param in xe_criterion.parameters():
                    param.grad.data *= (1. / args.loss_ratio)

        # optimizer.step()
        if args.scheduler == 'cyclic':
            scheduler.step()

        if (batch_idx + 1) % args.log_interval == 0:
            epoch_str = 'Train Epoch {}: [{:8d}/{:8d} ({:3.0f}%)]'.format(epoch, batch_idx * len(data),
                                                                          len(train_loader.dataset),
                                                                          100. * batch_idx / len(train_loader))

            if len(args.random_chunk) == 2 and args.random_chunk[0] <= args.random_chunk[1]:
                epoch_str += ' Batch Len: {:>3d}'.format(data.shape[-2])

            if orth_err > 0:
                epoch_str += ' Orth_err: {:>5d}'.format(int(orth_err))

            if args.loss_type in ['center', 'variance', 'mulcenter', 'gaussian', 'coscenter']:
                epoch_str += ' Center Loss: {:.4f}'.format(loss_xent.float())
            if args.loss_type in ['arcdist']:
                epoch_str += ' Dist Loss: {:.4f}'.format(loss_cent.float())
            epoch_str += ' Avg Loss: {:.4f} Batch Accuracy: {:.4f}%'.format(total_loss / (batch_idx + 1),
                                                                            100. * minibatch_acc)
            pbar.set_description(epoch_str)

    this_epoch_str = 'Epoch {:>2d}: \33[91mTrain Accuracy: {:.6f}%, Avg loss: {:6f}'.format(epoch, 100 * float(
        correct) / total_datasize, total_loss / len(train_loader))

    if other_loss != 0:
        this_epoch_str += ' {} Loss: {:6f}'.format(args.loss_type, other_loss / len(train_loader))

    this_epoch_str += '.\33[0m'
    print(this_epoch_str)
    writer.add_scalar('Train/Accuracy', correct / total_datasize, epoch)
    writer.add_scalar('Train/Loss', total_loss / len(train_loader), epoch)

    torch.cuda.empty_cache()


def hard_train(hard_loader, model, ce, optimizer, epoch):
    # switch to evaluate mode
    model.train()

    correct = 0.
    error = 0.
    total_datasize = 0.
    total_loss = 0.
    orth_err = 0
    other_loss = 0.

    ce_criterion, xe_criterion = ce
    pbar = tqdm(enumerate(hard_loader))
    output_softmax = nn.Softmax(dim=1)
    lambda_ = (epoch / args.epochs) ** 2

    # start_time = time.time()
    # pdb.set_trace()
    for batch_idx, (enroll_data, eval_data, label, ws) in pbar:
        if args.cuda:
            # pdb.set_trace()
            # label = label.cuda(non_blocking=True)
            # data = data.cuda(non_blocking=True)
            label = label.cuda()

            enroll_len = enroll_data.shape[0]
            data = torch.cat([enroll_data, eval_data])
            data = data.cuda()

            spk_label = torch.cat(ws).cuda()

        data = Variable(data)
        label = Variable(label)
        spk_label = Variable(spk_label)

        classfier, feats = model(data)
        enroll_feats = feats[:enroll_len]
        eval_feats = feats[-enroll_len:]
        # eval_classfier, eval_feats = model(eval_data)

        scores = l2_dist(enroll_feats, eval_feats)

        target = scores[torch.where(label == 1)[0]]
        nontarget = scores[torch.where(label == 0)[0]]

        hard_loss = ce_criterion(target, nontarget)

        loss = hard_loss + args.loss_ratio * xe_criterion(classfier, spk_label)

        # cos_theta, phi_theta = classfier
        # classfier_label = classfier
        # print('max logit is ', classfier_label.max())

        eer, eer_threshold, accuracy = evaluate_kaldi_eer(scores.detach().cpu().numpy(), label.cpu().numpy(),
                                                          cos=True if args.cos_sim else False,
                                                          re_thre=True)

        predicted_labels = output_softmax(classfier)
        predicted_one_labels = torch.max(predicted_labels, dim=1)[1]
        minibatch_correct = float((predicted_one_labels.cpu() == spk_label.cpu()).sum().item())
        minibatch_acc = minibatch_correct / len(predicted_one_labels)
        correct += minibatch_correct

        minibatch_error = eer * len(enroll_data)
        minibatch_eer = eer
        error += minibatch_error

        total_datasize += len(enroll_data)
        total_loss += float(loss.item())
        writer.add_scalar('Train/All_Loss', float(loss.item()), int((epoch - 1) * len(hard_loader) + batch_idx + 1))

        if np.isnan(loss.item()):
            pdb.set_trace()
            raise ValueError('Loss value is NaN!')

        # compute gradient and update weights
        loss.backward()

        if args.grad_clip > 0:
            this_lr = args.lr
            for param_group in optimizer.param_groups:
                this_lr = min(param_group['lr'], this_lr)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if ((batch_idx + 1) % args.accu_steps) == 0:
            # optimizer the net
            optimizer.step()  # update parameters of net
            optimizer.zero_grad()  # reset gradient

            if args.model == 'FTDNN' and ((batch_idx + 1) % 4) == 0:
                if isinstance(model, DistributedDataParallel):
                    model.module.step_ftdnn_layers()  # The key method to constrain the first two convolutions, perform after every SGD step
                    orth_err += model.module.get_orth_errors()
                else:
                    model.step_ftdnn_layers()  # The key method to constrain the first two convolutions, perform after every SGD step
                    orth_err += model.get_orth_errors()

        # optimizer.zero_grad()
        # loss.backward()
        if args.loss_ratio != 0:
            if args.loss_type in ['center', 'mulcenter', 'gaussian', 'coscenter']:
                for param in xe_criterion.parameters():
                    param.grad.data *= (1. / args.loss_ratio)

        # optimizer.step()
        # if args.scheduler == 'cyclic':
        #     scheduler.step()

        if (batch_idx + 1) % args.log_interval == 0:
            epoch_str = 'Hard Mining Epoch {}: [{:8d} ({:3.0f}%)]'.format(epoch, len(hard_loader),
                                                                          100. * batch_idx / len(hard_loader))

            if len(args.random_chunk) == 2 and args.random_chunk[0] <= args.random_chunk[1]:
                epoch_str += ' Batch Len: {:>3d}'.format(enroll_data.shape[-2])

            if orth_err > 0:
                epoch_str += ' Orth_err: {:>5d}'.format(int(orth_err))

            # if args.loss_type in ['center', 'variance', 'mulcenter', 'gaussian', 'coscenter']:
            #
            # if args.loss_type in ['arcdist']:
            #     epoch_str += ' Dist Loss: {:.4f}'.format(loss_cent.float())
            # epoch_str += ' '.format()
            epoch_str += ' EER: {:.4f}%'.format(100. * minibatch_eer)
            epoch_str += ' Accuracy: {:.4f}% Loss Hard: {:.4f} Avg: {:.4f} '.format(100. * minibatch_acc,
                                                                                    float(hard_loss),
                                                                                    total_loss / (batch_idx + 1))

            pbar.set_description(epoch_str)

    this_epoch_str = 'Hard Mining Epoch {:>2d}: \33[91mAccuracy: {:.6f}%, EER: {:.6f}%, Avg Loss: {:6f}'.format(epoch,
                                                                                                                50 * float(
                                                                                                                    correct) / total_datasize,
                                                                                                                100 * float(
                                                                                                                    error) / total_datasize,
                                                                                                                total_loss / len(
                                                                                                                    hard_loader))

    if other_loss != 0:
        this_epoch_str += ' {} Loss: {:6f}'.format(args.loss_type, other_loss / len(hard_loader))

    this_epoch_str += '.\33[0m'
    print(this_epoch_str)
    writer.add_scalar('Train/EER', correct / total_datasize, epoch)
    writer.add_scalar('Train/Hard_Loss', total_loss / len(hard_loader), epoch)

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
    lambda_ = (epoch / args.epochs) ** 2
    # 2. / (1 + np.exp(-10. * epoch / args.epochs)) - 1.

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(valid_loader):
            data = data.cuda()
            label = label.cuda()

            # compute output
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
            elif args.loss_type in ['amsoft', 'damsoft', 'arcsoft', 'minarcsoft', 'minarcsoft2', 'subarc', 'subam',
                                    'subdam']:
                loss = xe_criterion(classfier, label)
            elif args.loss_type == 'arcdist':
                loss_cent = args.loss_ratio * ce_criterion(classfier, label)
                if args.loss_lambda:
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

    valid_loss = total_loss / len(valid_loader)
    valid_accuracy = 100. * correct / total_datasize
    writer.add_scalar('Train/Valid_Loss', valid_loss, epoch)
    writer.add_scalar('Train/Valid_Accuracy', valid_accuracy, epoch)
    torch.cuda.empty_cache()

    this_epoch_str = '          \33[91mValid Accuracy: {:.6f}%, Avg loss: {:.6f}'.format(valid_accuracy, valid_loss)

    if other_loss != 0:
        this_epoch_str += ' {} Loss: {:6f}'.format(args.loss_type, other_loss / len(valid_loader))
    this_epoch_str += '.\33[0m'
    print(this_epoch_str)

    return valid_loss


def valid_test(train_extract_loader, model, epoch, xvector_dir):
    # switch to evaluate mode
    model.eval()

    this_xvector_dir = "%s/train/epoch_%s" % (xvector_dir, epoch)
    verification_extract(train_extract_loader, model, this_xvector_dir, epoch, test_input=args.test_input,
                         verbose=2, )

    verify_dir = ScriptVerifyDataset(dir=args.train_test_dir, trials_file=args.train_trials,
                                     xvectors_dir=this_xvector_dir,
                                     loader=read_vec_flt)
    verify_loader = torch.utils.data.DataLoader(verify_dir, batch_size=256, shuffle=False, **kwargs)
    eer, eer_threshold, mindcf_01, mindcf_001, dists = verification_test(test_loader=verify_loader,
                                                                         dist_type=('cos' if args.cos_sim else 'l2'),
                                                                         log_interval=args.log_interval,
                                                                         xvector_dir=this_xvector_dir,
                                                                         epoch=epoch,
                                                                         return_dist=True,
                                                                         verbose=2)

    print('          \33[91mTrain EER: {:.4f}%, Threshold: {:.4f}, ' \
          'mindcf-0.01: {:.4f}, mindcf-0.001: {:.4f}. \33[0m'.format(100. * eer,
                                                                     eer_threshold,
                                                                     mindcf_01,
                                                                     mindcf_001))

    i = 0
    with open(os.path.join(args.train_test_dir, args.train_trials), 'r') as f1, \
            open(os.path.join(this_xvector_dir, 'miss_trials'), 'w') as f2:
        for l in f1.readlines():  # tqdm(f1.readlines()):
            enroll_uid, eval_uid, truth = l.split()
            s = float(dists[i])

            if (s >= eer_threshold * (1 + args.inter_ratio) and truth == 'True') or (
                    s <= eer_threshold * (1 - args.inter_ratio) and truth == 'False'):
                pass
            else:
                f2.write(enroll_uid + ' ' + eval_uid + ' ' + truth + '\n')

            i += 1

    writer.add_scalar('Train/EER', 100. * eer, epoch)
    writer.add_scalar('Train/Threshold', eer_threshold, epoch)
    writer.add_scalar('Train/mindcf-0.01', mindcf_01, epoch)
    writer.add_scalar('Train/mindcf-0.001', mindcf_001, epoch)

    torch.cuda.empty_cache()


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
    # exit(0)

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
    elif args.loss_type in ['damsoft', 'subdam']:
        ce_criterion = None
        xe_criterion = DAMSoftmaxLoss(margin=args.margin, s=args.s)
    elif args.loss_type in ['arcsoft', 'subarc']:
        # ce_criterion = aAUCLoss()
        ce_criterion = pAUCLoss()
        if args.class_weight == 'cnc1':
            class_weight = torch.tensor(C.CNC1_WEIGHT) * args.max_cls_weight + 1 - args.max_cls_weight
            if len(class_weight) != train_dir.num_spks:
                class_weight = None
        else:
            class_weight = None

        xe_criterion = ArcSoftmaxLoss(margin=args.margin, s=args.s, iteraion=iteration,
                                      all_iteraion=args.all_iteraion, smooth_ratio=args.smooth_ratio,
                                      class_weight=class_weight)
    elif args.loss_type in ['aDCF']:
        # ce_criterion = aAUCLoss()
        ce_criterion = pAUCLoss()
        xe_criterion = aDCFLoss(alpha=args.s, beta=args.margin, gamma=(1 - args.margin), omega=args.smooth_ratio)

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

    if args.filter in ['fDLR', 'fBLayer', 'fLLayer', 'fBPLayer']:
        filter_params = list(map(id, model.filter_layer.parameters()))
        rest_params = filter(lambda p: id(p) not in filter_params, model_para[0]['params'])
        init_wd = args.filter_wd if args.filter_wd > 0 else args.weight_decay
        init_lr = args.lr * args.lr_ratio if args.lr_ratio > 0 else args.lr
        print('Set the lr and weight_decay of filter layer to %f and %f' % (init_lr, init_wd))
        model_para[0]['params'] = rest_params
        model_para.append({'params': model.filter_layer.parameters(), 'lr': init_lr,
                           'weight_decay': init_wd})

    optimizer = create_optimizer(model_para, args.optimizer, **opt_kwargs)

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
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=args.base_lr,
                                          max_lr=args.lr,
                                          step_size_up=5 * int(np.ceil(len(train_dir) / args.batch_size)),
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

        if args.noise_padding_dir != '':
            noise_padding_dir = EgsDataset(dir=args.noise_padding_dir,
                                           feat_dim=args.input_dim, loader=kaldiio.load_mat,
                                           transform=transform, verbose=0)
        else:
            noise_padding_dir = None

        train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size,
                                                   collate_fn=PadCollate(dim=pad_dim,
                                                                         num_batch=int(
                                                                             np.ceil(len(train_dir) / args.batch_size)),
                                                                         min_chunk_size=min_chunk_size,
                                                                         max_chunk_size=max_chunk_size,
                                                                         chisquare=args.chisquare,
                                                                         noise_padding=noise_padding_dir),
                                                   shuffle=args.shuffle, **kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_dir, batch_size=int(args.batch_size / 2),
                                                   collate_fn=PadCollate(dim=pad_dim, fix_len=True,
                                                                         min_chunk_size=args.chunk_size,
                                                                         max_chunk_size=args.chunk_size + 1),
                                                   shuffle=False, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=args.shuffle,
                                                   **kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_dir, batch_size=int(args.batch_size / 2), shuffle=False,
                                                   **kwargs)
    train_extract_loader = torch.utils.data.DataLoader(train_extract_dir, batch_size=1, shuffle=False, **extract_kwargs)

    if args.cuda:
        if len(args.gpu_id) > 1:
            print("Continue with gpu: %s ..." % str(args.gpu_id))
            torch.distributed.init_process_group(backend="nccl",
                                                 init_method='file:///home/ssd2020/yangwenhao/lstm_speaker_verification/data/sharedfile',
                                                 rank=0,
                                                 world_size=1)
            # if args.gain
            model = DistributedDataParallel(model.cuda(), find_unused_parameters=True)
            # model = DistributedDataParallel(model.cuda())

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

    try:
        for epoch in range(start, end):
            # pdb.set_trace()
            lr_string = '\n\33[1;34m Current \'{}\' learning rate is '.format(args.optimizer)
            for param_group in optimizer.param_groups:
                lr_string += '{:.10f} '.format(param_group['lr'])
            print('%s \33[0m' % lr_string)

            # train(train_loader, model, ce, optimizer, epoch, scheduler)
            valid_loss = valid_class(valid_loader, model, ce, epoch)

            # if (epoch == 1 or epoch != (end - 2)) and (
            #         epoch % args.test_interval == 1 or epoch in milestones or epoch == (end - 1)):
            #     model.eval()
            check_path = '{}/checkpoint_{}.pth'.format(args.check_path, epoch)
            model_state_dict = model.module.state_dict() \
                if isinstance(model, DistributedDataParallel) else model.state_dict()
            torch.save({'epoch': epoch, 'state_dict': model_state_dict, 'criterion': ce}, check_path)

            valid_test(train_extract_loader, model, epoch, xvector_dir)
            miss_trials = "%s/train/epoch_%s/miss_trials" % (xvector_dir, epoch)

            if epoch in [1, start] or (epoch != (end - 2) and (
                    epoch % args.test_interval == 1 or epoch in milestones or epoch == (end - 1))):
                test(model, epoch, writer, xvector_dir)

            # hard mining
            hard_dir = PairTrainDataset(dir=args.train_test_dir, miss_trials=miss_trials,
                                        target_ratio=args.target_ratio,
                                        loader=file_loader, transform=transform_H, segment_len=args.chunk_size,
                                        verbose=0)

            hard_loader = torch.utils.data.DataLoader(hard_dir, batch_size=args.batch_size,
                                                      collate_fn=PadCollate3d(dim=pad_dim,
                                                                              num_batch=int(np.ceil(
                                                                                  len(hard_dir) / args.batch_size)),
                                                                              min_chunk_size=min_chunk_size,
                                                                              max_chunk_size=max_chunk_size),
                                                      shuffle=False, **kwargs)

            hard_train(hard_loader, model, ce, optimizer, epoch)

            # if epoch != (end - 1):
            #     try:
            #         shutil.rmtree("%s/train/epoch_%s" % (xvector_dir, epoch))
            #         shutil.rmtree("%s/test/epoch_%s" % (xvector_dir, epoch))
            #     except Exception as e:
            #         print('rm dir xvectors error:', e)

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