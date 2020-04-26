#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: test_accuracy.py
@Time: 19-8-6 下午1:29
@Overview: Train the resnet 10 with asoftmax.
"""
from __future__ import print_function

import argparse
import os
import sys
import time
# Version conflict
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from kaldi_io import read_mat
from torch.autograd import Variable
from tqdm import tqdm

from Define_Model.SoftmaxLoss import AngleLinear, AdditiveMarginLinear
from Define_Model.model import PairwiseDistance
from Process_Data.KaldiDataset import ScriptTrainDataset, ScriptTestDataset, ScriptValidDataset
from Process_Data.audio_processing import to2tensor, varLengthFeat
from Process_Data.audio_processing import toMFB, totensor, truncatedinput, read_audio
from TrainAndTest.common_func import create_model
from eval_metrics import evaluate_kaldi_eer, evaluate_kaldi_mindcf
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
parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
# Data options
parser.add_argument('--train-dir', type=str,
                    help='path to dataset')
parser.add_argument('--test-dir', type=str,
                    help='path to voxceleb1 test dataset')
parser.add_argument('--sitw-dir', type=str,
                    default='/home/yangwenhao/local/project/lstm_speaker_verification/data/sitw',
                    help='path to voxceleb1 test dataset')
parser.add_argument('--nj', default=12, type=int, metavar='NJOB', help='num of job')

# parser.add_argument('--check-path',
#                     help='folder to output model checkpoints')
parser.add_argument('--resume',
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=20, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--veri-pairs', type=int, default=12800, metavar='VP',
                    help='number of epochs to train (default: 10)')

# Training options
# Model options
# ALSTM  ASiResNet34  ExResNet34  LoResNet10  ResNet20  SiResNet34  SuResCNN10  TDNN
parser.add_argument('--model', type=str,
                    choices=['ASTDNN', 'TDNN', 'LoResNet10', 'ResNet20', 'SiResNet34', 'SuResCNN10'],
                    help='path to voxceleb1 test dataset')
parser.add_argument('--resnet-size', default=8, type=int,
                    metavar='RES', help='The channels of convs layers)')
parser.add_argument('--statis-pooling', action='store_true', default=False,
                    help='using Cosine similarity')
parser.add_argument('--channels', default='64,128,256', type=str,
                    metavar='CHA', help='The channels of convs layers)')
parser.add_argument('--feat-dim', default=161, type=int, metavar='FEAT',
                    help='acoustic feature dimension')
parser.add_argument('--kernel-size', default='5,5', type=str, metavar='KE',
                    help='kernel size of conv filters')
parser.add_argument('--cos-sim', action='store_true', default=True,
                    help='using Cosine similarity')
parser.add_argument('--embedding-size', type=int, default=1024, metavar='ES',
                    help='Dimensionality of the embedding')

parser.add_argument('--input-per-spks', type=int, default=224, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--num-valid', type=int, default=5, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--test-input-per-file', type=int, default=1, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='BST',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--dropout-p', type=float, default=0., metavar='BST',
                    help='input batch size for testing (default: 64)')

# loss configure
parser.add_argument('--loss-type', type=str, default='soft', choices=['soft', 'asoft', 'center', 'amsoft'],
                    help='path to voxceleb1 test dataset')
parser.add_argument('--loss-ratio', type=float, default=0.1, metavar='LOSSRATIO',
                    help='the ratio softmax loss - triplet loss (default: 2.0')

# args for additive margin-softmax
parser.add_argument('--margin', type=float, default=0.3, metavar='MARGIN',
                    help='the margin value for the angualr softmax loss function (default: 3.0')
parser.add_argument('--s', type=float, default=15, metavar='S',
                    help='the margin value for the angualr softmax loss function (default: 3.0')

# args for a-softmax
parser.add_argument('--m', type=int, default=3, metavar='M',
                    help='the margin value for the angualr softmax loss function (default: 3.0')
parser.add_argument('--lambda-min', type=int, default=5, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--lambda-max', type=float, default=0.05, metavar='S',
                    help='random seed (default: 0)')

# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=123456, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=12, metavar='LI',
                    help='how many batches to wait before logging training status')

parser.add_argument('--acoustic-feature', choices=['fbank', 'spectrogram', 'mfcc'], default='fbank',
                    help='choose the acoustic features type.')
parser.add_argument('--makemfb', action='store_true', default=False,
                    help='need to make mfb file')
parser.add_argument('--makespec', action='store_true', default=False,
                    help='need to make spectrograms file')

args = parser.parse_args()

# Set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.multiprocessing.set_sharing_strategy('file_system')

if args.cuda:
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

# create logger
# Define visulaize SummaryWriter instance

kwargs = {'num_workers': args.nj, 'pin_memory': True} if args.cuda else {}
sys.stdout = NewLogger(os.path.join(os.path.dirname(args.resume), 'test.log'))


l2_dist = nn.CosineSimilarity(dim=1, eps=1e-6) if args.cos_sim else PairwiseDistance(2)

if args.acoustic_feature == 'fbank':
    transform = transforms.Compose([
        # concateinputfromMFB(num_frames=c.NUM_FRAMES_SPECT, remove_vad=False),
        varLengthFeat(),
        to2tensor()
    ])
    transform_T = transforms.Compose([
        # concateinputfromMFB(num_frames=c.NUM_FRAMES_SPECT, input_per_file=args.test_input_per_file, remove_vad=False),
        varLengthFeat(),
        to2tensor()
    ])

else:
    transform = transforms.Compose([
        truncatedinput(),
        toMFB(),
        totensor(),
        # tonormal()
    ])
    file_loader = read_audio

# pdb.set_trace()
file_loader = read_mat
train_dir = ScriptTrainDataset(dir=args.train_dir, samples_per_speaker=args.input_per_spks, loader=file_loader,
                               transform=transform, num_valid=args.num_valid)
test_dir = ScriptTestDataset(dir=args.test_dir, loader=file_loader, transform=transform_T)

# indices = list(range(len(test_dir)))
# random.shuffle(indices)
# indices = indices[:args.veri_pairs]
# test_part = torch.utils.data.Subset(test_dir, indices)

# sitw_test_dir = SitwTestDataset(sitw_dir=args.sitw_dir, sitw_set='eval', transform=transform_T, set_suffix='')
# if len(sitw_test_dir) < args.veri_pairs:
#     args.veri_pairs = len(sitw_test_dir)
#     print('There are %d verification pairs in sitw eval.' % len(sitw_test_dir))
# else:
#     sitw_test_dir.partition(args.veri_pairs)

valid_dir = ScriptValidDataset(valid_set=train_dir.valid_set, loader=file_loader, spk_to_idx=train_dir.spk_to_idx,
                               valid_uid2feat=train_dir.valid_uid2feat, valid_utt2spk_dict=train_dir.valid_utt2spk_dict,
                               transform=transform)


def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive
    # test_display_triplet_distance = False
    # print the experiment configuration
    print('\nCurrent time is \33[91m{}\33[0m.'.format(str(time.asctime())))
    print('Parsed options: {}'.format(vars(args)))
    # print('Number of Speakers: {}.\n'.format(train_dir.num_spks))

    # instantiate model and initialize weights
    kernel_size = args.kernel_size.split(',')
    kernel_size = [int(x) for x in kernel_size]
    padding = [int((x - 1) / 2) for x in kernel_size]

    kernel_size = tuple(kernel_size)
    padding = tuple(padding)

    channels = args.channels.split(',')
    channels = [int(x) for x in channels]

    model_kwargs = {'embedding_size': args.embedding_size,
                    'resnet_size': args.resnet_size,
                    'input_dim': args.feat_dim,
                    'num_classes': train_dir.num_spks,
                    'channels': channels,
                    'kernel_size': kernel_size,
                    'padding': padding,
                    'dropout_p': args.dropout_p}

    print('Model options: {}'.format(model_kwargs))
    model = create_model(args.model, **model_kwargs)
    if args.loss_type == 'asoft':
        model.classifier = AngleLinear(in_features=args.embedding_size, out_features=train_dir.num_spks, m=args.m)
    elif args.loss_type == 'amsoft':
        model.classifier = AdditiveMarginLinear(feat_dim=args.embedding_size, n_classes=train_dir.num_spks)

    if os.path.isfile(args.resume):
        print('=> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        # start_epoch = checkpoint['epoch']

        filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}
        model_dict = model.state_dict()
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        #
        model.dropout.p = args.dropout_p
    else:
        print('=> no checkpoint found at {}'.format(args.resume))

    start = args.start_epoch
    print('Epoch is : ' + str(start))

    if args.cuda:
        model.cuda()
    # train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dir, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dir, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    # sitw_test_loader = torch.utils.data.DataLoader(sitw_test_dir, batch_size=args.test_batch_size,
    #                                                shuffle=False, **kwargs)
    # sitw_dev_loader = torch.utils.data.DataLoader(sitw_dev_part, batch_size=args.test_batch_size, shuffle=False,
    #                                               **kwargs)

    test(test_loader, valid_loader, model)
    # sitw_test(sitw_test_loader, model, epoch)
    # sitw_test(sitw_dev_loader, model, epoch)


def test(test_loader, valid_loader, model):
    # switch to evaluate mode
    model.eval()

    valid_pbar = tqdm(enumerate(valid_loader))
    softmax = nn.Softmax(dim=1)

    correct = 0.
    total_datasize = 0.

    for batch_idx, (data, label) in valid_pbar:
        data = Variable(data.cuda())

        # compute output
        out, _ = model(data)
        if args.loss_type == 'asoft':
            predicted_labels, _ = out
        else:
            predicted_labels = out

        true_labels = Variable(label.cuda())

        # pdb.set_trace()
        predicted_one_labels = softmax(predicted_labels)
        predicted_one_labels = torch.max(predicted_one_labels, dim=1)[1]

        batch_correct = (predicted_one_labels.cuda() == true_labels.cuda()).sum().item()
        minibatch_acc = float(batch_correct / len(predicted_one_labels))
        correct += batch_correct
        total_datasize += len(predicted_one_labels)

        if batch_idx % args.log_interval == 0:
            valid_pbar.set_description('Valid: [{:8d}/{:8d} ({:3.0f}%)] Batch Accuracy: {:.4f}%'.format(
                batch_idx * len(data),
                len(valid_loader.dataset),
                100. * batch_idx / len(valid_loader),
                100. * minibatch_acc
            ))

    valid_accuracy = 100. * correct / total_datasize
    torch.cuda.empty_cache()

    labels, distances = [], []
    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:

        vec_shape = data_a.shape
        # pdb.set_trace()
        if vec_shape[1] != 1:
            data_a = data_a.reshape(vec_shape[0] * vec_shape[1], 1, vec_shape[2], vec_shape[3])
            data_p = data_p.reshape(vec_shape[0] * vec_shape[1], 1, vec_shape[2], vec_shape[3])

        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p = Variable(data_a), Variable(data_p)

        # compute output
        _, out_a = model(data_a)
        _, out_p = model(data_p)

        dists = l2_dist.forward(out_a, out_p)  # torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        dists = dists.reshape(vec_shape[0], vec_shape[1]).mean(axis=1)
        dists = dists.data.cpu().numpy()

        distances.append(dists)
        labels.append(label.numpy())

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Test: [{}/{} ({:.0f}%)]'.format(
                batch_idx * len(data_a), len(test_loader.dataset), 100. * batch_idx / len(test_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    eer, eer_threshold, accuracy = evaluate_kaldi_eer(distances, labels, cos=args.cos_sim, re_thre=True)
    mindcf_01, mindcf_001 = evaluate_kaldi_mindcf(distances, labels)

    dist_type = 'cos' if args.cos_sim else 'l2'
    print('\nFor %s_distance, %d pairs:' % (dist_type, len(labels)))
    print('  \33[91mTest ERR is {:.4f}%, Threshold is {}'.format(100. * eer, eer_threshold))
    print('  mindcf-0.01 {:.4f}, mindcf-0.001 {:.4f},'.format(mindcf_01, mindcf_001))
    print('  Valid Accuracy is %.4f %%.\33[0m' % valid_accuracy)

    torch.cuda.empty_cache()


def sitw_test(sitw_test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []
    pbar = tqdm(enumerate(sitw_test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:

        vec_shape = data_a.shape
        # pdb.set_trace()
        if vec_shape[1] != 1:
            data_a = data_a.reshape(vec_shape[0] * vec_shape[1], 1, vec_shape[2], vec_shape[3])
            data_p = data_p.reshape(vec_shape[0] * vec_shape[1], 1, vec_shape[2], vec_shape[3])

        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p = Variable(data_a), Variable(data_p)

        # compute output
        _, out_a = model(data_a)
        _, out_p = model(data_p)
        dists = l2_dist.forward(out_a, out_p)  # torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        if vec_shape[1] != 1:
            dists = dists.reshape(vec_shape[0], vec_shape[1]).mean(axis=1)

        dists = dists.data.cpu().numpy()

        distances.append(dists)
        labels.append(label.numpy())

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * vec_shape[0], len(sitw_test_loader.dataset),
                       100. * batch_idx / len(sitw_test_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    eer_t, eer_threshold_t, accuracy = evaluate_kaldi_eer(distances, labels, cos=args.cos_sim, re_thre=True)
    torch.cuda.empty_cache()

    print('\33[91mFor Sitw Test ERR: {:.4f}%, Threshold: {}.\n\33[0m'.format(100. * eer_t, eer_threshold_t))


# python TrainAndTest/Spectrogram/train_surescnn10_kaldi.py > Log/SuResCNN10/spect_161/

if __name__ == '__main__':
    main()
