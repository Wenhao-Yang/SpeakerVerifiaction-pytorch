#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: extract_xvector_kaldi.py
@Time: 2019/12/10 下午10:32
@Overview: Exctract speakers vectors for kaldi PLDA.
"""
from __future__ import print_function

import argparse
import os
import time

import numpy as np
import torch
import torch._utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from kaldi_io import read_mat
from tqdm import tqdm

from Define_Model.model import PairwiseDistance
# from Process_Data.voxceleb2_wav_reader import voxceleb2_list_reader
from Process_Data.KaldiDataset import KaldiExtractDataset
from Process_Data.audio_processing import toMFB, totensor, truncatedinput, read_audio, ConcateVarInput
from TrainAndTest.common_func import create_model

# Version conflict

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor


    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
import warnings

warnings.filterwarnings("ignore")

# Training settings
parser = argparse.ArgumentParser(description='Extract x-vector for plda')
# Model options
parser.add_argument('--enroll-dir', type=str, help='path to dataset')
parser.add_argument('--train-spk-a', type=int, metavar='NJOB', help='num of job')
parser.add_argument('--train-spk-b', type=int, metavar='NJOB', help='num of job')

parser.add_argument('--test-dir', type=str, help='path to voxceleb1 test dataset')
parser.add_argument('--sitw-dir', type=str, help='path to voxceleb1 test dataset')

parser.add_argument('--trials', type=str, default='trials', help='path to voxceleb1 test dataset')
parser.add_argument('--remove-vad', action='store_true', default=False, help='using Cosine similarity')
parser.add_argument('--extract', action='store_true', default=True, help='need to make mfb file')
parser.add_argument('--extract-path', type=str, help='need to make mfb file')

parser.add_argument('--nj', default=8, type=int, metavar='NJOB', help='num of job')
parser.add_argument('--feat-format', type=str, default='kaldi', choices=['kaldi', 'npy'],
                    help='number of jobs to make feats (default: 10)')

parser.add_argument('--check-path', default='Data/checkpoint/GradResNet8/vox1/spect_egs/soft_dp25',
                    help='folder to output model checkpoints')
parser.add_argument('--save-init', action='store_true', default=True, help='need to make mfb file')
parser.add_argument('--resume',
                    default='Data/checkpoint/GradResNet8/vox1/spect_egs/soft_dp25/checkpoint_10.pth', type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=20, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--scheduler', default='multi', type=str,
                    metavar='SCH', help='The optimizer to use (default: Adagrad)')
parser.add_argument('--gamma', default=0.75, type=float,
                    metavar='GAMMA', help='The optimizer to use (default: Adagrad)')
parser.add_argument('--milestones', default='10,15', type=str,
                    metavar='MIL', help='The optimizer to use (default: Adagrad)')
parser.add_argument('--min-softmax-epoch', type=int, default=40, metavar='MINEPOCH',
                    help='minimum epoch for initial parameter using softmax (default: 2')
parser.add_argument('--veri-pairs', type=int, default=20000, metavar='VP',
                    help='number of epochs to train (default: 10)')

# Training options
# Model options
parser.add_argument('--model', type=str, help='path to voxceleb1 test dataset')
parser.add_argument('--resnet-size', default=8, type=int,
                    metavar='RES', help='The channels of convs layers)')
parser.add_argument('--filter', type=str, default='None', help='replace batchnorm with instance norm')
parser.add_argument('--transform', type=str, default='None', help='add a transform layer after embedding layer')

parser.add_argument('--vad', action='store_true', default=False, help='vad layers')
parser.add_argument('--inception', action='store_true', default=False, help='multi size conv layer')
parser.add_argument('--inst-norm', action='store_true', default=False, help='batchnorm with instance norm')
parser.add_argument('--input-norm', type=str, default='Mean', help='batchnorm with instance norm')
parser.add_argument('--encoder-type', type=str, default='SAP', help='path to voxceleb1 test dataset')
parser.add_argument('--channels', default='64,128,256', type=str,
                    metavar='CHA', help='The channels of convs layers)')
parser.add_argument('--feat-dim', default=64, type=int, metavar='N', help='acoustic feature dimension')
parser.add_argument('--input-dim', default=257, type=int, metavar='N', help='acoustic feature dimension')
parser.add_argument('--accu-steps', default=1, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument('--alpha', default=12, type=float, metavar='FEAT', help='acoustic feature dimension')
parser.add_argument('--kernel-size', default='5,5', type=str, metavar='KE', help='kernel size of conv filters')
parser.add_argument('--padding', default='', type=str, metavar='KE', help='padding size of conv filters')
parser.add_argument('--stride', default='2', type=str, metavar='ST', help='stride size of conv filters')
parser.add_argument('--fast', action='store_true', default=False, help='max pooling for fast')

parser.add_argument('--cos-sim', action='store_true', default=False, help='using Cosine similarity')
parser.add_argument('--avg-size', type=int, default=4, metavar='ES', help='Dimensionality of the embedding')
parser.add_argument('--time-dim', default=2, type=int, metavar='FEAT', help='acoustic feature dimension')
parser.add_argument('--embedding-size', type=int, default=128, metavar='ES',
                    help='Dimensionality of the embedding')
parser.add_argument('--batch-size', type=int, default=128, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--input-per-spks', type=int, default=224, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--num-valid', type=int, default=5, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--test-input-per-file', type=int, default=4, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--test-batch-size', type=int, default=4, metavar='BST',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--dropout-p', type=float, default=0.25, metavar='BST',
                    help='input batch size for testing (default: 64)')

# loss configure
parser.add_argument('--loss-type', type=str, default='soft', choices=['soft', 'asoft', 'center', 'amsoft',
                                                                      'wasse', 'mulcenter', 'arcsoft'],
                    help='path to voxceleb1 test dataset')
parser.add_argument('--num-center', type=int, default=2, help='the num of source classes')
parser.add_argument('--source-cls', type=int, default=1951,
                    help='the num of source classes')

parser.add_argument('--finetune', action='store_true', default=False,
                    help='using Cosine similarity')
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
parser.add_argument('--lambda-max', type=float, default=1000, metavar='S',
                    help='random seed (default: 0)')

parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.125)')
parser.add_argument('--lr-decay', default=0, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-4')
parser.add_argument('--weight-decay', default=5e-4, type=float,
                    metavar='WEI', help='weight decay (default: 0.0)')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='MOM', help='momentum for sgd (default: 0.9)')
parser.add_argument('--dampening', default=0, type=float,
                    metavar='DAM', help='dampening for sgd (default: 0.0)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')
parser.add_argument('--grad-clip', default=10., type=float,
                    help='momentum for sgd (default: 0.9)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=123456, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
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

if args.cuda:
    cudnn.benchmark = True

# create logger
# Define visulaize SummaryWriter instance

kwargs = {'num_workers': 12, 'pin_memory': True} if args.cuda else {}
l2_dist = nn.CosineSimilarity(dim=1, eps=1e-6) if args.cos_sim else PairwiseDistance(2)

if args.acoustic_feature == 'fbank':
    transform = transforms.Compose([
        ConcateVarInput(),
    ])
    transform_T = transforms.Compose([
        ConcateVarInput(),
    ])
    file_loader = read_mat
else:
    transform = transforms.Compose([
        truncatedinput(),
        toMFB(),
        totensor(),
        # tonormal()
    ])
    file_loader = read_audio

# pdb.set_trace()
enroll_dir = KaldiExtractDataset(dir=args.enroll_dir, filer_loader=file_loader, transform=transform)
test_dir = KaldiExtractDataset(dir=args.test_dir, filer_loader=file_loader, transform=transform)


# test_dir = ScriptTestDataset(dir=args.test_dir, loader=file_loader, transform=transform_T)

def extract(data_loader, model, set_id, extract_path):
    model.eval()
    uids, xvector = [], []
    pbar = tqdm(enumerate(data_loader))

    with torch.no_grad():
        for batch_idx, (data, uid) in pbar:

            vec_shape = data.shape
            if vec_shape[1] != 1:
                data = data.reshape(vec_shape[0] * vec_shape[1], 1, vec_shape[2], vec_shape[3])

            data = data.cuda()
            _, feats = model(data)
            feats = feats.data.cpu().numpy()

            xvector.append(feats)
            uids.append(uid[0])
            # xvector = torch.cat((xvector, feats), dim=0)
            # for i in range(len(uid)):
            #     uids.append(uid[i])

            if batch_idx % args.log_interval == 0:
                pbar.set_description(
                    'Extract {}: [{:8d}/{:8d} ({:3.0f}%)] '.format(
                        set_id,
                        batch_idx * len(data),
                        len(data_loader.dataset),
                        100. * batch_idx / len(data_loader)))
                # break

    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    if not os.path.exists(extract_path + '/npy_vectors'):
        os.makedirs(extract_path + '/npy_vectors')

    with open(extract_path + '/utt2vec', 'w') as scp:

        assert len(uids) == len(xvector)
        for i in range(len(uids)):
            xvector_path = os.path.join(extract_path, 'npy_vectors', uids[i] + '.npy')

            np.save(xvector_path, xvector[i])
            scp.write(uids[i] + " " + xvector_path + '\n')


def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive

    # print the experiment configuration
    print('\nCurrent time is\33[91m {}\33[0m.'.format(str(time.asctime())))
    print('Parsed options: {}'.format(vars(args)))
    print('Number of Speakers: {}\n'.format(args.train_spk))

    # instantiate model and initialize weights
    kernel_size = args.kernel_size.split(',')
    kernel_size = [int(x) for x in kernel_size]
    if args.padding == '':
        padding = [int((x - 1) / 2) for x in kernel_size]
    else:
        padding = args.padding.split(',')
        padding = [int(x) for x in padding]

    kernel_size = tuple(kernel_size)
    padding = tuple(padding)
    stride = args.stride.split(',')
    stride = [int(x) for x in stride]

    channels = args.channels.split(',')
    channels = [int(x) for x in channels]

    model_kwargs = {'input_dim': args.input_dim, 'feat_dim': args.feat_dim, 'kernel_size': kernel_size,
                    'filter': args.filter, 'inst_norm': args.inst_norm, 'input_norm': args.input_norm,
                    'stride': stride, 'fast': args.fast, 'avg_size': args.avg_size, 'time_dim': args.time_dim,
                    'padding': padding, 'encoder_type': args.encoder_type, 'vad': args.vad,
                    'transform': args.transform, 'embedding_size': args.embedding_size, 'ince': args.inception,
                    'resnet_size': args.resnet_size, 'num_classes_a': args.train_spk_a,
                    'num_classes_b': args.train_spk_b,
                    'channels': channels, 'alpha': args.alpha, 'dropout_p': args.dropout_p}

    print('Model options: {}'.format(model_kwargs))
    model = create_model(args.model, **model_kwargs)

    if args.cuda:
        model.cuda()

    if os.path.isfile(args.resume):
        # print('=> loading checkpoint {}'.format(resume))
        checkpoint = torch.load(args.resume)
        filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}
        model_dict = model.state_dict()
        model_dict.update(filtered)
        model.load_state_dict(model_dict)

    else:
        raise Exception('=> no checkpoint found at {}'.format(args.resume))

    # Extract enroll set vectors
    # train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=False, **kwargs)
    enroll_loader = torch.utils.data.DataLoader(enroll_dir, batch_size=args.batch_size, shuffle=False, **kwargs)
    extract(enroll_loader, model, set_id='enroll', extract_path=args.extract_path + '/x_vector')

    # Extract test set vectors
    test_loader = torch.utils.data.DataLoader(test_dir, batch_size=args.batch_size, shuffle=False, **kwargs)
    extract(test_loader, model, set_id='test', extract_path=args.extract_path + '/x_vector')

    print('Extract x-vector completed for train and test in %s!\n' % (args.extract_path + '/x_vector'))


f

if __name__ == '__main__':
    main()
