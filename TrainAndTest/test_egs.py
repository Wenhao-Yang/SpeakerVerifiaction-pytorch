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
from collections import OrderedDict

import kaldi_io
import numpy as np
import psutil
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from kaldi_io import read_mat, read_vec_flt
from torch.autograd import Variable
from tqdm import tqdm

from Define_Model.SoftmaxLoss import AngleLinear, AdditiveMarginLinear
from Define_Model.model import PairwiseDistance
from Eval.eval_metrics import evaluate_kaldi_eer, evaluate_kaldi_mindcf
from Process_Data.Datasets.KaldiDataset import ScriptTrainDataset, ScriptValidDataset, KaldiExtractDataset, \
    ScriptVerifyDataset
from Process_Data.audio_processing import ConcateOrgInput, ConcateVarInput, mvnormal
from TrainAndTest.common_func import create_model
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
parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition TEST')
# Data options
parser.add_argument('--train-dir', type=str, required=True, help='path to dataset')
parser.add_argument('--train-test-dir', type=str, help='path to dataset')
parser.add_argument('--valid-dir', type=str, help='path to dataset')
parser.add_argument('--test-dir', type=str, required=True, help='path to voxceleb1 test dataset')
parser.add_argument('--log-scale', action='store_true', default=False, help='log power spectogram')

parser.add_argument('--trials', type=str, default='trials', help='path to voxceleb1 test dataset')
parser.add_argument('--train-trials', type=str, default='trials', help='path to voxceleb1 test dataset')

parser.add_argument('--test-input', type=str, default='fix', help='path to voxceleb1 test dataset')
parser.add_argument('--remove-vad', action='store_true', default=False, help='using Cosine similarity')
parser.add_argument('--extract', action='store_false', default=True, help='need to make mfb file')
parser.add_argument('--frame-shift', default=200, type=int, metavar='N', help='acoustic feature dimension')

parser.add_argument('--nj', default=10, type=int, metavar='NJOB', help='num of job')
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
parser.add_argument('--xvector-dir', type=str, help='The dir for extracting xvectors')

parser.add_argument('--scheduler', default='multi', type=str,
                    metavar='SCH', help='The optimizer to use (default: Adagrad)')
parser.add_argument('--patience', default=2, type=int,
                    metavar='PAT', help='patience for scheduler (default: 4)')
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
parser.add_argument('--resnet-size', default=8, type=int, metavar='RES', help='The channels of convs layers)')
parser.add_argument('--filter', type=str, default='None', help='replace batchnorm with instance norm')
parser.add_argument('--mask-layer', type=str, default='None', help='replace batchnorm with instance norm')
parser.add_argument('--mask-len', type=int, default=20, help='maximum length of time or freq masking layers')
parser.add_argument('--block-type', type=str, default='None', help='replace batchnorm with instance norm')
parser.add_argument('--relu-type', type=str, default='relu', help='replace batchnorm with instance norm')
parser.add_argument('--transform', type=str, default="None", help='add a transform layer after embedding layer')

parser.add_argument('--vad', action='store_true', default=False, help='vad layers')
parser.add_argument('--inception', action='store_true', default=False, help='multi size conv layer')
parser.add_argument('--inst-norm', action='store_true', default=False, help='batchnorm with instance norm')
parser.add_argument('--input-norm', type=str, default='Mean', help='batchnorm with instance norm')
parser.add_argument('--encoder-type', type=str, default='None', help='path to voxceleb1 test dataset')
parser.add_argument('--channels', default='64,128,256', type=str,
                    metavar='CHA', help='The channels of convs layers)')
parser.add_argument('--context', default='5,3,3,5', type=str, metavar='KE', help='kernel size of conv filters')

parser.add_argument('--feat-dim', default=64, type=int, metavar='N', help='acoustic feature dimension')
parser.add_argument('--input-dim', default=257, type=int, metavar='N', help='acoustic feature dimension')
parser.add_argument('--input-length', type=str, help='batchnorm with instance norm')

parser.add_argument('--accu-steps', default=1, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument('--alpha', default=12, type=float, metavar='FEAT', help='acoustic feature dimension')
parser.add_argument('--kernel-size', default='5,5', type=str, metavar='KE', help='kernel size of conv filters')
parser.add_argument('--padding', default='', type=str, metavar='KE', help='padding size of conv filters')
parser.add_argument('--stride', default='2', type=str, metavar='ST', help='stride size of conv filters')
parser.add_argument('--fast', action='store_true', default=False, help='max pooling for fast')

parser.add_argument('--cos-sim', action='store_true', default=False, help='using Cosine similarity')
parser.add_argument('--avg-size', type=int, default=4, metavar='ES', help='Dimensionality of the embedding')
parser.add_argument('--time-dim', default=1, type=int, metavar='FEAT', help='acoustic feature dimension')
parser.add_argument('--embedding-size', type=int, default=128, metavar='ES',
                    help='Dimensionality of the embedding')
parser.add_argument('--batch-size', type=int, default=1, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--input-per-spks', type=int, default=224, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--num-valid', type=int, default=5, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--test-input-per-file', type=int, default=4, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='BST',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--dropout-p', type=float, default=0.25, metavar='BST',
                    help='input batch size for testing (default: 64)')

# loss configure
parser.add_argument('--loss-type', type=str, default='soft',
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
parser.add_argument('--grad-clip', default=0., type=float,
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
parser.add_argument('--mvnorm', action='store_true', default=False,
                    help='need to make spectrograms file')
parser.add_argument('--valid', action='store_true', default=False,
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
kwargs = {'num_workers': args.nj, 'pin_memory': False} if args.cuda else {}
sys.stdout = NewLogger(os.path.join(os.path.dirname(args.resume), 'test.log'))


l2_dist = nn.CosineSimilarity(dim=1, eps=1e-6) if args.cos_sim else PairwiseDistance(2)

if args.input_length == 'var':
    transform = transforms.Compose([
        ConcateOrgInput(remove_vad=args.remove_vad),
    ])
    transform_T = transforms.Compose([
        ConcateOrgInput(remove_vad=args.remove_vad),
    ])

elif args.input_length == 'fix':
    transform = transforms.Compose([
        ConcateVarInput(frame_shift=args.frame_shift, remove_vad=args.remove_vad),
    ])
    transform_T = transforms.Compose([
        ConcateVarInput(frame_shift=args.frame_shift, remove_vad=args.remove_vad),
    ])
else:
    raise ValueError('input length must be var or fix.')

if args.mvnorm:
    transform.transforms.append(mvnormal())
    transform_T.transforms.append(mvnormal())

# pdb.set_trace()
if args.feat_format == 'kaldi':
    file_loader = read_mat
    torch.multiprocessing.set_sharing_strategy('file_system')
elif args.feat_format == 'npy':
    file_loader = np.load

if not args.valid:
    args.num_valid = 0

train_dir = ScriptTrainDataset(dir=args.train_dir, samples_per_speaker=args.input_per_spks, loader=file_loader,
                               transform=transform, num_valid=args.num_valid)

verfify_dir = KaldiExtractDataset(dir=args.test_dir, transform=transform_T, filer_loader=file_loader)

if args.valid:
    valid_dir = ScriptValidDataset(valid_set=train_dir.valid_set, loader=file_loader,
                                   spk_to_idx=train_dir.spk_to_idx,
                                   valid_uid2feat=train_dir.valid_uid2feat,
                                   valid_utt2spk_dict=train_dir.valid_utt2spk_dict,
                                   transform=transform)


def valid(valid_loader, model):
    model.eval()

    valid_pbar = tqdm(enumerate(valid_loader))
    softmax = nn.Softmax(dim=1)

    correct = 0.
    total_datasize = 0.

    for batch_idx, (data, label) in valid_pbar:
        data = Variable(data.cuda())
        # print(model.conv1.weight)
        # print(data)
        # pdb.set_trace()

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
    print('  \33[91mValid Accuracy is %.4f %%.\33[0m' % valid_accuracy)
    torch.cuda.empty_cache()


def extract(test_loader, model, xvector_dir, ark_num=50000):
    model.eval()

    if not os.path.exists(xvector_dir):
        os.makedirs(xvector_dir)
        print('Creating xvector path: %s' % xvector_dir)

    pbar = tqdm(enumerate(test_loader))
    vectors = []
    uids = []
    for batch_idx, (data, uid) in pbar:
        # print(model.conv1.weight)
        # print(data)
        # raise ValueError('Conv1')

        vec_shape = data.shape
        # pdb.set_trace()
        if vec_shape[1] != 1:
            # print(data.shape)
            data = data.reshape(vec_shape[0] * vec_shape[1], 1, vec_shape[2], vec_shape[3])

        if args.cuda:
            data = data.cuda()

        data = Variable(data)

        # compute output
        _, out = model(data)

        if vec_shape[1] != 1:
            out = out.reshape(vec_shape[0], vec_shape[1], out.shape[-1]).mean(dim=1)

        # pdb.set_trace()

        vectors.append(out.squeeze().data.cpu().numpy())
        uids.append(uid[0])

        del data, out
        if batch_idx % args.log_interval == 0:
            pbar.set_description('Extracting: [{}/{} ({:.0f}%)]'.format(
                batch_idx, len(test_loader.dataset), 100. * batch_idx / len(test_loader)))

    assert len(uids) == len(vectors)
    print('There are %d vectors' % len(uids))
    scp_file = xvector_dir + '/xvectors.scp'
    scp = open(scp_file, 'w')

    # write scp and ark file
    # pdb.set_trace()
    for set_id in range(int(np.ceil(len(uids) / ark_num))):
        ark_file = xvector_dir + '/xvector.{}.ark'.format(set_id)
        with open(ark_file, 'wb') as ark:
            ranges = np.arange(len(uids))[int(set_id * ark_num):int((set_id + 1) * ark_num)]
            for i in ranges:
                vec = vectors[i]
                len_vec = len(vec.tobytes())
                key = uids[i]
                kaldi_io.write_vec_flt(ark, vec, key=key)
                # print(ark.tell())
                scp.write(str(uids[i]) + ' ' + str(ark_file) + ':' + str(ark.tell() - len_vec - 10) + '\n')
    scp.close()
    print('There are %d vectors. Saving to %s' % (len(uids), xvector_dir))
    torch.cuda.empty_cache()


def test(test_loader):
    # switch to evaluate mode
    labels, distances = [], []
    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:

        out_a = torch.tensor(data_a)
        out_p = torch.tensor(data_p)

        dists = l2_dist.forward(out_a, out_p)  # torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        dists = dists.numpy()

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
    test_directorys = args.test_dir.split('/')
    test_set_name = '-'
    for i, dir in enumerate(test_directorys):
        if dir == 'data':
            test_subset = test_directorys[i + 3].split('_')[0]
            test_set_name = "-".join((test_directorys[i + 1], test_subset))

    print('\nFor %s_distance, %d pairs:' % (dist_type, len(labels)))
    print('\33[91m')
    print('+--------------+-------------+-------------+-------------+--------------+-------------------+')
    print('|{: ^14s}|{: ^13s}|{: ^13s}|{: ^13s}|{: ^14s}|{: ^19s}|'.format('Test Set',
                                                                           'EER (%)',
                                                                           'Threshold',
                                                                           'MinDCF-0.01',
                                                                           'MinDCF-0.001',
                                                                           'Date'))
    print('+--------------+-------------+-------------+-------------+--------------+-------------------+')
    eer = '{:.4f}%'.format(eer * 100.)
    threshold = '{:.4f}'.format(eer_threshold)
    mindcf_01 = '{:.4f}'.format(mindcf_01)
    mindcf_001 = '{:.4f}'.format(mindcf_001)
    date = time.strftime("%Y%m%d %H:%M:%S", time.localtime())

    print('    |{: ^14s}|{: ^13s}|{: ^13s}|{: ^13s}|{: ^14s}|{: ^19s}|'.format(test_set_name,
                                                                               eer,
                                                                               threshold,
                                                                               mindcf_01,
                                                                               mindcf_001,
                                                                               date))
    print('+--------------+-------------+-------------+-------------+--------------+-------------------+')
    print('\33[0m')


if __name__ == '__main__':

    # Views the training images and displays the distance on anchor-negative and anchor-positive
    # test_display_triplet_distance = False
    # print the experiment configuration
    print('\nCurrent time is \33[91m{}\33[0m.'.format(str(time.asctime())))
    opts = vars(args)
    keys = list(opts.keys())
    keys.sort()

    options = []
    for k in keys:
        options.append("\'%s\': \'%s\'" % (str(k), str(opts[k])))

    print('Parsed options: \n{ %s }' % (', '.join(options)))
    print('Number of Speakers: {}.\n'.format(train_dir.num_spks))

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
    context = args.context.split(',')
    context = [int(x) for x in context]

    model_kwargs = {'input_dim': args.input_dim, 'feat_dim': args.feat_dim, 'kernel_size': kernel_size,
                    'mask_layer': args.mask_layer, 'mask_len': args.mask_len, 'block_type': args.block_type,
                    'filter': args.filter, 'inst_norm': args.inst_norm, 'input_norm': args.input_norm,
                    'stride': stride, 'fast': args.fast, 'avg_size': args.avg_size, 'time_dim': args.time_dim,
                    'padding': padding, 'encoder_type': args.encoder_type, 'vad': args.vad,
                    'transform': args.transform, 'embedding_size': args.embedding_size, 'ince': args.inception,
                    'resnet_size': args.resnet_size, 'num_classes': train_dir.num_spks,
                    'channels': channels, 'context': context,
                    'alpha': args.alpha, 'dropout_p': args.dropout_p,
                    'loss_type': args.loss_type, 'm': args.m, 'margin': args.margin, 's': args.s, }

    print('Model options: {}'.format(model_kwargs))
    dist_type = 'cos' if args.cos_sim else 'l2'
    print('Testing with %s distance, ' % dist_type)

    if args.valid or args.extract:
        model = create_model(args.model, **model_kwargs)
        # if args.loss_type == 'asoft':
        #     model.classifier = AngleLinear(in_features=args.embedding_size, out_features=train_dir.num_spks, m=args.m)
        # elif args.loss_type in ['amsoft', 'arcsoft']:
        #     model.classifier = AdditiveMarginLinear(feat_dim=args.embedding_size, n_classes=train_dir.num_spks)

        assert os.path.isfile(args.resume)
        print('=> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        # start_epoch = checkpoint['epoch']

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
            filtered = new_state_dict

        if 'fc1.1.weight' in filtered:
            model.fc1 = nn.Sequential(
                nn.Linear(model.encoder_output, model.embedding_size),
                nn.BatchNorm1d(model.embedding_size)
            )

        model_dict = model.state_dict()
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        # model.dropout.p = args.dropout_p
        #
        try:
            model.dropout.p = args.dropout_p
        except:
            pass
        start = args.start_epoch
        print('Epoch is : ' + str(start))

        if args.cuda:
            model.cuda()
        # train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=True, **kwargs)

        if args.valid:
            valid_loader = torch.utils.data.DataLoader(valid_dir, batch_size=args.test_batch_size, shuffle=False,
                                                       **kwargs)
            valid(valid_loader, model)

        del train_dir  # , valid_dir
        print('Memery Usage: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))

        if args.extract:
            verify_loader = torch.utils.data.DataLoader(verfify_dir, batch_size=args.test_batch_size, shuffle=False,
                                                        **kwargs)
            extract(verify_loader, model, args.xvector_dir)

    file_loader = read_vec_flt
    test_dir = ScriptVerifyDataset(dir=args.test_dir, trials_file=args.trials, xvectors_dir=args.xvector_dir,
                                   loader=file_loader)
    test_loader = torch.utils.data.DataLoader(test_dir, batch_size=args.test_batch_size * 64, shuffle=False, **kwargs)
    test(test_loader)

# python TrainAndTest/Spectrogram/train_surescnn10_kaldi.py > Log/SuResCNN10/spect_161/

# test easy spectrogram soft 161 vox1
#   Test ERR is 1.6076%, Threshold is 0.31004807353019714
#   mindcf-0.01 0.2094, mindcf-0.001 0.3767.

# test hard spectrogram soft 161 vox1
#   Test ERR is 2.9182%, Threshold is 0.35036733746528625
#   mindcf-0.01 0.3369, mindcf-0.001 0.5494.
