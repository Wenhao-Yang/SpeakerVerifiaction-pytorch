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
from TrainAndTest.common_func import argparse_adv

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
args = argparse_adv("Pytorch Speaker Recogniton: Adversarial Testing")

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

train_dir_a = ScriptTrainDataset(dir=args.train_dir_a, samples_per_speaker=args.input_per_spks, loader=file_loader,
                                 transform=transform, num_valid=args.num_valid)
train_dir_b = ScriptTrainDataset(dir=args.train_dir_b, samples_per_speaker=args.input_per_spks, loader=file_loader,
                                 transform=transform, num_valid=args.num_valid)

verfify_dir = KaldiExtractDataset(dir=args.test_dir, transform=transform_T, filer_loader=file_loader)

if args.valid:
    valid_dir_a = ScriptValidDataset(valid_set=train_dir_a.valid_set, loader=file_loader,
                                     spk_to_idx=train_dir_a.spk_to_idx,
                                     valid_uid2feat=train_dir_a.valid_uid2feat,
                                     valid_utt2spk_dict=train_dir_a.valid_utt2spk_dict,
                                     transform=transform)

    valid_dir_b = ScriptValidDataset(valid_set=train_dir_b.valid_set, loader=file_loader,
                                     spk_to_idx=train_dir_b.spk_to_idx,
                                     valid_uid2feat=train_dir_b.valid_uid2feat,
                                     valid_utt2spk_dict=train_dir_b.valid_utt2spk_dict,
                                     transform=transform)


def valid(valid_loader, model):
    model.eval()

    valid_loader_a, valid_loader_b = valid_loader

    valid_pbar_a = tqdm(enumerate(valid_loader_a))
    softmax = nn.Softmax(dim=1)

    correct = 0.
    total_datasize = 0.

    for batch_idx, (data, label) in valid_pbar_a:
        data = Variable(data.cuda())
        # print(model.conv1.weight)
        # print(data)
        # pdb.set_trace()

        # compute output
        _, out = model(data)
        predicted_labels = model.classifier_a(out)
        true_labels = Variable(label.cuda())

        # pdb.set_trace()
        predicted_one_labels = softmax(predicted_labels)
        predicted_one_labels = torch.max(predicted_one_labels, dim=1)[1]

        batch_correct = (predicted_one_labels.cuda() == true_labels.cuda()).sum().item()
        minibatch_acc = float(batch_correct / len(predicted_one_labels))
        correct += batch_correct
        total_datasize += len(predicted_one_labels)

        if batch_idx % args.log_interval == 0:
            valid_pbar_a.set_description('Valid: [{:8d}/{:8d} ({:3.0f}%)] Batch Accuracy: {:.4f}%'.format(
                batch_idx * len(data),
                len(valid_loader.dataset),
                100. * batch_idx / len(valid_loader),
                100. * minibatch_acc
            ))

    valid_accuracy = 100. * correct / total_datasize
    print('  \33[91mValid Accuracy for set A is %.4f %%.\33[0m' % valid_accuracy)

    valid_pbar_b = tqdm(enumerate(valid_loader_b))
    softmax = nn.Softmax(dim=1)

    correct = 0.
    total_datasize = 0.

    for batch_idx, (data, label) in valid_pbar_b:
        data = Variable(data.cuda())

        # compute output
        _, out = model(data)
        predicted_labels = model.classifier_b(out)
        true_labels = Variable(label.cuda())

        # pdb.set_trace()
        predicted_one_labels = softmax(predicted_labels)
        predicted_one_labels = torch.max(predicted_one_labels, dim=1)[1]

        batch_correct = (predicted_one_labels.cuda() == true_labels.cuda()).sum().item()
        minibatch_acc = float(batch_correct / len(predicted_one_labels))
        correct += batch_correct
        total_datasize += len(predicted_one_labels)

        if batch_idx % args.log_interval == 0:
            valid_pbar_a.set_description('Valid: [{:8d}/{:8d} ({:3.0f}%)] Batch Accuracy: {:.4f}%'.format(
                batch_idx * len(data),
                len(valid_loader.dataset),
                100. * batch_idx / len(valid_loader),
                100. * minibatch_acc
            ))

    valid_accuracy = 100. * correct / total_datasize
    print('  \33[91mValid Accuracy for set B is %.4f %%.\33[0m' % valid_accuracy)

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
    print('\nFor %s_distance, %d pairs:' % (dist_type, len(labels)))
    print('  \33[91mTest ERR is {:.4f}%, Threshold is {}'.format(100. * eer, eer_threshold))
    print('  mindcf-0.01 {:.4f}, mindcf-0.001 {:.4f}.\33[0m'.format(mindcf_01, mindcf_001))


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
    print('Number of Speakers for set A: {}.'.format(train_dir_a.num_spks))
    print('Number of Speakers for set B: {}.\n'.format(train_dir_b.num_spks))

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
                    'mask_layer': args.mask_layer, 'mask_len': args.mask_len, 'block_type': args.block_type,
                    'filter': args.filter, 'inst_norm': args.inst_norm, 'input_norm': args.input_norm,
                    'stride': stride, 'fast': args.fast, 'avg_size': args.avg_size, 'time_dim': args.time_dim,
                    'padding': padding, 'encoder_type': args.encoder_type, 'vad': args.vad,
                    'transform': args.transform, 'embedding_size': args.embedding_size, 'ince': args.inception,
                    'resnet_size': args.resnet_size, 'num_classes_a': train_dir_a.num_spks,
                    'num_classes_b': train_dir_b.num_spks, 'channels': channels,
                    'alpha': args.alpha, 'dropout_p': args.dropout_p}

    print('Model options: {}'.format(model_kwargs))
    dist_type = 'cos' if args.cos_sim else 'l2'
    print('Testing with %s distance, ' % dist_type)

    if args.valid or args.extract:
        model = create_model(args.model, **model_kwargs)
        if args.loss_type == 'asoft':
            model.classifier_a = AngleLinear(in_features=args.embedding_size, out_features=train_dir_a.num_spks,
                                             m=args.m)
            model.classifier_b = AngleLinear(in_features=args.embedding_size, out_features=train_dir_b.num_spks,
                                             m=args.m)
        elif args.loss_type in ['amsoft', 'arcsoft']:
            model.classifier_a = AdditiveMarginLinear(feat_dim=args.embedding_size, n_classes=train_dir_a.num_spks)
            model.classifier_b = AdditiveMarginLinear(feat_dim=args.embedding_size, n_classes=train_dir_b.num_spks)

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

            model.load_state_dict(new_state_dict)
        else:
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
            valid_loader_a = torch.utils.data.DataLoader(valid_dir_a, batch_size=args.test_batch_size, shuffle=False,
                                                         **kwargs)
            valid_loader_b = torch.utils.data.DataLoader(valid_dir_b, batch_size=args.test_batch_size, shuffle=False,
                                                         **kwargs)
            valid_loader = [valid_loader_a, valid_loader_b]
            valid(valid_loader, model)

        del train_dir_a, train_dir_b  # , valid_dir
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
