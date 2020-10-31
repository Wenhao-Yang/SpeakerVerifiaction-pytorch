#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: enroll_eval.py
@Time: 2020/10/31 13:47
@Overview:
"""

from __future__ import print_function

import argparse
import os

import numpy as np
import torch
import torch._utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
from kaldi_io import read_mat

from Define_Model.model import PairwiseDistance

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
parser.add_argument('--data-dir', type=str, help='path to dataset')
parser.add_argument('--enroll-dir', type=str, help='path to voxceleb1 test dataset')
parser.add_argument('--test-dir', type=str, help='path to voxceleb1 test dataset')

parser.add_argument('--split-set', action='store_true', default=False, help='using Cosine similarity')
parser.add_argument('--cos-sim', action='store_true', default=False, help='using Cosine similarity')

parser.add_argument('--trials', type=str, default='trials', help='path to voxceleb1 test dataset')
parser.add_argument('--extract-path', type=str, help='need to make mfb file')
parser.add_argument('--nj', default=10, type=int, metavar='NJOB', help='num of job')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=123456, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')

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

dist_type = nn.CosineSimilarity(dim=1, eps=1e-6) if args.cos_sim else PairwiseDistance(2)


# file_loader = np.load


def Split_Set(data_dir, xvector_dir, file_loader=read_mat, split_set=True):
    if not split_set:
        return os.path.join(xvector_dir, 'enroll'), os.path.join(xvector_dir, 'test')

    feats_scp = os.path.join(data_dir, 'feats.scp')
    utt2dur = os.path.join(data_dir, 'utt2dur')
    utt2num_frames = os.path.join(data_dir, 'utt2num_frames')

    utt2vec_scp = os.path.join(xvector_dir, 'utt2vec')
    spk2utt_scp = os.path.join(data_dir, 'spk2utt')

    spk2utt_enroll_scp = os.path.join(xvector_dir, 'enroll', 'spk2utt')
    utt2spk_enroll_scp = os.path.join(xvector_dir, 'enroll', 'utt2spk')
    enroll_utt2vec_scp = os.path.join(data_dir, 'enroll', 'utt2vec')

    spk2utt_test_scp = os.path.join(xvector_dir, 'test', 'spk2utt')
    utt2spk_test_scp = os.path.join(xvector_dir, 'test', 'utt2spk')
    test_utt2vec_scp = os.path.join(data_dir, 'test', 'utt2vec')

    print("Getting duration for utterances...")
    utt2len = {}  # utt2seconds
    if os.path.exists(utt2dur):
        with open(utt2dur, 'r') as f:
            for l in f.readlines():
                utt, ulen = l.split()
                utt2len[utt] = float(ulen)
    elif os.path.exists(utt2num_frames):
        with open(utt2num_frames, 'r') as f:
            for l in f.readlines():
                utt, ulen = l.split()
                utt2len[utt] = int(ulen) / 100.
    else:
        with open(feats_scp, 'r') as f:
            for l in f.readlines():
                utt, vec = l.split()
                utt2len[utt] = len(file_loader(vec)) / 100.


    utt2vec = {}
    with open(utt2vec_scp, 'r') as s2u:
        for l in s2u.readlines():
            utt, vec = l.split()
            utt2vec[utt] = vec

    spk2utt = {}
    enroll_spk2utt = {}
    test_spk2utt = {}
    print("Splitting enroll and test set for utterances...")
    with open(spk2utt_scp, 'r') as s2u:
        for l in s2u.readlines():
            spkutts = l.split()
            this_spk = spkutts[0]
            this_utts = spkutts[1:]
            this_utts.sort()

            spk2utt[this_spk] = this_utts
    for spk in spk2utt.keys():
        if spk not in enroll_spk2utt.keys():
            enroll_spk2utt[spk] = []
        if spk not in test_spk2utt.keys():
            test_spk2utt[spk] = []

        enroll_len = 0
        for idx, u in enumerate(spk2utt[spk]):
            if enroll_len < 45. and len(spk2utt[spk]) - idx + 1 > 0:
                enroll_spk2utt[spk].append(u)
                enroll_len += utt2len[u]
            else:
                test_spk2utt[spk].append(u)

    print("Writing enroll files in %s..." % os.path.join(xvector_dir, 'enroll'))
    with open(spk2utt_enroll_scp, 'w') as f1, \
            open(enroll_utt2vec_scp, 'w') as f2, \
            open(utt2spk_enroll_scp, 'w') as f3:
        for spk in enroll_spk2utt:
            f1.write(spk + ' ')
            f1.write(' '.join(enroll_spk2utt[spk]))
            f1.write('\n')

            for u in enroll_spk2utt[spk]:
                f2.write(u + ' ' + utt2vec[u] + '\n')
                f3.write(u + ' ' + spk + '\n')

    print("Writing test files in %s..." % os.path.join(xvector_dir, 'test'))
    with open(spk2utt_test_scp, 'w') as f1, \
            open(test_utt2vec_scp, 'w') as f2, \
            open(utt2spk_test_scp, 'w') as f3:
        for spk in test_spk2utt:
            f1.write(spk + ' ')
            f1.write(' '.join(test_spk2utt[spk]))
            f1.write('\n')

            for u in test_spk2utt[spk]:
                f2.write(u + ' ' + utt2vec[u] + '\n')
                f3.write(u + ' ' + spk + '\n')

    return os.path.join(xvector_dir, 'enroll'), os.path.join(xvector_dir, 'test')


def Enroll(enroll_dir, file_loader=np.load):
    num_update = 0
    spk_xvector_dir = os.path.join(enroll_dir, 'spk2vec_dir')
    if not os.path.exists(spk_xvector_dir):
        os.makedirs(spk_xvector_dir)

    spk2utt_scp = os.path.join(enroll_dir, 'spk2utt')
    utt2vec_scp = os.path.join(enroll_dir, 'utt2vec')
    spk2xve_scp = os.path.join(enroll_dir, 'spk2vec')

    uid2xve_dict = {}
    assert os.path.exists(utt2vec_scp), print('%s Existed??' % utt2vec_scp)
    with open(utt2vec_scp, 'r') as f:
        for l in f.readlines():
            uid, xve_path = l.split()
            assert (uid not in uid2xve_dict.keys())
            uid2xve_dict[uid] = xve_path

    spk2utt_dict = {}
    assert os.path.exists(spk2utt_scp), print('%s Existed??' % spk2utt_scp)
    with open(spk2utt_scp, 'r') as f:
        for l in f.readlines():
            sid, uid = l.split()
            xve_path = uid2xve_dict[uid]
            if sid in spk2utt_dict.keys():
                spk2utt_dict[sid].append(xve_path)
            else:
                spk2utt_dict[sid] = [xve_path]

    sids = list(spk2utt_dict.keys())
    sids.sort()

    print("Averaging enrolled spk vector...")
    spk2xve_dict = {}
    # print('[', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '] Saving in npy')
    for sid in sids:
        xvector = []
        for xve_path in spk2utt_dict[sid]:
            this_xve = file_loader(xve_path)
            if len(this_xve.shape) == 0:
                print('Empty??')
                continue
            xvector.append(this_xve)
        all_xvector = np.concatenate(xvector, axis=0)
        this_vec_len = len(all_xvector)
        mean_xvector = np.mean(xvector, axis=0)
        new_xvector = np.append([this_vec_len], mean_xvector)

        vec_path = '/'.join((spk_xvector_dir, '%s.npy' % sid))
        np.save(vec_path, new_xvector)
        spk2xve_dict[sid] = vec_path

    # assert os.path.exists(spk2xve_scp), print('%s ?'%spk2xve_scp)
    if not os.path.exists(os.path.dirname(spk2xve_scp)):
        os.makedirs(os.path.dirname(spk2xve_scp))

    print("Saving enrolled spk vector list...")
    with open(spk2xve_scp, 'w') as f:
        for sid in sids:
            xve_path = spk2xve_dict[sid]
            if os.path.exists(xve_path):
                f.write('%s %s\n' % (sid, xve_path))
                num_update += 1

    return num_update


def Eval(enroll_dir, eval_dir, file_loader=np.load):
    """
    :param enroll_dir: 已有说话人sid2xve列表的文件夹
    :param eval_dir: 需要识别的语音列表文件夹, utt2vec列表的文件夹
    :param file_loader:
    :return: 写入"uid sid" utt2sid文件中，未知的说话人用 unknown 标识
    """

    num_eval = 0

    utt2vec_scp = os.path.join(eval_dir, 'utt2vec')
    utt2spk_scp = os.path.join(eval_dir, 'utt2spk')
    enroll_spk2xve_scp = os.path.join(enroll_dir, 'spk2xve')

    utt2vec_dict = {}
    assert os.path.exists(utt2vec_scp), print('%s ?' % utt2vec_scp)
    with open(utt2vec_scp, 'r') as f:
        for l in f.readlines():
            uid, xve = l.split()
            utt2vec_dict[uid] = xve

    eval_utt2spk_dict = {}
    assert os.path.exists(utt2spk_scp), print('%s ?' % utt2spk_scp)
    with open(utt2spk_scp, 'r') as f:
        for l in f.readlines():
            uid, sid = l.split()
            eval_utt2spk_dict[uid] = sid

    enroll_spk2xve_dict = {}
    assert os.path.exists(enroll_spk2xve_scp), print('%s ?' % enroll_spk2xve_scp)
    with open(enroll_spk2xve_scp, 'r') as f:
        for l in f.readlines():
            sid, xve = l.split()
            enroll_spk2xve_dict[sid] = xve

    sids = list(enroll_spk2xve_dict.keys())
    uids = list(utt2vec_dict.keys())
    sids.sort()
    uids.sort()

    print("Getting the ground truth for utterraces...")
    real_uid2sid = []
    for uid in uids:
        sid = eval_utt2spk_dict[uid]
        sid_idx = sids.index(sid)
        real_uid2sid.append(sid_idx)

    spks_tensor = torch.tensor([])
    # spk_dur_factor = []
    for sid in sids:
        sid2vec = enroll_spk2xve_dict[sid]
        # vec = torch.tensor(file_loader(sid2vec).mean(axis=0)).unsqueeze(1).float()
        vec = torch.tensor(file_loader(sid2vec)).float()
        # spk_dur_factor.append(vec[0])
        spks_tensor = torch.cat((spks_tensor, vec[1:]), dim=1)

    # spk_dur_factor = torch.tensor(spk_dur_factor)# .clamp_max(1.0)

    uids_tensor = torch.tensor([])
    # dur_factor = []
    for uid in uids:
        num_eval += 1
        uid2vec = utt2vec_dict[uid]
        # vec = torch.tensor(file_loader(uid2vec).mean(axis=0)).unsqueeze(0).float()
        vec = torch.tensor(file_loader(uid2vec)).float()
        uids_tensor = torch.cat((uids_tensor, vec[1:]), dim=0)
        # dur_factor.append(vec[0])

    # dur_factor = torch.tensor(dur_factor)# .clamp(0.85, 1.0)
    print("Normalization and Cosine similarity...")
    uids_tensor = uids_tensor / uids_tensor.norm(p=2, dim=1, keepdim=True)
    spks_tensor = spks_tensor / spks_tensor.norm(p=2, dim=0, keepdim=True)
    spk_pro = torch.matmul(uids_tensor, spks_tensor)
    # spk_pro = torch.mul(dur_factor.unsqueeze(1).expand(spk_pro.shape[0], spk_pro.shape[1]), spk_pro)
    # spk_pro = torch.mul(spk_pro, spk_dur_factor.unsqueeze(0).expand(spk_pro.shape[0], spk_pro.shape[1]), )

    np.save('eval_dir/result.npy', spk_pro.numpy())
    np.save('eval_dir/answer.npy', real_uid2sid)

    # num_gthre = torch.where(spk_pro > 0.4, torch.tensor([1.]), torch.tensor([0.])).sum(dim=1)
    #
    # # print(spk_pro)
    # spk_res = spk_pro.max(dim=1)
    # spk_val = spk_res.values
    # for l in range(spk_val.shape[0]):
    #     if num_gthre[l] >= 3:
    #         spk_val[l] *= 0.85
    #
    #     if num_gthre[l] == 1:
    #         spk_val[l] /= 0.85
    # # print(spk_val)
    # inset_threshold = 0.7
    # spk_lab = torch.where(spk_val > inset_threshold, spk_res.indices, torch.tensor([-1]))
    # # out of set: 0.0-0.5
    # # in set : 0.5-1.0
    #
    # confidence = torch.where(spk_val > inset_threshold, (spk_val-inset_threshold)/(1-inset_threshold)*0.5+0.5, spk_val/inset_threshold*0.5).clamp(min=0.01, max=0.99).tolist()
    #
    # sids.append('unknown')
    # reco_sid = list(np.array(sids)[spk_lab.numpy()])
    #
    # result = [(x, y, z) for x, y, z in zip(uids, reco_sid, confidence)]
    # utt2spk_scp = os.path.join(eval_dir, 'utt2sid')
    # with open(utt2spk_scp, 'w') as f:
    #     for uid, sid, con in result:
    #         f.write(uid + " " + sid + " " + str(con) + '\n')
    # # return num_eval, result


if __name__ == '__main__':
    enroll_dir, test_dir = Split_Set(data_dir=args.data_dir, xvector_dir=args.extract_path, split_set=args.split_set)
    Enroll(enroll_dir=enroll_dir)
    Eval(enroll_dir=enroll_dir, eval_dir=test_dir)
