#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: LmdbDataset.py
@Time: 2020/8/20 16:55
@Overview:
"""
import os
import random
import time

import lmdb
import numpy as np
from kaldi_io import read_mat
from torch.utils.data import Dataset
from tqdm import tqdm

import Process_Data.constants as c


def _read_data_lmdb(txn, key, size):
    """read data array from lmdb with key (w/ and w/o fixed size)
    size: feat-dim"""
    # with env.begin(write=False) as txn:
    buf = txn.get(key.encode('ascii'))
    data_flat = np.frombuffer(buf, dtype=np.float32)

    return data_flat.reshape(int(data_flat.shape[0] / size), size)


class LmdbVerifyDataset(Dataset):
    def __init__(self, dir, xvectors_dir, trials_file='trials', loader=np.load, return_uid=False):

        feat_scp = xvectors_dir + '/xvectors.scp'
        trials = dir + '/%s' % trials_file

        if not os.path.exists(feat_scp):
            raise FileExistsError(feat_scp)
        if not os.path.exists(trials):
            raise FileExistsError(trials)

        uid2feat = {}
        with open(feat_scp, 'r') as f:
            for line in f.readlines():
                uid, feat_offset = line.split()
                uid2feat[uid] = feat_offset

        print('\n==> There are {} utterances in Verification trials.'.format(len(uid2feat)))

        trials_pair = []
        positive_pairs = 0
        with open(trials, 'r') as t:
            all_pairs = t.readlines()
            for line in all_pairs:
                pair = line.split()
                if pair[2] == 'nontarget' or pair[2] == '0':
                    pair_true = False
                else:
                    pair_true = True
                    positive_pairs += 1

                trials_pair.append((pair[0], pair[1], pair_true))

        trials_pair = np.array(trials_pair)
        trials_pair = trials_pair[trials_pair[:, 2].argsort()[::-1]]

        print('    There are {} pairs in trials with {} positive pairs'.format(len(trials_pair),
                                                                               positive_pairs))

        self.uid2feat = uid2feat
        self.trials_pair = trials_pair
        self.numofpositive = positive_pairs

        self.loader = loader
        self.return_uid = return_uid

    def __getitem__(self, index):
        uid_a, uid_b, label = self.trials_pair[index]

        feat_a = self.uid2feat[uid_a]
        feat_b = self.uid2feat[uid_b]
        data_a = self.loader(feat_a)
        data_b = self.loader(feat_b)

        if label == 'True' or label == True:
            label = True
        else:
            label = False

        if self.return_uid:
            # pdb.set_trace()
            # print(uid_a, uid_b)
            return data_a, data_b, label, uid_a, uid_b

        return data_a, data_b, label

    def partition(self, num):
        if num > len(self.trials_pair):
            print('%d is greater than the total number of pairs')

        elif num * 0.3 > self.numofpositive:
            indices = list(range(self.numofpositive, len(self.trials_pair)))
            random.shuffle(indices)
            indices = indices[:(num - self.numofpositive)]
            positive_idx = list(range(self.numofpositive))

            positive_pairs = self.trials_pair[positive_idx].copy()
            nagative_pairs = self.trials_pair[indices].copy()

            self.trials_pair = np.concatenate((positive_pairs, nagative_pairs), axis=0)
        else:
            indices = list(range(self.numofpositive, len(self.trials_pair)))
            random.shuffle(indices)
            indices = indices[:(num - int(0.3 * num))]

            positive_idx = list(range(self.numofpositive))
            random.shuffle(positive_idx)
            positive_idx = positive_idx[:int(0.3 * num)]
            positive_pairs = self.trials_pair[positive_idx].copy()
            nagative_pairs = self.trials_pair[indices].copy()

            self.numofpositive = len(positive_pairs)
            self.trials_pair = np.concatenate((positive_pairs, nagative_pairs), axis=0)

        assert len(self.trials_pair) == num
        num_positive = 0
        for x, y, z in self.trials_pair:
            if z == 'True':
                num_positive += 1

        assert len(self.trials_pair) == num, '%d != %d' % (len(self.trials_pair), num)
        assert self.numofpositive == num_positive, '%d != %d' % (self.numofpositive, num_positive)
        print('%d positive pairs remain.' % num_positive)

    def __len__(self):
        return len(self.trials_pair)


class LmdbTrainDataset(Dataset):
    def __init__(self, dir, feat_dim, samples_per_speaker, transform, loader=_read_data_lmdb, num_valid=5,
                 return_uid=False):

        # feat_scp = dir + '/feats.scp'
        spk2utt = dir + '/spk2utt'
        utt2spk = dir + '/utt2spk'
        # utt2num_frames = dir + '/utt2num_frames'
        lmdb_file = dir + '/feat'

        if not os.path.exists(lmdb_file):
            raise FileExistsError(lmdb_file)
        if not os.path.exists(spk2utt):
            raise FileExistsError(spk2utt)

        dataset = {}
        with open(spk2utt, 'r') as u:
            all_cls = u.readlines()
            for line in all_cls:
                spk_utt = line.split()
                spk_name = spk_utt[0]
                if spk_name not in dataset.keys():
                    dataset[spk_name] = [x for x in spk_utt[1:]]
                    # dataset[spk_name] = [x for x in spk_utt[1:] if x not in invalid_uid]

        utt2spk_dict = {}
        with open(utt2spk, 'r') as u:
            all_cls = u.readlines()
            for line in all_cls:
                utt_spk = line.split()
                uid = utt_spk[0]
                # if uid in invalid_uid:
                #     continue
                if uid not in utt2spk_dict.keys():
                    utt2spk_dict[uid] = utt_spk[-1]
        # pdb.set_trace()

        speakers = [spk for spk in dataset.keys()]
        speakers.sort()
        print('==> There are {} speakers in Dataset.'.format(len(speakers)))
        spk_to_idx = {speakers[i]: i for i in range(len(speakers))}
        idx_to_spk = {i: speakers[i] for i in range(len(speakers))}

        print('    There are {} utterances in Train Dataset'.format(len(utt2spk_dict.keys())))
        if num_valid > 0:
            valid_set = {}
            valid_utt2spk_dict = {}

            for spk in speakers:
                if spk not in valid_set.keys():
                    valid_set[spk] = []
                    for i in range(num_valid):
                        if len(dataset[spk]) <= 1:
                            break
                        j = np.random.randint(len(dataset[spk]))
                        utt = dataset[spk].pop(j)
                        valid_set[spk].append(utt)

                        valid_utt2spk_dict[utt] = utt2spk_dict[utt]

            print('    Spliting {} utterances for Validation.'.format(len(valid_utt2spk_dict.keys())))
            self.valid_set = valid_set
            self.valid_utt2spk_dict = valid_utt2spk_dict

        self.all_utts = list(utt2spk_dict.keys())
        # for uid in uid2feat.keys():
        #     for i in range(int(np.ceil(utt2len_dict[uid] / c.NUM_FRAMES_SPECT))):
        #         self.all_utts.append(uid)
        env = lmdb.open(lmdb_file, readonly=True, lock=False, readahead=False,
                        meminit=False)
        self.env = env.begin(write=False, buffers=True)  # as txn:
        self.speakers = speakers
        self.dataset = dataset

        self.spk_to_idx = spk_to_idx
        self.idx_to_spk = idx_to_spk
        self.num_spks = len(speakers)
        self.utt2spk_dict = utt2spk_dict

        self.feat_dim = feat_dim
        self.loader = loader
        self.transform = transform
        self.samples_per_speaker = samples_per_speaker
        self.return_uid = return_uid

        # if self.return_uid:
        #     self.utt_dataset = []
        #     for i in range(self.samples_per_speaker * self.num_spks):
        #         sid = i % self.num_spks
        #         spk = self.idx_to_spk[sid]
        #         utts = self.dataset[spk]
        #         uid = utts[random.randrange(0, len(utts))]
        #         self.utt_dataset.append([uid, sid])

    def __getitem__(self, sid):
        # start_time = time.time()
        # if self.return_uid:
        #     uid, label = self.utt_dataset[sid]
        #     y = self.loader(self.uid2feat[uid])
        #     feature = self.transform(y)
        #     return feature, label, uid

        sid %= self.num_spks
        spk = self.idx_to_spk[sid]
        utts = self.dataset[spk]

        y = np.array([[]]).reshape(0, self.feat_dim)

        while len(y) < c.N_SAMPLES:
            uid = random.randrange(0, len(utts))

            feature = self.loader(self.env, utts[uid], self.feat_dim)
            y = np.concatenate((y, feature), axis=0)

        feature = self.transform(y)
        # print(sid)
        label = sid

        return feature, label

    def __len__(self):
        return self.samples_per_speaker * len(self.speakers)  # 返回一个epoch的采样数


class LmdbValidDataset(Dataset):
    def __init__(self, valid_set, spk_to_idx, env, valid_utt2spk_dict, transform, feat_dim, loader=_read_data_lmdb,
                 return_uid=False):
        self.env = env
        self.feat_dim = feat_dim

        speakers = [spk for spk in valid_set.keys()]
        speakers.sort()
        self.speakers = speakers

        self.valid_set = valid_set

        uids = list(valid_utt2spk_dict.keys())
        uids.sort()
        print(uids[:10])
        self.uids = uids
        self.utt2spk_dict = valid_utt2spk_dict
        self.spk_to_idx = spk_to_idx
        self.num_spks = len(speakers)

        self.loader = loader
        self.transform = transform
        self.return_uid = return_uid

    def __getitem__(self, index):
        uid = self.uids[index]
        spk = self.utt2spk_dict[uid]
        y = self.loader(self.env, uid, self.feat_dim)

        feature = self.transform(y)
        label = self.spk_to_idx[spk]

        if self.return_uid:
            return feature, label, uid

        return feature, label

    def __len__(self):
        return len(self.uids)


class LmdbTestDataset(Dataset):
    def __init__(self, dir, transform, feat_dim, loader=_read_data_lmdb, return_uid=False):

        lmdb_file = dir + '/feat'
        spk2utt = dir + '/spk2utt'
        trials = dir + '/trials'

        if not os.path.exists(lmdb_file):
            raise FileExistsError(lmdb_file)
        if not os.path.exists(spk2utt):
            raise FileExistsError(spk2utt)
        if not os.path.exists(trials):
            raise FileExistsError(trials)

        dataset = {}
        with open(spk2utt, 'r') as u:
            all_cls = u.readlines()
            for line in all_cls:
                spk_utt = line.split(' ')
                spk_name = spk_utt[0]
                if spk_name not in dataset.keys():
                    spk_utt[-1] = spk_utt[-1].rstrip('\n')
                    dataset[spk_name] = spk_utt[1:]

        speakers = [spk for spk in dataset.keys()]
        speakers.sort()
        print('    There are {} speakers in Test Dataset.'.format(len(speakers)))

        trials_pair = []
        positive_pairs = 0
        with open(trials, 'r') as t:
            all_pairs = t.readlines()
            for line in all_pairs:
                pair = line.split()
                if pair[2] == 'nontarget' or pair[2] == '0':
                    pair_true = False
                else:
                    pair_true = True
                    positive_pairs += 1

                trials_pair.append((pair[0], pair[1], pair_true))
        trials_pair = np.array(trials_pair)
        trials_pair = trials_pair[trials_pair[:, 2].argsort()[::-1]]

        print('==>There are {} pairs in test Dataset with {} positive pairs'.format(len(trials_pair), positive_pairs))

        env = lmdb.open(lmdb_file, readonly=True, lock=False, readahead=False,
                        meminit=False)
        self.env = env.begin(write=False, buffers=True)
        self.feat_dim = feat_dim
        self.speakers = speakers
        self.trials_pair = trials_pair
        self.num_spks = len(speakers)
        self.numofpositive = positive_pairs

        self.loader = loader
        self.transform = transform
        self.return_uid = return_uid

    def __getitem__(self, index):
        uid_a, uid_b, label = self.trials_pair[index]

        y_a = self.loader(self.env, uid_a, self.feat_dim)
        y_b = self.loader(self.env, uid_b, self.feat_dim)

        data_a = self.transform(y_a)
        data_b = self.transform(y_b)

        if label == 'True' or label == True:
            label = True
        else:
            label = False

        if self.return_uid:
            # pdb.set_trace()
            # print(uid_a, uid_b)
            return data_a, data_b, label, uid_a, uid_b

        return data_a, data_b, label

    def partition(self, num):
        if num > len(self.trials_pair):
            print('%d is greater than the total number of pairs')

        elif num * 0.3 > self.numofpositive:
            indices = list(range(self.numofpositive, len(self.trials_pair)))
            random.shuffle(indices)
            indices = indices[:(num - self.numofpositive)]
            positive_idx = list(range(self.numofpositive))

            positive_pairs = self.trials_pair[positive_idx].copy()
            nagative_pairs = self.trials_pair[indices].copy()

            self.trials_pair = np.concatenate((positive_pairs, nagative_pairs), axis=0)
        else:
            indices = list(range(self.numofpositive, len(self.trials_pair)))
            random.shuffle(indices)
            indices = indices[:(num - int(0.3 * num))]

            positive_idx = list(range(self.numofpositive))
            random.shuffle(positive_idx)
            positive_idx = positive_idx[:int(0.3 * num)]
            positive_pairs = self.trials_pair[positive_idx].copy()
            nagative_pairs = self.trials_pair[indices].copy()

            self.numofpositive = len(positive_pairs)
            self.trials_pair = np.concatenate((positive_pairs, nagative_pairs), axis=0)

        assert len(self.trials_pair) == num
        num_positive = 0
        for x, y, z in self.trials_pair:
            if z == 'True':
                num_positive += 1

        assert len(self.trials_pair) == num, '%d != %d' % (len(self.trials_pair), num)
        assert self.numofpositive == num_positive, '%d != %d' % (self.numofpositive, num_positive)
        print('    %d positive pairs remain.' % num_positive)

    def __len__(self):
        return len(self.trials_pair)


class EgsDataset(Dataset):
    def __init__(self, dir, feat_dim, transform, loader=read_mat, domain=False, random_chunk=[], batch_size=0):

        feat_scp = dir + '/feats.scp'

        if not os.path.exists(feat_scp):
            raise FileExistsError(feat_scp)

        dataset = []
        spks = set([])
        doms = set([])

        with open(feat_scp, 'r') as u:
            all_cls_upath = tqdm(u.readlines())
            for line in all_cls_upath:
                try:
                    cls, upath = line.split()
                    dom_cls = -1
                except ValueError as v:
                    cls, dom_cls, upath = line.split()
                    dom_cls = int(dom_cls)

                cls = int(cls)

                dataset.append((cls, dom_cls, upath))
                doms.add(dom_cls)
                spks.add(cls)

        print('==> There are {} speaker with {} utterances speakers in Dataset.'.format(len(spks), len(dataset)))

        self.dataset = dataset
        self.feat_dim = feat_dim
        self.loader = loader
        self.transform = transform
        self.num_spks = len(spks)
        self.num_doms = len(doms)
        self.domain = domain
        self.chunk_size = []
        self.batch_size = batch_size

        if len(random_chunk) == 2 and batch_size > 0:
            print('==> Generating random length...')
            num_batch = int(np.ceil(len(dataset) / batch_size))
            for i in range(num_batch):
                random_size = np.random.randint(low=random_chunk[0], high=random_chunk[1])
                self.chunk_size.append(random_size)

    def __getitem__(self, idx):
        time_s = time.time()
        print('Starting loading...')
        label, dom_label, upath = self.dataset[idx]

        y = self.loader(upath)
        if len(self.chunk_size) > 0:
            bat_idx = idx // self.batch_size
            this_len = self.chunk_size[bat_idx]
            start = np.random.randint(0, len(y) - this_len)
            # print('This batch len is %d' % this_len)
            y = y[start:(start + this_len)]

        feature = self.transform(y)
        time_e = time.time()
        print('Using %d for loading egs' % (time_e - time_s))
        if self.domain:
            return feature, label, dom_label
        else:
            return feature, label

    def __len__(self):
        return len(self.dataset)  # 返回一个epoch的采样数
