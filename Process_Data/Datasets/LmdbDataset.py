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
import pdb
import random

import kaldi_io
import lmdb
import numpy as np
from kaldi_io import read_mat
from torch.utils.data import Dataset
import torch
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
    def __init__(self, dir, feat_dim, transform, loader=read_mat, domain=False,
                 random_chunk=[], batch_size=0, label_dir='', verbose=1):

        feat_scp = dir + '/feats.scp'

        if not os.path.exists(feat_scp):
            raise FileExistsError(feat_scp)

        dataset = []
        spks = set([])
        doms = set([])

        with open(feat_scp, 'r') as u:

            all_cls_upath = tqdm(u.readlines()) if verbose > 0 else u.readlines()

            for line in all_cls_upath:
                try:
                    cls, upath = line.split()
                    dom_cls = -1
                except ValueError as v:
                    cls, dom_cls, upath = line.split()
                    dom_cls = int(dom_cls)
                try:
                    cls = int(cls)
                except ValueError as v:
                    pass

                dataset.append((cls, dom_cls, upath))
                doms.add(dom_cls)
                spks.add(cls)

        label_feat_scp = label_dir + '/feat.scp'
        guide_label = []
        if os.path.exists(label_feat_scp):
            with open(label_feat_scp, 'r') as u:
                all_lb_upath = tqdm(u.readlines())
                for line in all_lb_upath:
                    lb, lpath = line.split()
                    guide_label.append((int(lb), lpath))
        if verbose > 0:
            print('==> There are {} speakers in Dataset.'.format(len(spks)))
            print('    There are {} egs in Dataset'.format(len(dataset)))
        if len(guide_label) > 0:
            if verbose > 0:
                print('    There are {} guide labels for egs in Dataset'.format(len(guide_label)))
            assert len(guide_label) == len(dataset)

        self.dataset = dataset
        self.guide_label = guide_label

        self.feat_dim = feat_dim
        self.loader = loader
        self.transform = transform
        self.num_spks = len(spks)
        self.num_doms = len(doms)
        self.domain = domain
        self.chunk_size = []
        self.batch_size = batch_size

    def __getitem__(self, idx):
        # time_s = time.time()
        # print('Starting loading...')
        label, dom_label, upath = self.dataset[idx]

        y = self.loader(upath)

        feature = self.transform(y)
        # time_e = time.time()
        # print('Using %d for loading egs' % (time_e - time_s))

        if len(self.guide_label) > 0:
            _, lpath = self.guide_label[idx]
            guide_label = kaldi_io.read_vec_flt(lpath)
            guide_label = torch.tensor(guide_label, dtype=torch.float32)

            if self.domain:
                return feature, label, dom_label, guide_label
            else:
                return feature, label, guide_label
        if self.domain:
            return feature, label, dom_label
        else:
            return feature, label

    def __getrandomitem__(self):
        # time_s = time.time()
        # print('Starting loading...')
        idx = np.random.randint(low=0, high=self.__len__())
        label, dom_label, upath = self.dataset[idx]

        y = self.loader(upath)
        feature = self.transform(y)

        return feature

    def __len__(self):
        return len(self.dataset)  # 返回一个epoch的采样数


class CrossEgsDataset(Dataset):
    def __init__(self, dir, feat_dim, transform, loader=read_mat, domain=False, num_meta_spks=0,
                 random_chunk=[], batch_size=144, enroll_utt=5, label_dir='', verbose=1):

        feat_scp = dir + '/feats.scp'

        if not os.path.exists(feat_scp):
            raise FileExistsError(feat_scp)

        dataset_len = 0
        spks = set([])
        doms = set([])
        cls2dom2utt = {}

        with open(feat_scp, 'r') as u:

            all_cls_upath = tqdm(u.readlines()) if verbose > 0 else u.readlines()

            for line in all_cls_upath:
                try:
                    cls, upath = line.split()
                    dom_cls = -1
                except ValueError as v:
                    cls, dom_cls, upath = line.split()
                    dom_cls = int(dom_cls)
                try:
                    cls = int(cls)
                except ValueError as v:
                    pass

                dataset_len += 1
                # dataset.append((cls, dom_cls, upath))
                doms.add(dom_cls)
                spks.add(cls)

                cls2dom2utt.setdefault(cls, {})
                cls2dom2utt[cls].setdefault(dom_cls, [])

                cls2dom2utt[cls][dom_cls].append(upath)

        if num_meta_spks > 0:
            spks = list(spks)
            random.shuffle(spks)
            meta_spks = spks[-num_meta_spks:]
            # meta_cls2dom2utt = {}
            spks = spks[:-num_meta_spks]
            #
            # for cls in meta_spks:
            #     meta_cls2dom2utt[cls] = cls2dom2utt.pop(cls)
            #
            # self.meta_cls2dom2utt = meta_cls2dom2utt
            self.meta_spks = meta_spks

        self.dataset = cls2dom2utt
        self.dataset_len = dataset_len
        # self.guide_label = guide_label

        self.feat_dim = feat_dim
        self.enroll_utt = enroll_utt
        self.loader = loader
        self.transform = transform
        self.spks = list(spks)
        self.num_spks = len(spks)
        self.num_doms = len(doms)
        self.domain = domain
        self.chunk_size = []
        self.batch_size = batch_size
        self.batch_spks = int(batch_size / (enroll_utt + 1))

    def __getitem__(self, idx):
        # time_s = time.time()
        # print('Starting loading...')

        batch_spks = set([])
        while len(batch_spks) < self.batch_spks:
            batch_spks.add(random.choice(self.spks))

        # print('Batch_spks: ', self.batch_spks)
        features = []
        label = []
        for spk_idx in batch_spks:
            label.extend([spk_idx] * (self.enroll_utt + 1))
            this_dom2utt = self.dataset[spk_idx].copy()

            test_utt = []
            enroll_utts = set([])

            if len(this_dom2utt) == 1:
                this_spks_utts = this_dom2utt[list(this_dom2utt.keys())[0]]
                test_utt.append(random.choice(this_spks_utts))

                while len(enroll_utts) < self.enroll_utt:
                    rand_enroll_utt = random.choice(this_spks_utts)
                    if rand_enroll_utt not in test_utt:
                        enroll_utts.add(rand_enroll_utt)
            else:
                this_spk_doms = list(this_dom2utt.keys())
                test_dom = random.choice(this_spk_doms)
                enroll_dom = random.choice(this_spk_doms)

                while enroll_dom == test_dom:
                    enroll_dom = random.choice(this_spk_doms)

                test_utt.append(random.choice(this_dom2utt[test_dom]))

                while len(enroll_utts) < self.enroll_utt:
                    enroll_utts.add(random.choice(this_dom2utt[enroll_dom]))

            utts_feat = [self.transform(self.loader(upath)) for upath in test_utt]
            utts_feat.extend([self.transform(self.loader(upath)) for upath in enroll_utts])
            features.append(torch.stack(utts_feat, dim=0))
        # time_e = time.time()
        # print('Using %d for loading egs' % (time_e - time_s))
        # 24, 6, 1, time, feat_dim
        features = torch.stack(features, dim=0).squeeze()
        feat_shape = features.shape
        # print('Features shape: ', feat_shape)

        return features.reshape(feat_shape[0] * feat_shape[1], feat_shape[2], feat_shape[3]), torch.LongTensor(label)

    def __getrandomitem__(self):
        # time_s = time.time()
        # print('Starting loading...')
        idx = np.random.randint(low=0, high=self.__len__())
        label, dom_label, upath = self.dataset[idx]

        y = self.loader(upath)
        feature = self.transform(y)

        return feature

    def __len__(self):
        return int(self.dataset_len / self.batch_size)  # 返回一个epoch的采样数


class CrossValidEgsDataset(Dataset):
    def __init__(self, dir, feat_dim, transform, loader=read_mat, domain=False, num_meta_spks=0,
                 random_chunk=[], batch_size=144, enroll_utt=5, label_dir='', verbose=1):

        feat_scp = dir + '/feats.scp'

        if not os.path.exists(feat_scp):
            raise FileExistsError(feat_scp)

        dataset_len = 0
        spks = set([])
        doms = set([])
        cls2dom2utt = {}

        with open(feat_scp, 'r') as u:

            all_cls_upath = tqdm(u.readlines()) if verbose > 0 else u.readlines()

            for line in all_cls_upath:
                try:
                    cls, upath = line.split()
                    dom_cls = -1
                except ValueError as v:
                    cls, dom_cls, upath = line.split()
                    dom_cls = int(dom_cls)
                try:
                    cls = int(cls)
                except ValueError as v:
                    pass

                dataset_len += 1
                # dataset.append((cls, dom_cls, upath))
                doms.add(dom_cls)
                spks.add(cls)

                cls2dom2utt.setdefault(cls, {})
                cls2dom2utt[cls].setdefault(dom_cls, [])

                cls2dom2utt[cls][dom_cls].append(upath)

        if num_meta_spks > 0:
            spks = list(spks)
            random.shuffle(spks)
            meta_spks = spks[-num_meta_spks:]
            # meta_cls2dom2utt = {}
            spks = spks[:-num_meta_spks]
            #
            # for cls in meta_spks:
            #     meta_cls2dom2utt[cls] = cls2dom2utt.pop(cls)
            #
            # self.meta_cls2dom2utt = meta_cls2dom2utt
            self.meta_spks = meta_spks

        self.dataset = cls2dom2utt
        self.dataset_len = dataset_len
        # self.guide_label = guide_label

        self.feat_dim = feat_dim
        self.enroll_utt = enroll_utt
        self.loader = loader
        self.transform = transform
        self.spks = list(spks)
        self.num_spks = len(spks)
        self.num_doms = len(doms)
        self.domain = domain
        self.chunk_size = []
        self.batch_size = batch_size
        self.batch_spks = min(int(batch_size / (enroll_utt + 1)), len(spks))

    def __getitem__(self, idx):
        # time_s = time.time()
        print('Starting loading...')

        batch_spks = set([])
        while len(batch_spks) < self.batch_spks:
            batch_spks.add(random.choice(self.spks))

        # print('Batch spks: ', self.batch_spks)
        features = []
        label = []
        for spk_idx in batch_spks:
            label.extend([spk_idx] * (self.enroll_utt + 1))
            this_dom2utt = self.dataset[spk_idx].copy()

            test_utt = []
            enroll_utts = set([])

            if len(this_dom2utt) == 1:
                # print('Enroll dom == 1')
                this_spks_utts = this_dom2utt[list(this_dom2utt.keys())[0]]

                if len(this_spks_utts) == 1:
                    continue

                test_utt.append(random.choice(this_spks_utts))

                if len(this_spks_utts) - 1 >= self.enroll_utt:
                    while len(enroll_utts) < self.enroll_utt:
                        enroll_uid = random.choice(this_spks_utts)
                        if enroll_uid not in test_utt:
                            enroll_utts.add(enroll_uid)
                else:
                    for i in this_spks_utts:
                        if i not in test_utt:
                            enroll_utts.add(i)

                    enroll_utts = list(enroll_utts)
                    while len(enroll_utts) < self.enroll_utt:
                        enroll_utts.extend([random.choice(enroll_utts)])

            else:
                print('Enroll dom > 1')
                this_spk_doms = list(this_dom2utt.keys())
                test_dom = random.choice(this_spk_doms)
                enroll_dom = random.choice(this_spk_doms)

                while enroll_dom == test_dom:
                    enroll_dom = random.choice(this_spk_doms)
                print('Enroll dom: ', enroll_dom)

                test_utt.append(random.choice(this_dom2utt[test_dom]))

                while len(enroll_utts) < self.enroll_utt:
                    enroll_utts.add(random.choice(this_dom2utt[enroll_dom]))

            utts_feat = [self.transform(self.loader(upath)) for upath in test_utt]
            utts_feat.extend([self.transform(self.loader(upath)) for upath in enroll_utts])
            features.append(torch.stack(utts_feat, dim=0))
        # time_e = time.time()
        # print('Using %d for loading egs' % (time_e - time_s))
        # 24, 6, 1, time, feat_dim
        features = torch.stack(features, dim=0).squeeze()
        feat_shape = features.shape
        print('Feat_shape: ', feat_shape)

        return features.reshape(feat_shape[0] * feat_shape[1], feat_shape[2], feat_shape[3]), torch.LongTensor(label)

    def __getrandomitem__(self):
        # time_s = time.time()
        # print('Starting loading...')
        idx = np.random.randint(low=0, high=self.__len__())
        label, dom_label, upath = self.dataset[idx]

        y = self.loader(upath)
        feature = self.transform(y)

        return feature

    def __len__(self):
        return int(self.dataset_len / self.batch_size)  # 返回一个epoch的采样数


class CrossMetaEgsDataset(Dataset):
    def __init__(self, dir, feat_dim, transform, spks, loader=read_mat, domain=False,
                 random_chunk=[], batch_size=144, enroll_utt=5, label_dir='', verbose=1):

        feat_scp = dir + '/feats.scp'

        if not os.path.exists(feat_scp):
            raise FileExistsError(feat_scp)

        dataset_len = 0
        self.spks = spks

        doms = set([])
        cls2dom2utt = {}

        with open(feat_scp, 'r') as u:

            all_cls_upath = tqdm(u.readlines()) if verbose > 0 else u.readlines()

            for line in all_cls_upath:
                try:
                    cls, upath = line.split()
                    dom_cls = -1
                except ValueError as v:
                    cls, dom_cls, upath = line.split()
                    dom_cls = int(dom_cls)
                try:
                    cls = int(cls)
                except ValueError as v:
                    pass

                if cls in self.spks:
                    dataset_len += 1
                    # dataset.append((cls, dom_cls, upath))
                    doms.add(dom_cls)
                    # spks.add(cls)

                    cls2dom2utt.setdefault(cls, {})
                    cls2dom2utt[cls].setdefault(dom_cls, [])
                    cls2dom2utt[cls][dom_cls].append(upath)

        self.dataset = cls2dom2utt
        self.dataset_len = int(dataset_len / batch_size)
        # self.guide_label = guide_label

        self.feat_dim = feat_dim
        self.enroll_utt = enroll_utt
        self.loader = loader
        self.transform = transform
        self.spks = list(spks)
        self.num_spks = len(spks)
        self.num_doms = len(doms)
        self.domain = domain
        self.chunk_size = []
        self.batch_size = batch_size
        self.batch_spks = min(int(batch_size / (enroll_utt + 1)), len(spks))

    def __getitem__(self, idx):
        # time_s = time.time()
        # print('Starting loading...')

        batch_spks = set([])
        while len(batch_spks) < self.batch_spks:
            batch_spks.add(random.choice(self.spks))

        # print('Batch spks: ', self.batch_spks)
        features = []
        label = []
        for spk_idx in batch_spks:
            label.extend([spk_idx] * (self.enroll_utt + 1))
            this_dom2utt = self.dataset[spk_idx].copy()

            test_utt = []
            enroll_utts = set([])

            if len(this_dom2utt) == 1:
                this_spks_utts = this_dom2utt[list(this_dom2utt.keys())[0]]
                test_utt.append(random.choice(this_spks_utts))

                while len(enroll_utts) < self.enroll_utt:
                    rand_enroll_utt = random.choice(this_spks_utts)
                    if rand_enroll_utt not in test_utt:
                        enroll_utts.add(rand_enroll_utt)
            else:
                this_spk_doms = list(this_dom2utt.keys())
                test_dom = random.choice(this_spk_doms)
                enroll_dom = random.choice(this_spk_doms)

                while enroll_dom == test_dom:
                    enroll_dom = random.choice(this_spk_doms)

                test_utt.append(random.choice(this_dom2utt[test_dom]))

                while len(enroll_utts) < self.enroll_utt:
                    enroll_utts.add(random.choice(this_dom2utt[enroll_dom]))

            utts_feat = [self.transform(self.loader(upath)) for upath in test_utt]
            utts_feat.extend([self.transform(self.loader(upath)) for upath in enroll_utts])
            features.append(torch.stack(utts_feat, dim=0))
        # time_e = time.time()
        # print('Using %d for loading egs' % (time_e - time_s))
        # 24, 6, 1, time, feat_dim
        features = torch.stack(features, dim=0).squeeze()
        feat_shape = features.shape
        # print('Feat_shape: ', feat_shape)

        return features.reshape(feat_shape[0] * feat_shape[1], feat_shape[2], feat_shape[3]), torch.LongTensor(label)

    def __getrandomitem__(self):
        # time_s = time.time()
        # print('Starting loading...')
        idx = np.random.randint(low=0, high=self.__len__())
        label, dom_label, upath = self.dataset[idx]

        y = self.loader(upath)
        feature = self.transform(y)

        return feature

    def __len__(self):
        return self.dataset_len  # 返回一个epoch的采样数
