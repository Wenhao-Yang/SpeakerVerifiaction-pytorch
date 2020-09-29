#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author: WILLIAM
@file: fratio_dataset.py
@Time: 2020/9/29 
@From: ASUS Win10
@Overview: 
"""
import random
from torch.utils.data import Dataset
import os
from kaldi_io import read_mat
import numpy as np


class SpeakerDataset(Dataset):
    def __init__(self, dir, samples_per_speaker, transform, loader=read_mat,
                 return_uid=False):
        self.return_uid = return_uid

        feat_scp = dir + '/feats.scp'
        spk2utt = dir + '/spk2utt'
        # utt2spk = dir + '/utt2spk'
        utt2num_frames = dir + '/utt2num_frames'
        # utt2dom = dir + '/utt2dom'

        if not os.path.exists(feat_scp):
            raise FileExistsError(feat_scp)
        if not os.path.exists(spk2utt):
            raise FileExistsError(spk2utt)

        invalid_uid = []
        with open(utt2num_frames, 'r') as f:
            for l in f.readlines():
                uid, num_frames = l.split()
                if int(num_frames) < 50:
                    invalid_uid.append(uid)

        dataset = {}
        with open(spk2utt, 'r') as u:
            all_cls = u.readlines()
            for line in all_cls:
                spk_utt = line.split()
                spk_name = spk_utt[0]
                if spk_name not in dataset.keys():
                    dataset[spk_name] = [x for x in spk_utt[1:] if x not in invalid_uid]

        # pdb.set_trace()

        speakers = [spk for spk in dataset.keys()]
        speakers.sort()
        print('==> There are {} speakers in Dataset.'.format(len(speakers)))
        spk_to_idx = {speakers[i]: i for i in range(len(speakers))}
        idx_to_spk = {i: speakers[i] for i in range(len(speakers))}

        uid2feat = {}  # 'Eric_McCormack-Y-qKARMSO7k-0001.wav': feature[frame_length, feat_dim]
        with open(feat_scp, 'r') as f:
            for line in f.readlines():
                uid, feat_offset = line.split()
                if uid in invalid_uid:
                    continue
                uid2feat[uid] = feat_offset

        print('    There are {} utterances in Dataset, where {} utterances are removed.'.format(len(uid2feat),
                                                                                                      len(invalid_uid)))
        self.speakers = speakers
        self.dataset = dataset
        self.uid2feat = uid2feat
        self.spk_to_idx = spk_to_idx
        self.idx_to_spk = idx_to_spk
        self.num_spks = len(speakers)

        self.loader = loader
        self.feat_dim = loader(uid2feat[dataset[speakers[0]][0]]).shape[1]
        self.transform = transform
        self.samples_per_speaker = samples_per_speaker

    def __getitem__(self, sid):
        # start_time = time.time()
        if self.return_uid:
            uid, label = self.utt_dataset[sid]
            y = self.loader(self.uid2feat[uid])
            feature = self.transform(y)
            return feature, label, uid

        spk = self.idx_to_spk[sid]
        utts = self.dataset[spk]
        num_utt = len(utts)

        y = np.array([[]]).reshape(0, self.feat_dim)
        uid = utts[np.random.randint(0, num_utt)]

        feature = self.loader(self.uid2feat[uid])
        y = np.concatenate((y, feature), axis=0)

        while len(y) < self.samples_per_speaker:
            uid = utts[np.random.randint(0, num_utt)]
            feature = self.loader(self.uid2feat[uid])
            y = np.concatenate((y, feature), axis=0)
        # transform features if required
        feature = self.transform(y)
        label = sid

        return feature, label

    def __len__(self):
        return len(self.speakers)  # 返回
