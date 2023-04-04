#!/usr/bin/env python
# encoding: utf-8
'''
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: VS Code
@File: convert2hdf5.py
@Time: 2023/03/09 18:39
@Overview: 
'''


from __future__ import print_function

import argparse
import os
import shutil
import sys
import time

import kaldi_io
import h5py
import soundfile as sf
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Computing Filter banks!')
parser.add_argument('--nj', type=int, default=4, metavar='E', help='number of jobs to make feats (default: 10)')
parser.add_argument('--data-dir', type=str, help='number of jobs to make feats (default: 10)')
parser.add_argument('--data-format', type=str, default='wav', choices=['flac', 'wav'],
                    help='number of jobs to make feats (default: 10)')
parser.add_argument('--out-dir', type=str, required=True, help='number of jobs to make feats (default: 10)')
parser.add_argument('--out-set', type=str, default='dev_reverb', help='number of jobs to make feats (default: 10)')
parser.add_argument('--feat-format', type=str, default='kaldi', choices=['kaldi', 'npy', 'wav'],
                    help='number of jobs to make feats (default: 10)')

parser.add_argument('--feat-type', type=str, default='fbank', choices=['fbank', 'spectrogram', 'mfcc', 'wav'],
                    help='number of jobs to make feats (default: 10)')

parser.add_argument('--conf', type=str, default='condf/spect.conf', metavar='E',
                    help='number of epochs to train (default: 10)')
args = parser.parse_args()

def read_WaveInt(filename, start=0, stop=None):
    """
    read features from npy files
    :param filename: the path of wav files.
    :return:
    """
    # audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    # audio = audio.flatten()
    audio, sample_rate = sf.read(
        filename, dtype='int16', start=start, stop=stop)
    return audio


if __name__ == "__main__":

    nj = args.nj
    data_dir = args.data_dir
    out_dir = os.path.join(args.out_dir, args.out_set)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        # os.removedirs(out_dir)
        print('Remove old dir!')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if args.feat_format == 'kaldi':
        feats_scp_f = os.path.join(data_dir, 'feats.scp')
        feat_loader = kaldi_io.read_mat
    else:
        feats_scp_f = os.path.join(data_dir, 'wav.scp')
        feat_loader = read_WaveInt

    assert os.path.exists(data_dir)
    assert os.path.exists(feats_scp_f)

    if data_dir != out_dir:
        print('Copy wav.scp, spk2utt, utt2spk, trials to %s' % out_dir)
        for f in ['wav.scp', 'spk2utt', 'utt2spk', 'trials','utt2dur',]:
            orig_f = os.path.join(data_dir, f)
            targ_f = os.path.join(out_dir, f)
            if os.path.exists(orig_f):
                os.system('cp %s %s' % (orig_f, targ_f))

    with open(feats_scp_f, 'r') as f:
        feat_scp = f.readlines()
        assert len(feat_scp) > 0

    num_utt = len(feat_scp)
    start_time = time.time()

    h5py_file = os.path.join(out_dir, 'feat.h5py')
    # lmdb_file = os.path.join(out_dir, 'feat')
    # env = lmdb.open(lmdb_file, map_size=1099511627776)  # 1TB
    # txn = env.begin(write=True)
    error_queue = []
    # print('Plan to make feats for %d utterances in %s with %d jobs.' % (num_utt, str(time.asctime()), nj))

    pbar = tqdm(enumerate(feat_scp))
    with h5py.File(h5py_file, 'w') as f:  # 写入的时候是‘w’
        for idx, u in pbar:
            try:
                
                key = u.split()[0]

                if len(u.split())==2:
                    feat_path = u.split()[1]
                else:
                    feat_path = u.split()[4]
                    
                feat = feat_loader(feat_path)
                f.create_dataset(key, data=feat)
            except:
                error_queue.append(key)
            
    if len(error_queue) > 0:
        print('\n>>>> Saving Completed with errors in: ')
        print(error_queue)
    else:
        print('\n>>>> Saving Completed without errors.!')
    end_time = time.time()
    print('All files in: %s, %.2fs collapse.' % (out_dir, end_time - start_time))
    sys.exit()

"""
For multi threads, average making seconds for 47 speakers is 4.579958657
For one threads, average making seconds for 47 speakers is 4.11888732301

For multi process, average making seconds for 47 speakers is 1.67094940328
For one process, average making seconds for 47 speakers is 3.64203325738
"""
