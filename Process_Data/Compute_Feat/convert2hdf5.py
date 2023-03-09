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
import pdb
import shutil
import sys
import time
from multiprocessing import Pool, Manager
import traceback
import kaldi_io
import h5py
import soundfile as sf
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Computing Filter banks!')
parser.add_argument('--nj', type=int, default=6, metavar='E', help='number of jobs to make feats (default: 10)')
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


def Load_Process(lock_i, lock_w, f, i_queue, error_queue, feat_loader):
    print('Process {} Start'.format(str(os.getpid())))
    
    while True:
        # print(os.getpid(), " acqing lock i")
        lock_i.acquire()  # 加上锁
        # print(" %d Acqed lock i " % os.getpid(), end='')
        if not i_queue.empty():
            key, feat_path = i_queue.get()
            lock_i.release()
        else:
            lock_i.release()
            break
        
        lock_w.acquire()
        try:
            feat = feat_loader(feat_path)
            f.create_dataset(key, data=feat)
        except:
            error_queue.append(key)
        lock_w.release()

        print('\rProcess [{:8>s}]: [{:>8d}] samples Left'.format
              (str(os.getpid()), i_queue.qsize()), end='')
        

if __name__ == "__main__":

    nj = args.nj
    data_dir = args.data_dir
    out_dir = os.path.join(args.out_dir, args.out_set)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
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
        for f in ['wav.scp', 'spk2utt', 'utt2spk', 'trials']:
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

    # print('Plan to make feats for %d utterances in %s with %d jobs.' % (num_utt, str(time.asctime()), nj))
    manager = Manager()
    read_lock = manager.Lock()
    write_lock = manager.Lock()
    read_queue = manager.Queue()
    error_queue = manager.Queue()

    pbar = tqdm(enumerate(feat_scp), ncols=100)
    for idx, u in pbar:
        key, feat_path = u.split()
        read_queue.put((key, feat_path))

        if idx == 1000:
            break
    
    pdb.set_trace()

    with h5py.File(h5py_file, 'w') as f:  # 写入的时候是‘w’
        pool = Pool(processes=int(nj))  # 创建nj个进程
        for i in range(0, nj):
            pool.apply_async(Load_Process, args=(read_lock, write_lock, f, read_queue, error_queue, feat_loader))
    
        pool.close()  # 关闭进程池，表示不能在往进程池中添加进程
        try:
            pool.join()  # 等待进程池中的所有进程执行完毕，必须在close()之后调用
        except:
            traceback.print_exc()
        
        if error_queue.qsize() > 0:
            print('\n>>>> Saving Completed with errors in: ')
            while not error_queue.empty():
                    print(error_queue.get() + ' ', end='')
            print('')
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
