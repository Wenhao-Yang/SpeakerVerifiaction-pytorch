#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: make_egs.py
@Time: 2020/8/21 11:19
@Overview:
"""

from __future__ import print_function

import argparse
import os
import random
import shutil
import sys
import time
import traceback
from multiprocessing import Pool, Manager

import kaldi_io
import numpy as np
import psutil
import torch
from kaldiio import WriteHelper
from tqdm import tqdm

from Process_Data.Datasets.KaldiDataset import ScriptValidDataset, ScriptTrainDataset
from Process_Data.audio_augment.common import RunCommand
from Process_Data.audio_processing import ConcateNumInput
from logger import NewLogger

parser = argparse.ArgumentParser(description='Computing Filter banks!')
parser.add_argument('--nj', type=int, default=8, metavar='E', help='number of jobs to make feats (default: 10)')
parser.add_argument('--data-dir', type=str,
                    help='number of jobs to make feats (default: 10)')
parser.add_argument('--data-format', type=str, default='wav', choices=['flac', 'wav'],
                    help='number of jobs to make feats (default: 10)')
parser.add_argument('--domain', action='store_true', default=False, help='set domain in dataset')

parser.add_argument('--out-dir', type=str, required=True, help='number of jobs to make feats (default: 10)')
parser.add_argument('--out-set', type=str, default='dev_reverb', help='number of jobs to make feats (default: 10)')
parser.add_argument('--feat-format', type=str, choices=['kaldi', 'npy', 'kaldi_cmp'],
                    help='number of jobs to make feats (default: 10)')
parser.add_argument('--out-format', type=str, choices=['kaldi', 'npy', 'kaldi_cmp'],
                    help='number of jobs to make feats (default: 10)')
parser.add_argument('--num-frames', type=int, default=300, metavar='E',
                    help='number of jobs to make feats (default: 10)')

parser.add_argument('--feat-type', type=str, default='fbank', choices=['fbank', 'spectrogram', 'mfcc'],
                    help='number of jobs to make feats (default: 10)')
parser.add_argument('--train', action='store_true', default=False, help='using Cosine similarity')

parser.add_argument('--remove-vad', action='store_true', default=False, help='using Cosine similarity')
parser.add_argument('--compress', action='store_true', default=False, help='using Cosine similarity')
parser.add_argument('--input-per-spks', type=int, default=384, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--num-valid', type=int, default=2, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')
parser.add_argument('--seed', type=int, default=123456, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--conf', type=str, default='condf/spect.conf', metavar='E',
                    help='number of epochs to train (default: 10)')
args = parser.parse_args()

torch.multiprocessing.set_sharing_strategy('file_system')

def PrepareEgProcess(lock_i, lock_t, train_dir, idx_queue, t_queue):
    while True:
        try:
            # print(os.getpid(), " acqing lock i")
            lock_i.acquire()  # 加上锁
            # print(" %d Acqed lock i " % os.getpid(), end='')
            if not idx_queue.empty():
                idx = idx_queue.get()
            else:
                break
        except Exception as e:
            traceback.print_exc(e)
        finally:
            lock_i.release()  # 释放锁

        try:
            if args.domain:
                feature, label, domlab = train_dir.__getitem__(idx)
                pairs = (label, domlab, feature)
            else:
                feature, label = train_dir.__getitem__(idx)
                pairs = (label, feature)

            # lock_t.acquire()
            while t_queue.full():
                print("task queue is full!")
                time.sleep(2)
            # lock_t.acquire()  # 加上锁
            t_queue.put(pairs)
        except Exception as e:
            traceback.print_exc(e)


def SaveEgProcess(lock_t, out_dir, ark_dir, ark_prefix, proid, t_queue, e_queue, i_queue):
    #  wav_scp = os.path.join(data_path, 'wav.scp')
    feat_dir = os.path.join(ark_dir, ark_prefix)
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)

    feat_scp = os.path.join(out_dir, 'feat.%d.temp.scp' % proid)
    feat_ark = os.path.join(feat_dir, 'feat.%d.ark' % proid)
    feat_scp_f = open(feat_scp, 'w')

    if args.out_format == 'kaldi':
        feat_ark_f = open(feat_ark, 'wb')
    if args.out_format == 'kaldi_cmp':
        writer = WriteHelper('ark,scp:%s,%s' % (feat_ark, feat_scp), compression_method=1)

    temp_dir = out_dir + '/temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    saved_egs = 0

    while True:
        # print(os.getpid(), "acq lock t")
        lock_t.acquire()  # 加上锁
        # print(os.getpid(), "acqed lock t", end='')
        if not t_queue.empty():
            comm = t_queue.get()
            lock_t.release()  # 释放锁
            # print(os.getpid(), " real lock t")

            try:
                if args.domain:
                    key = ' '.join((str(comm[0]), str(comm[1])))
                else:
                    key = str(comm[0])
                feat = comm[-1].astype(np.float32).squeeze()
                # print(feat.shape)

                if args.out_format == 'kaldi':
                    kaldi_io.write_mat(feat_ark_f, feat, key='')
                    offsets = feat_ark + ':' + str(feat_ark_f.tell() - len(feat.tobytes()) - 15)
                    # print(offsets)
                    feat_scp_f.write(key + ' ' + offsets + '\n')

                elif args.out_format == 'kaldi_cmp':
                    writer(str(key), feat)

                elif args.feat_format == 'npy':
                    npy_path = os.path.join(feat_dir, '%s.npy' % key)
                    np.save(npy_path, feat)
                    feat_scp_f.write(key + ' ' + npy_path + '\n')

                del comm, feat
                saved_egs += 1

            except Exception as e:
                print(e)
                e_queue.put(key)

            # if saved_egs.qsize() % 1 == 0:
            # if saved_egs % 10 == 0:
            print('\rProcess [{:8>s}]:  [{:>8d}] idx Left and [{:>6d}] egs Left, with [{:>6d}] errors.'.format
                  (str(os.getpid()), i_queue.qsize(), t_queue.qsize(), e_queue.qsize()), end='')

            # if saved_egs % 2000 == 0:
            #     feat_scp_f.flush()
            #     feat_ark_f.flush()

        elif not i_queue.empty():
            lock_t.release()
            # print(os.getpid(), " real lock t")
            time.sleep(20)

        else:
            lock_t.release()
            # print('\n>> Process {}: all queue empty!'.format(os.getpid()))
            break

    feat_scp_f.close()

    if args.out_format == 'kaldi':
        feat_ark_f.close()

    elif args.out_format == 'kaldi_cmp':
        writer.close()

    new_feat_scp = os.path.join(out_dir, 'feat.%d.scp' % proid)
    # print('new feat.scp is : ', new_feat_scp)
    if args.feat_format == 'kaldi' and args.compress == True:
        new_feat_ark = os.path.join(feat_dir, 'feat.%d.ark' % proid)
        compress_command = "copy-feats --compress=true scp:{} ark,scp:{},{}".format(feat_scp, new_feat_ark,
                                                                                    new_feat_scp)
        pid, stdout, stderr = RunCommand(compress_command)
        # print(stdout)
        if os.path.exists(new_feat_scp) and os.path.exists(new_feat_ark):
            os.remove(feat_ark)
    else:
        # print('Cp %s to %s' % (feat_scp, new_feat_scp))
        shutil.copy(feat_scp, new_feat_scp)
        # pass
    assert os.path.exists(new_feat_scp)


transform = ConcateNumInput(num_frames=args.num_frames, remove_vad=args.remove_vad)

if args.feat_format == 'npy':
    file_loader = np.load
elif args.feat_format == 'kaldi':
    file_loader = kaldi_io.read_mat

train_dir = ScriptTrainDataset(dir=args.data_dir, samples_per_speaker=args.input_per_spks, loader=file_loader,
                               transform=transform, num_valid=args.num_valid, domain=args.domain)
# train_dir = LoadScriptDataset(dir=args.data_dir, samples_per_speaker=args.input_per_spks, loader=file_loader,
#                                transform=transform, num_valid=args.num_valid, domain=args.domain)

valid_dir = ScriptValidDataset(valid_set=train_dir.valid_set, loader=file_loader, spk_to_idx=train_dir.spk_to_idx,
                               dom_to_idx=train_dir.dom_to_idx, valid_utt2dom_dict=train_dir.valid_utt2dom_dict,
                               valid_uid2feat=train_dir.valid_uid2feat, valid_utt2spk_dict=train_dir.valid_utt2spk_dict,
                               transform=transform, domain=args.domain)

if __name__ == "__main__":
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    nj = args.nj
    data_dir = args.data_dir
    out_dir = os.path.join(args.out_dir, args.out_set)

    sys.stdout = NewLogger(
        os.path.join(out_dir, 'log', 'egs.%s.conf' % time.strftime("%Y.%m.%d", time.localtime())))

    print('\nCurrent time is \33[91m{}\33[0m.'.format(str(time.asctime())))
    opts = vars(args)
    keys = list(opts.keys())
    keys.sort()

    options = []
    for k in keys:
        options.append("\'%s\': \'%s\'" % (str(k), str(opts[k])))

    print('Preparing egs options: \n{ %s }' % (', '.join(options)))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    wav_scp_f = os.path.join(data_dir, 'wav.scp')
    spk2utt_f = os.path.join(data_dir, 'spk2utt')
    assert os.path.exists(data_dir), data_dir
    assert os.path.exists(wav_scp_f), wav_scp_f
    assert os.path.exists(spk2utt_f), spk2utt_f

    if data_dir != out_dir:
        print('Copy wav.scp, spk2utt, utt2spk, trials to %s' % out_dir)
        for f in ['wav.scp', 'spk2utt', 'utt2spk', 'trials']:
            orig_f = os.path.join(data_dir, f)
            targ_f = os.path.join(out_dir, f)
            if os.path.exists(orig_f):
                os.system('cp %s %s' % (orig_f, targ_f))

    start_time = time.time()

    manager = Manager()
    lock_i = manager.Lock()
    lock_t = manager.Lock()

    feat_dim = train_dir.__getitem__(1)[0].shape[-1]
    mem_data = psutil.virtual_memory()
    free_mem = mem_data.available
    maxsize = int(free_mem / (args.num_frames * feat_dim * 4) * 0.5)
    print('Maxsize for Queue is %d' % maxsize)

    task_queue = manager.Queue(maxsize=maxsize)
    idx_queue = manager.Queue()
    error_queue = manager.Queue()
    prep_jb = 3
    if args.train:

        utts = [i for i in range(len(train_dir))]

        random.seed(args.seed)
        random.shuffle(utts)
        for i in tqdm(utts):
            idx_queue.put(i)

        num_utt = len(utts)

        print('\n>> Plan to make egs for %d speakers with %d egs in %s with %d jobs.\n' % (
            train_dir.num_spks, len(utts), str(time.asctime()), nj))

        pool = Pool(processes=int(nj))  # 创建nj个进程
        for i in range(0, nj):
            write_dir = os.path.join(out_dir, 'Split%d/%d' % (nj, i))
            if not os.path.exists(write_dir):
                os.makedirs(write_dir)

            ark_dir = os.path.join(args.out_dir, args.feat_type)
            if not os.path.exists(ark_dir):
                os.makedirs(ark_dir)
            if (i + 1) % prep_jb != 0:
                pool.apply_async(PrepareEgProcess, args=(lock_i, lock_t, train_dir, idx_queue, task_queue))
                # (lock_i, lock_t, train_dir, idx_queue, t_queue)
            else:
                pool.apply_async(SaveEgProcess, args=(lock_t, write_dir, ark_dir, args.out_set,
                                                      i, task_queue, error_queue, idx_queue))
                # lock_t, out_dir, ark_dir, ark_prefix, proid, t_queue, e_queue, i_queu

    else:

        # valid set
        num_utt = len(valid_dir)
        for i in tqdm(range(len(valid_dir))):
            idx_queue.put(i)

        print('\n>> Plan to make feats for %d speakers with %d egs in %s with %d jobs.\n' % (
            idx_queue.qsize(), num_utt, str(time.asctime()), nj))

        pool = Pool(processes=int(nj * 1.5))  # 创建nj个进程
        for i in range(0, nj):
            write_dir = os.path.join(out_dir, 'Split%d/%d' % (nj, i))
            if not os.path.exists(write_dir):
                os.makedirs(write_dir)

            ark_dir = os.path.join(args.out_dir, args.feat_type)
            if not os.path.exists(ark_dir):
                os.makedirs(ark_dir)

            # if (i + 1) % prep_jb != 1:
            pool.apply_async(PrepareEgProcess, args=(lock_i, lock_t, valid_dir, idx_queue, task_queue))
            if (i + 1) % 2 == 1:
                pool.apply_async(SaveEgProcess, args=(lock_t, write_dir, ark_dir, args.out_set,
                                                      i, task_queue, error_queue, idx_queue))

    pool.close()  # 关闭进程池，表示不能在往进程池中添加进程
    try:
        pool.join()  # 等待进程池中的所有进程执行完毕，必须在close()之后调用
    except:
        traceback.print_exc()

    try:
        if error_queue.qsize() > 0:
            print('\n>> Saving Completed with errors in: ')
            while not error_queue.empty():
                print(error_queue.get() + ' ', end='')
            print('')
        else:
            print('\n>> Saving Completed without errors.!')
    except Exception as e:
        print(e)

    Split_dir = os.path.join(out_dir, 'Split%d' % nj)
    print('   Splited Data root is \n     %s. \n   Concat all scripts together.' % str(Split_dir))

    all_scp_path = [os.path.join(Split_dir, '%d/feat.%d.scp' % (i, i)) for i in range(nj)]
    assert len(all_scp_path) > 0, print(Split_dir)
    feat_scp = os.path.join(out_dir, 'feats.scp')
    numofutt = 0
    with open(feat_scp, 'w') as feat_scp_f:
        for item in all_scp_path:
            if not os.path.exists(item):
                continue
            for txt in open(item, 'r').readlines():
                feat_scp_f.write(txt)
                numofutt += 1
    if numofutt != num_utt:
        print('Errors in %s ?' % feat_scp)

    # print('Delete tmp files in: %s' % Split_dir)

    end_time = time.time()
    all_time = end_time - start_time
    hours = int(all_time / 3600)
    mins = int(all_time % 3600 // 60)
    secs = int(all_time % 60)

    print('Write all files in: \n\t{:s}. \nAnd {:0>2d}:{:0>2d}:{:0>2d}s collapse.\n'.format(out_dir, hours, mins, secs))
    sys.exit()

"""

"""
