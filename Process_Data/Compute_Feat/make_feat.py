#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: make_feat.py
@Time: 2020/4/1 11:25 AM
@Overview:
"""

from __future__ import print_function

import argparse
import os
import shutil
import sys
import time
import traceback
from multiprocessing import Pool, Manager

import kaldi_io
import numpy as np
from kaldiio import WriteHelper

from Process_Data.audio_augment.common import RunCommand
from Process_Data.audio_processing import Make_Fbank, Make_Spect, Make_MFCC
from logger import NewLogger

parser = argparse.ArgumentParser(description='Computing Filter banks!')
parser.add_argument('--nj', type=int, default=16, metavar='E', help='number of jobs to make feats (default: 10)')
parser.add_argument('--data-dir', type=str, help='number of jobs to make feats (default: 10)')
parser.add_argument('--data-format', type=str, default='wav', choices=['flac', 'wav'],
                    help='number of jobs to make feats (default: 10)')
parser.add_argument('--out-dir', type=str, required=True, help='number of jobs to make feats (default: 10)')
parser.add_argument('--out-set', type=str, default='dev_reverb', help='number of jobs to make feats (default: 10)')
parser.add_argument('--feat-format', type=str, required=True, choices=['kaldi', 'npy', 'kaldi_cmp'],
                    help='number of jobs to make feats (default: 10)')
parser.add_argument('--feat-type', type=str, default='fbank', choices=['fbank', 'spectrogram', 'mfcc'],
                    help='number of jobs to make feats (default: 10)')

parser.add_argument('--log-scale', action='store_true', default=False, help='log power spectogram')
parser.add_argument('--energy', action='store_true', default=False, help='log power spectogram')

parser.add_argument('--filter-type', type=str, default='mel', help='number of jobs to make feats (default: 10)')

parser.add_argument('--filters', type=int, help='number of jobs to make feats (default: 10)')
parser.add_argument('--multi-weight', action='store_true', default=False, help='using Cosine similarity')
parser.add_argument('--numcep', type=int, default=24, help='number of cepstrum bin to make feats (default: 24)')
parser.add_argument('--windowsize', type=float, default=0.02, choices=[0.02, 0.025],
                    help='number of jobs to make feats (default: 10)')
parser.add_argument('--stride', type=float, default=0.01, help='number of jobs to make feats (default: 10)')

parser.add_argument('--bandpass', action='store_true', default=False, help='using butter bandpass filter for signal')
parser.add_argument('--lowfreq', type=int, default=0, help='number of jobs to make feats (default: 10)')
parser.add_argument('--highfreq', type=int, default=0, help='number of jobs to make feats (default: 10)')
parser.add_argument('--nfft', type=int, required=True, help='number of jobs to make feats (default: 10)')
parser.add_argument('--normalize', action='store_true', default=False, help='using Cosine similarity')
parser.add_argument('--compress', action='store_true', default=False, help='using Cosine similarity')

parser.add_argument('--conf', type=str, default='condf/spect.conf', metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--vad-proportion-threshold', type=float, default=0.12, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--vad-frames-context', type=int, default=2, metavar='E',
                    help='number of epochs to train (default: 10)')
args = parser.parse_args()

def MakeFeatsProcess(lock, out_dir, ark_dir, ark_prefix, proid, t_queue, e_queue):
    #  wav_scp = os.path.join(data_path, 'wav.scp')
    feat_scp = os.path.join(out_dir, 'feat.%d.temp.scp' % proid)

    utt2dur = os.path.join(out_dir, 'utt2dur.%d' % proid)
    utt2num_frames = os.path.join(out_dir, 'utt2num_frames.%d' % proid)

    feat_scp_f = open(feat_scp, 'w')
    utt2dur_f = open(utt2dur, 'w')

    utt2num_frames_f = open(utt2num_frames, 'w')
    feat_dir = os.path.join(ark_dir, ark_prefix)
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)

    if args.feat_format == 'kaldi':
        feat_ark = os.path.join(feat_dir, 'feat.%d.ark' % proid)
        feat_ark_f = open(feat_ark, 'wb')
    elif args.feat_format == 'kaldi_cmp':
        feat_scp_f.close()  # kaldiio
        feat_ark = os.path.join(out_dir, '%s_feat.%d.ark' % (ark_prefix, proid))
        writer = WriteHelper('ark,scp:%s,%s' % (feat_ark, feat_scp), compression_method=1)

    temp_dir = out_dir + '/temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    while True:
        lock.acquire()  # 加上锁
        if not t_queue.empty():
            comms = t_queue.get()
            lock.release()  # 释放锁

            for comm in comms:
                pair = comm.split()
                key = pair[0]
                try:
                    if len(pair) > 2:
                        command = ' '.join(pair[1:])
                        if command.endswith('|'):
                            command = command.rstrip('|')

                        temp_wav = temp_dir + '/%s.%s' % (key, args.data_format)
                        command = command.rstrip('-') + ' %s'.format(temp_wav)
                        spid, stdout, error = RunCommand(command)
                        # os.waitpid(spid, 0)

                        # with open(temp_wav, 'wb') as wav_f:
                        #     wav_f.write(stdout)
                        if args.feat_type == 'fbank':
                            feat, duration = Make_Fbank(filename=temp_wav, filtertype=args.filter_type, use_energy=True,
                                                        lowfreq=args.lowfreq, log_scale=args.log_scale,
                                                        nfft=args.nfft, nfilt=args.filters, normalize=args.normalize,
                                                        duration=True, windowsize=args.windowsize,
                                                        multi_weight=args.multi_weight)
                        elif args.feat_type == 'spectrogram':
                            feat, duration = Make_Spect(wav_path=temp_wav, windowsize=args.windowsize,
                                                        lowfreq=args.lowfreq, stride=args.stride, duration=True,
                                                        nfft=args.nfft, normalize=args.normalize)
                        elif args.feat_type == 'mfcc':
                            feat, duration = Make_MFCC(filename=temp_wav, numcep=args.numcep, nfilt=args.filters,
                                                       lowfreq=args.lowfreq, normalize=args.normalize, duration=True,
                                                       use_energy=args.energy)

                        os.remove(temp_wav)

                    else:

                        if args.feat_type == 'fbank':
                            feat, duration = Make_Fbank(filename=pair[1], filtertype=args.filter_type, use_energy=True,
                                                        nfft=args.nfft, windowsize=args.windowsize,
                                                        lowfreq=args.lowfreq,
                                                        log_scale=args.log_scale, nfilt=args.filters, duration=True,
                                                        normalize=args.normalize, multi_weight=args.multi_weight)
                        elif args.feat_type == 'spectrogram':
                            feat, duration = Make_Spect(wav_path=pair[1], windowsize=args.windowsize,
                                                        bandpass=args.bandpass, lowfreq=args.lowfreq,
                                                        highfreq=args.highfreq,
                                                        log_scale=args.log_scale,
                                                        stride=args.stride, duration=True, nfft=args.nfft,
                                                        normalize=args.normalize)
                        elif args.feat_type == 'mfcc':
                            feat, duration = Make_MFCC(filename=pair[1], numcep=args.numcep, nfilt=args.filters,
                                                       lowfreq=args.lowfreq,
                                                       normalize=args.normalize, duration=True, use_energy=args.energy)
                        # feat = np.load(pair[1]).astype(np.float32)

                    feat = feat.astype(np.float32)
                    if args.feat_format == 'kaldi':
                        kaldi_io.write_mat(feat_ark_f, feat, key='')
                        offsets = feat_ark + ':' + str(feat_ark_f.tell() - len(feat.tobytes()) - 15)
                        # print(offsets)
                        feat_scp_f.write(key + ' ' + offsets + '\n')
                    elif args.feat_format == 'kaldi_cmp':
                        writer(str(key), feat)
                        # print(str(key))
                    elif args.feat_format == 'npy':
                        npy_path = os.path.join(feat_dir, '%s.npy' % key)
                        np.save(npy_path, feat)
                        feat_scp_f.write(key + ' ' + npy_path + '\n')

                    utt2dur_f.write('%s %.6f\n' % (key, duration))
                    utt2num_frames_f.write('%s %d\n' % (key, len(feat)))
                except Exception as e:
                    print(e)

                    print('line: ', e.__traceback__.tb_lineno)  # 发生异常所在的行数
                    print('file: ', e.__traceback__.tb_frame.f_globals["__file__"])  # 发生异常所在的文件

                    e_queue.put(key)

            # if t_queue.qsize() % 100 == 0:
            print('\rProcess [%6s] There are [%6s] speakers' \
                  ' left, with [%6s] errors.' % (str(os.getpid()), str(t_queue.qsize()), str(e_queue.qsize())),
                  end='')
        else:
            lock.release()  # 释放锁
            # print('\n>> Process {}:  queue empty!'.format(os.getpid()))
            break
    try:
        feat_scp_f.close()
        utt2dur_f.close()
    except:
        pass

    if args.feat_format == 'kaldi':
        feat_ark_f.close()
    elif args.feat_format == 'kaldi_cmp':
        writer.close()

    utt2num_frames_f.close()

    new_feat_scp = os.path.join(out_dir, 'feat.%d.scp' % proid)
    if args.feat_format == 'kaldi' and args.compress:
        new_feat_ark = os.path.join(feat_dir, 'feat.%d.ark' % proid)
        compress_command = "copy-feats --compress=true scp:{} ark,scp:{},{}".format(feat_scp, new_feat_ark,
                                                                                    new_feat_scp)
        pid, stdout, stderr = RunCommand(compress_command)
        # print(stdout)
        if os.path.exists(new_feat_scp) and os.path.exists(new_feat_ark):
            os.remove(feat_ark)
    else:
        shutil.copy(feat_scp, new_feat_scp)
        # pass

if __name__ == "__main__":

    nj = args.nj
    data_dir = args.data_dir
    out_dir = os.path.join(args.out_dir, args.out_set)
    sys.stdout = NewLogger(
        os.path.join(out_dir, 'log', 'feat.%s.conf' % time.strftime("%Y.%m.%d", time.localtime())))

    print('\nCurrent time is \33[91m{}\33[0m.'.format(str(time.asctime())))
    opts = vars(args)
    keys = list(opts.keys())
    keys.sort()

    options = []
    for k in keys:
        options.append("\'%s\': \'%s\'" % (str(k), str(opts[k])))

    print('Preparing feats options: \n{ %s }' % (', '.join(options)))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    wav_scp_f = os.path.join(data_dir, 'wav.scp')
    spk2utt_f = os.path.join(data_dir, 'spk2utt')
    assert os.path.exists(data_dir), print(data_dir)
    assert os.path.exists(wav_scp_f)
    assert os.path.exists(spk2utt_f)

    if data_dir != out_dir:
        print('Copy wav.scp, spk2utt, utt2spk, trials to %s' % out_dir)
        for f in ['wav.scp', 'spk2utt', 'utt2spk', 'trials', 'utt2dom']:
            orig_f = os.path.join(data_dir, f)
            targ_f = os.path.join(out_dir, f)
            if os.path.exists(orig_f):
                os.system('cp %s %s' % (orig_f, targ_f))

    uid2path = {}
    with open(wav_scp_f, 'r') as f:
        for line in f.readlines():
            ids = line.split()
            uid = ids[0]
            uid2path[uid] = line

        assert len(uid2path.keys()) > 0

    spk2utt = {}
    with open(spk2utt_f, 'r') as f:
        for line in f.readlines():
            ids = line.split()
            sid = ids[0]
            uids = ids[1:]
            spk2utt[sid] = uids

        assert len(spk2utt.keys()) > 0

    num_utt = len(uid2path.keys())
    start_time = time.time()

    manager = Manager()
    lock = manager.Lock()
    task_queue = manager.Queue()
    error_queue = manager.Queue()

    for sid in spk2utt.keys():
        pairs = []
        utts = spk2utt[sid]
        for uid in utts:
            pairs.append(uid2path[uid])

        task_queue.put(pairs)
    print('>>> Plan to make feats for %d speakers with %d utterances with %d jobs.\n' % (
        task_queue.qsize(), num_utt, nj))

    pool = Pool(processes=nj)  # 创建nj个进程
    for i in range(0, nj):
        write_dir = os.path.join(out_dir, 'Split%d/%d' % (nj, i))
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        ark_dir = os.path.join(args.out_dir, args.feat_type)
        if not os.path.exists(ark_dir):
            os.makedirs(ark_dir)

        pool.apply_async(MakeFeatsProcess, args=(lock, write_dir, ark_dir, args.out_set,
                                                 i, task_queue, error_queue))

    pool.close()  # 关闭进程池，表示不能在往进程池中添加进程
    pool.join()  # 等待进程池中的所有进程执行完毕，必须在close()之后调用

    if error_queue.qsize() > 0:
        print('\n>> Saving Completed with errors in: ')
        while not error_queue.empty():
            print(error_queue.get() + ' ', end='')
        print('')
    else:
        print('\n>> Saving Completed without errors.!')

    Split_dir = os.path.join(out_dir, 'Split%d' % nj)
    print('  >> Splited Data root is \n\t%s. \n\tConcat all scripts together.' % str(Split_dir))

    all_scp_path = [os.path.join(Split_dir, '%d/feat.%d.scp' % (i, i)) for i in range(nj)]
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
        print('Errors num of utterances: %s !=  ?' % feat_scp)

    numofutt = 0
    all_scp_path = [os.path.join(Split_dir, '%d/utt2dur.%d' % (i, i)) for i in range(nj)]
    utt2dur = os.path.join(out_dir, 'utt2dur')
    with open(utt2dur, 'w') as utt2dur_f:
        for item in all_scp_path:
            if not os.path.exists(item):
                continue
            for txt in open(str(item), 'r').readlines():
                utt2dur_f.write(txt)
                numofutt += 1
    if numofutt != num_utt:
        print('Errors in %s ?' % utt2dur)

    numofutt = 0
    all_scp_path = [os.path.join(Split_dir, '%d/utt2num_frames.%d' % (i, i)) for i in range(nj)]
    utt2num_frames = os.path.join(out_dir, 'utt2num_frames')
    with open(utt2num_frames, 'w') as utt2num_frames_f:
        for item in all_scp_path:
            if not os.path.exists(item):
                continue
            for txt in open(str(item), 'r').readlines():
                utt2num_frames_f.write(txt)
                numofutt += 1
    if numofutt != num_utt:
        print('Errors in %s ?' % utt2num_frames)

    print('Delete tmp files in: \n\t%s' % Split_dir)
    if args.compress:
        shutil.rmtree(Split_dir)

    end_time = time.time()
    all_time = end_time - start_time
    hours = int(all_time / 3600)
    mins = int(all_time % 3600 // 60)
    secs = int(all_time % 60)

    print('Write all files in: \n\t{:s}. \nAnd {:0>2d}:{:0>2d}:{:0>2d}s collapse.\n'.format(out_dir, hours, mins, secs))
    sys.exit()

"""
For multi threads, average making seconds for 47 speakers is 4.579958657
For one threads, average making seconds for 47 speakers is 4.11888732301

For multi process, average making seconds for 47 speakers is 1.67094940328
For one process, average making seconds for 47 speakers is 3.64203325738
"""
