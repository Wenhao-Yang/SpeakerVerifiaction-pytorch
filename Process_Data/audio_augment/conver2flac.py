#!/usr/bin/env python
# encoding: utf-8
'''
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: VS Code
@File: conver2flac.py
@Time: 2024/04/04 22:06
@Overview: 
'''
from Process_Data.audio_augment.common import RunCommand
import os
import time
from multiprocessing import Pool, Manager
import argparse


parser = argparse.ArgumentParser(description='Computing Filter banks!')
parser.add_argument('--nj', type=int, default=24,
                    help='number of jobs to make feats (default: 10)')

parser.add_argument('--data-dir', type=str, default='/home/yangwenhao/project/lstm_speaker_verification/data/cnceleb2/dev2/wav.scp',
                    help='number of jobs to make feats (default: 10)')

parser.add_argument("--replace-str", type=str,
                    default='/work/2023/yangwenhao/dataset/CN-Celeb2/data/,/data2022/yangwenhao/dataset/CN-Celeb2/data_radsnr1/') #'/wav/,/wav_nb_randsnr1_off4/'

args = parser.parse_args()

def ConvertProcess(lock, t_queue):

    while True:
        lock.acquire()  # 加上锁
        if not t_queue.empty():
            command = t_queue.get()
            lock.release()  # 释放锁

            spid, stdout, error = RunCommand(command)
            if t_queue.qsize() % 100 == 0:
                print('\rProcess [%6s] There are [%6s] utterances left.' % (str(os.getpid()), str(t_queue.qsize())), end='')
        else:
            lock.release()  # 释放锁
            # print('\n>> Process {}:  queue empty!'.format(os.getpid()))
            break



if __name__ == "__main__":

    nj = args.nj
    replace_str = args.replace_str.split(',')

    print('\nCurrent time is \33[91m{}\33[0m.'.format(str(time.asctime())))
    opts = vars(args)
    keys = list(opts.keys())
    keys.sort()

    options = []
    for k in keys:
        options.append("\'%s\': \'%s\'" % (str(k), str(opts[k])))

    print('Convert options: \n{ %s }' % (', '.join(options)))

    start_time = time.time()

    manager = Manager()
    lock = manager.Lock()
    task_queue = manager.Queue()
    error_queue = manager.Queue()

    uid2path = {}
    with open(args.data_dir, 'r') as f:
        for l in f.readlines():
            lst = l.split()
            if len(lst) == 2:
                uid, wav = lst
            elif len(lst) == 7:
                uid, wav = lst[0], lst[5]
            elif len(lst) == 15:
                uid, wav = lst[0], lst[4]
            
            recieve_path = wav.replace(replace_str[0], replace_str[1])
            if not os.path.isfile(recieve_path):
                parent_dir = os.path.dirname(recieve_path)
                if not os.path.isdir(parent_dir):
                    os.makedirs(parent_dir)

            commd = 'sox {} {}'.format()
            task_queue.put(commd)

    print('>>> Plan to convert flac %d utterances with %d jobs.\n' % (task_queue.qsize(), nj))

    pool = Pool(processes=nj)  # 创建nj个进程
    for i in range(0, nj):
        pool.apply_async(ConvertProcess, args=(lock, task_queue))

    pool.close()  # 关闭进程池，表示不能在往进程池中添加进程
    pool.join()  # 等待进程池中的所有进程执行完毕，必须在close()之后调用

    print('\n>> Ending !')


