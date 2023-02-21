#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: analysis.py
@Time: 2022/12/20 下午4:14
@Overview:
"""
import os
import numpy as np

def format_eer_file(file_path):
    assert os.path.exists(file_path)
    eer = []
    mindcf01 = []
    mindcf001 = []
    model_str = ''
    with open(file_path, 'r') as f:
        for l in f.readlines():
            ls = l.split()
            if len(ls) ==0:
                continue
            elif len(ls)< 10:
                model_str = "-".join(ls[1:])
                # print("-".join(ls[1:]))
                test_set=''
                eer = []
                mindcf01 = []
                mindcf001 = []

            elif len(ls)>= 10:
                eer.append(float(ls[3]))
                mindcf01.append(float(ls[7]))
                mindcf001.append(float(ls[9]))
                test_set=ls[1]

            if len(eer)==3:
                print("#|{: ^19s}".format(test_set)+"|  {:>5.2f}±{:<.2f}  |".format(np.mean(eer), np.std(eer)), end=' ')
                print("%.4f±%.4f"%(np.mean(mindcf01), np.std(mindcf01)), end=' ')
                print("| %.4f±%.4f | %s"%(np.mean(mindcf001), np.std(mindcf001), model_str)) 
                
                eer = []
                mindcf01 = []
                mindcf001 = []
                
def format_eer_file_train(file_path):
    assert os.path.exists(file_path)
    eer = []
    mindcf01 = []
    mindcf001 = []
    mix2 = []
    mix3 = []
    model_str = ''
    with open(file_path, 'r') as f:
        for l in f.readlines():
            ls = l.split()
            if len(ls) ==0:
                continue
            elif len(ls)< 10:
                model_str = "-".join(ls[1:])
                # print("-".join(ls[1:]))
                test_set=''
                eer = []
                mindcf01 = []
                mindcf001 = []

            elif len(ls)>= 10:
                eer.append(float(ls[1]))
                mindcf01.append(float(ls[5]))
                mindcf001.append(float(ls[7]))
                mix2.append(float(ls[9].rstrip(',')))
                mix3.append(float(ls[10].rstrip('.')))
                test_set=''# ls[1]

            if len(eer)==3:
                print("#|{: ^19s}".format(test_set)+"|  {:>5.2f}±{:<.2f}  |".format(np.mean(eer), np.std(eer)), end=' ')
                print("%.4f±%.4f"%(np.mean(mindcf01), np.std(mindcf01)), end=' ')
                print("| %.4f±%.4f"%(np.mean(mindcf001), np.std(mindcf001)), end=' ') 
                print("| %.4f±%.4f"%(np.mean(mix2), np.std(mix2)), end=' ') 
                print("| %.4f±%.4f | %s"%(np.mean(mix3), np.std(mix3), model_str)) 
                
                eer = []
                mindcf01 = []
                mindcf001 = []

                
def read_eer_file(file_path):
    assert os.path.exists(file_path)
    eer = []
    mindcf01 = []
    mindcf001 = []

    result_lst = []
    result_idx = []
    with open(file_path, 'r') as f:
        for l in f.readlines():
            ls = l.split()
            if len(ls) ==0:
                continue
            elif len(ls)>= 10:
                eer.append(float(ls[3]))
                mindcf01.append(float(ls[7]))
                mindcf001.append(float(ls[9]))
                test_set=int(ls[1].split(',')[1])-1 #vox1-test-0,1

            if len(eer)==3:
                result_idx.append(test_set)
                result_lst.append([np.mean(eer), np.mean(mindcf01), np.mean(mindcf001)])
                eer = []
                mindcf01 = []
                mindcf001 = []
                
    result_lst = np.array(result_lst)
    result_idx = np.array(result_idx)
    
    return result_idx, result_lst
