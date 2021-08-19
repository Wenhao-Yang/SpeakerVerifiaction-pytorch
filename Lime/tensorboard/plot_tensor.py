#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: plot_tensor.py
@Time: 2020/3/24 6:20 PM
@Overview:
"""
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import json
from scipy import interpolate
from scipy.interpolate import spline

path_root = '/Users/yang/PycharmProjects/SpeakerVerification-pytorch/Lime/Data/tensorboard/EER'
path_root = pathlib.Path(path_root)
all_json = list(path_root.glob('*.json'))
all_json = [str(j) for j in all_json]
all_json.sort()

al_data = []
models = ['ExResNet34-cmvn', 'ExResNet34', 'SiResNet34', 'SuResCNN10']
al_data_label = ['Augment', 'Original']

for j in all_json:
    f_j = open(j, 'r')
    f_data = json.load(f_j)

    f_n = pathlib.Path(j).name
    model = f_n.split('_')[1]
    m_set = f_n.split('_')[3]
    m_set = m_set.split('-')[0]

    al_data.append(f_data)
    # al_data_label.append(model+'-'+m_set)

for idx, model in enumerate(models):

    plt.rc('font',family='Times New Roman')
    plt.figure(figsize=(6, 6))
    plt.title(model, fontsize=25)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.tick_params(labelsize=15)
    n = 0
    for l in al_data[(idx*2):(idx*2+2)]:
        lnp=np.array(l)
        x = lnp[:, 1]
        y = lnp[:, 2]

        model_str = '-aug' if n % 2==0 else '-original'
        n+=1
        print('%s Final EER is [%.2f]%%.' % (model + model_str, np.mean(np.array(y[-3:]))))
        f = interpolate.interp1d(x, y)
        xnew = np.arange(np.min(x), np.max(x), 1)
        ynew = f(xnew)

        plt.plot(xnew, ynew)
    print('--')
    plt.legend(al_data_label, loc='upper right', fontsize=15)
    # plt.plot(SiRes_aug[:, 1], SiRes_aug[:, 2])
    plt.show()

# plt.rc('font',family='Times New Roman')
# plt.figure(figsize=(6, 6))
# plt.title('SiResNet34', fontsize=25)
# plt.xlabel('Epoch', fontsize=15)
# plt.ylabel('Loss', fontsize=15)
# plt.tick_params(labelsize=15)
# for l in al_data[2:4]:
#     lnp=np.array(l)
#     x = lnp[:, 1]
#     y = lnp[:, 2]
#
#     print('Final EER is [%s]%%.' % np.mean(np.array(y[-3:])))
#     f = interpolate.interp1d(x, y)
#     xnew = np.arange(np.min(x), np.max(x), 1)
#     ynew = f(xnew)
#
#     plt.plot(xnew, ynew)
#
#
# plt.legend(al_data_label, loc='upper right', fontsize=15)
# # plt.plot(SiRes_aug[:, 1], SiRes_aug[:, 2])
# plt.show()
#
#
# plt.figure(figsize=(6, 6))
# plt.title('SuResCNN10', fontsize=25)
# plt.xlabel('Epoch', fontsize=15)
# plt.ylabel('Loss', fontsize=15)
# plt.tick_params(labelsize=15)
# for l in al_data[4:]:
#     lnp=np.array(l)
#     x = lnp[:, 1]
#     y = lnp[:, 2]
#     print('Final EER is [%s]%%.' % np.mean(np.array(y[-3:])))
#     f = interpolate.interp1d(x, y)
#     xnew = np.arange(np.min(x), np.max(x), 1)
#     ynew = f(xnew)
#     plt.plot(xnew, ynew)
#
#
# plt.legend(al_data_label, loc='upper right', fontsize=15)
# # plt.plot(SiRes_aug[:, 1], SiRes_aug[:, 2])
# plt.show()