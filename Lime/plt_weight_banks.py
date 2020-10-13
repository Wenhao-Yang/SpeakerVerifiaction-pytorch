#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: plt_weight_banks.py
@Time: 2020/10/12 23:16
@Overview:
"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import interpolate

import Process_Data.constants as c
from Process_Data.xfcc.common import get_filterbanks

timit_soft = get_filterbanks(nfilt=23, nfft=320, samplerate=16000, lowfreq=0,
                             highfreq=None, filtertype='dnn.timit.soft', multi_weight=False)
mel = get_filterbanks(nfilt=23, nfft=320, samplerate=16000, lowfreq=0,
                      highfreq=None, filtertype='mel', multi_weight=False)

tim_so = np.array(c.TIMIT_FIlTER_SOFT)

x = np.linspace(0, 8000, 161)
m = np.arange(0, 2840.0230467083188)
m = 700 * (10 ** (m / 2595.0) - 1)
n = np.array([m[i] - m[i - 1] for i in range(1, len(m))])
n = 1 / n

f = interpolate.interp1d(m[1:], n)
xnew = np.arange(np.min(m[1:]), np.max(m[1:]), (np.max(m[1:]) - np.min(m[1:])) / 161)
ynew = f(xnew)
ynew = ynew / ynew.sum()

pdf = PdfPages('Lime/LoResNet8/timit/soft/grad.filter.pdf')
plt.rc('font', family='Times New Roman')

fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(4, 1)
ax = fig.add_subplot(gs[0:2, 0])
ax.plot(xnew, ynew)
ax.plot(x, tim_so)
ax.legend(['Mel', 'Gradient'])
ax = fig.add_subplot(gs[2, 0])
for m in mel:
    ax.plot(xnew, m, color='#1f77b4', linewidth=0.8)
ax = fig.add_subplot(gs[3, 0])
for m in timit_soft:
    ax.plot(x, m, color='#ff7f0e', linewidth=0.8)

pdf.savefig()
pdf.close()
plt.show()
