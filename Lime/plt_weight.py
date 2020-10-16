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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import interpolate

import Process_Data.constants as c
from Lime import gassuan_weight
from Process_Data.xfcc.common import get_filterbanks

timit_soft = get_filterbanks(nfilt=23, nfft=320, samplerate=16000, lowfreq=0,
                             highfreq=None, filtertype='dnn.timit.soft', multi_weight=False)
mel = get_filterbanks(nfilt=23, nfft=320, samplerate=16000, lowfreq=0,
                      highfreq=None, filtertype='mel', multi_weight=False)

vox1_so = np.array(c.VOX_FILTER_SOFT)

vox1_res20 = np.load('Lime/ResNet20/soft_wcmvn/epoch_24/grad.veri.npy')
x_257 = np.linspace(0, 8000, 257)

vox1_thi = np.load('Lime/ExResNet34/vox1/soft_wcmvn_l/grad.veri.npy')


tim_fr = gassuan_weight(tim_fr)
tim_fr = tim_fr / tim_fr.sum()

x = np.linspace(0, 8000, 161)
m = np.arange(0, 2840.0230467083188)
m = 700 * (10 ** (m / 2595.0) - 1)
n = np.array([m[i] - m[i - 1] for i in range(1, len(m))])
n = 1 / n

f = interpolate.interp1d(m[1:], n)
xnew = np.arange(np.min(m[1:]), np.max(m[1:]), (np.max(m[1:]) - np.min(m[1:])) / 161)
ynew = f(xnew)
ynew = ynew / ynew.sum()

pdf = PdfPages('Lime/LoResNet8/timit/soft/grad_filter.pdf')
plt.rc('font', family='Times New Roman')

# ax.set_title('Resolution')
plt.plot(xnew, ynew)
plt.plot(x, tim_fr)
plt.plot(x, tim_so)
plt.ylim(0, 0.03)
plt.legend(['Mel Scale', 'F-ratio', 'NN Gradient'])

plt.ylabel('Weight')
plt.xlabel('Frequency')
pdf.savefig()
pdf.close()
plt.show()
