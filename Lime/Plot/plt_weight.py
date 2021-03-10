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
x_161 = np.linspace(0, 8000, 161)
vox1_so /= vox1_so.sum()

vox1_res20 = np.load('Lime/Data/ResNet20/soft_wcmvn/epoch_24/grad.veri.npy')
vox1_res20 = gassuan_weight(vox1_res20)
x_257 = np.linspace(0, 8000, 257)
vox1_res20 /= vox1_res20.sum()
# plt.plot(x_257, vox1_res20)

vox2_lo = np.load('Lime/Data/LoResNet8/vox2/cbam_arcsoft_em256_k57/train.grad.veri.npy')
vox2_lo = gassuan_weight(vox2_lo)
vox2_lo /= vox2_lo.sum()

# vox1_thin = np.load('Lime/ThinResNet34/vox1/soft_None/train.grad.veri.npy')
vox1_thin = np.load('Lime/Data/ExResNet34/vox1/soft_wcmvn_l/grad.veri.npy')
# vox1_thin = gassuan_weight(vox1_thin, kernel_size=11)

# fb64_m = np.linspace(0, 2840.0230467083188, 64)
fb64_m = np.linspace(0, 2840.0230467083188, 65)
fb64_m = fb64_m[1:]

fb64_m = 700 * (10 ** (fb64_m / 2595.0) - 1)
mel_64 = get_filterbanks(nfilt=64)
vox1_thin_g = vox1_thin / mel_64.sum(axis=1)
vox1_thin_g /= vox1_thin_g.sum()
vox1_thin_g = gassuan_weight(vox1_thin_g, kernel_size=11)

# vox1_thin_g = gassuan_weight(vox1_thin_g, kernel_size=11)
# plt.plot(fb64_m, vox1_thin/fb64_m.sum(axis=1))

# vox1_tdnn = np.load('Lime/TDNN/vox1/fb40_stap/train.grad.veri.npy')
vox1_tdnn = np.load('Lime/Data/TDNN/vox1/grad.veri.npy')
# vox1_tdnn = gassuan_weight(vox1_tdnn, kernel_size=11)

# fb40_m = np.linspace(0, 2840.0230467083188, 40)
fb40_m = np.linspace(0, 2840.0230467083188, 41)
fb40_m = fb40_m[1:]
fb40_m = 700 * (10 ** (fb40_m / 2595.0) - 1)
mel_40 = get_filterbanks(nfilt=40)
vox1_tdnn_g = vox1_tdnn / mel_40.sum(axis=1)
vox1_tdnn_g /= vox1_tdnn_g.sum()

vox1_tdnn_g = gassuan_weight(vox1_tdnn_g, kernel_size=11)

# vox1_tdnn_g = gassuan_weight(vox1_tdnn_g, kernel_size=11)
# plt.plot(fb40_m, vox1_tdnn/fb40_m.sum(axis=1))

m = np.arange(0, 2840.0230467083188)
m = 700 * (10 ** (m / 2595.0) - 1)
n = np.array([m[i] - m[i - 1] for i in range(1, len(m))])
n = 1 / n

f = interpolate.interp1d(m[1:], n)
xnew = np.arange(np.min(m[1:]), np.max(m[1:]), (np.max(m[1:]) - np.min(m[1:])) / 257)
ynew = f(xnew)
ynew = ynew / ynew.sum()

pdf = PdfPages('Lime/Analysis/vox1.grad.pdf')
plt.rc('font', family='Times New Roman')
plt.figure(figsize=(8.5, 4.5))
# ax.set_title('Resolution')
# plt.plot(xnew, ynew)

f = interpolate.interp1d(x_161, vox1_so)
plt.plot(xnew, f(xnew) / f(xnew).sum())

f = interpolate.interp1d(x_161, vox2_lo)
plt.plot(xnew, f(xnew) / f(xnew).sum())

plt.legend(['vox1', 'vox2'], fontsize=18)
plt.ylabel('Weight', fontsize=18)
plt.xlabel('Frequency', fontsize=18)

plt.show()
plt.figure(figsize=(8.5, 5))

f = interpolate.interp1d(x_161, vox1_so)
plt.plot(xnew, f(xnew) / f(xnew).sum())

f = interpolate.interp1d(x_257, vox1_res20)
plt.plot(xnew, f(xnew) / f(xnew).sum())

# plt.plot(x_257, vox1_res20)
# plt.plot(x_257, vox1_res20)

f = interpolate.interp1d(fb64_m, vox1_thin_g, fill_value="extrapolate")
plt.plot(xnew, f(xnew) / f(xnew).sum())

f = interpolate.interp1d(fb40_m, vox1_tdnn_g, fill_value="extrapolate")
plt.plot(xnew, f(xnew) / f(xnew).sum())

# plt.plot(fb40_m, vox1_tdnn_g)
# plt.plot(x, tim_so)
# plt.ylim(0, 0.03)
plt.annotate(s='Drop', xy=(1800, 0.0042), xytext=(1000, 0.002), fontsize=16,
             arrowprops=dict(arrowstyle='-|>', connectionstyle='arc3', color='red'))
plt.legend(['ResCNN_v1', 'ResNet20', 'ThinResNet34', 'TDNN'], fontsize=18)

plt.ylabel('Weight', fontsize=18)
plt.xlabel('Frequency', fontsize=18)
pdf.savefig()
pdf.close()
plt.show()
