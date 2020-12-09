#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: plt_time_weight.py
@Time: 2020/11/28 13:35
@Overview:
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import interpolate

import Process_Data.constants as c
from Lime import gassuan_weight
from Process_Data.xfcc.common import get_filterbanks

time_data = np.load('Lime/LoResNet8/timit/soft/time.data.pickle')

data = time_data[0][0]
grad = time_data[0][1]

fig = plt.figure(figsize=(8,6))

fig.tight_layout()  # 调整整体空白
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)


ax = plt.subplot(312)
plt.imshow(data.transpose(), aspect='auto')
ax.set_title('Log Spectrogram')

ax = plt.subplot(311)
plt.plot(np.log(np.exp(data).sum(axis=1)))
plt.xlim(0, 320)
ax.set_title('Log Power Energy')

ax = plt.subplot(313)
plt.plot(np.abs(grad).mean(axis=1)/np.abs(grad).mean(axis=1).sum())
plt.xlim(0, 320)
ax.set_title('Gradient along time axis')


# plt.subplot(414)
# plt.plot(np.abs(data).mean(axis=1)/np.abs(data).mean(axis=1).sum()*np.abs(grad).mean(axis=1))
# plt.xlim(0, 320)


# fb64_m = 700 * (10 ** (fb64_m / 2595.0) - 1)

# plt.ylabel('Weight', fontsize=18)
# plt.xlabel('Frequency', fontsize=18)
# pdf.savefig()
# pdf.close()
plt.show()
