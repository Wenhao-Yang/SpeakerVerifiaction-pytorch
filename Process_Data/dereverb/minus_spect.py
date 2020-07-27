#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: minus_spect.py
@Time: 2020/7/27 16:46
@Overview:
"""
import math

# !/usr/bin/env python
import numpy as np
import soundfile as sf
# 打开WAV文档
from scipy.signal import stft, istft

wav, sr = sf.read('Data/dataset/wav_test/sample/CHN01/D01-U000043-8k.wav', dtype='float32')

fs = sr
# 读取波形数据
# 将波形数据转换为数组
# 计算参数

len_ = int(25 * fs // 1000)  # 样本中帧的大小
PERC = 0.6  # 窗口重叠占帧的百分比
len1 = int(len_ * PERC // 100)  # 重叠窗口
len2 = int(len_ - len1)  # 非重叠窗口
# 设置默认参数
Thres = 3
Expnt = 2.0
beta = 0.002
G = 0.9
# 初始化汉明窗
win = np.hamming(200)
# normalization gain for overlap+add with 50% overlap
winGain = len2 / sum(win)

# Noise magnitude calculations - assuming that the first 5 frames is noise/silence
nFFT = 200
noise_mean = np.zeros(nFFT)

# j = 0
# for k in range(1, 6):
#     noise_mean = noise_mean + abs(np.fft.fft(win * x[j:j + len_], nFFT))
#     j = j + len_

wav1, sr1 = sf.read('Data/dataset/wav_test/noise/CHN01/D01-U000000-8k.wav', dtype='float32')
f, t, Zxx = stft(wav1, fs=sr, nperseg=int(sr * 0.025), noverlap=int(sr * 0.015), nfft=nFFT,
                 window=np.hamming(int(sr * 0.025)))

noise_mu = np.absolute(Zxx).mean(1)

# --- allocate memory and initialize various variables
k = 1
img = 1j
x_old = np.zeros(len1)
# Nframes = len(x) // len2 - 1
xfinal = []

f, t, Zxx = stft(wav, fs=sr, nperseg=int(sr * 0.025), noverlap=int(sr * 0.015), nfft=nFFT,
                 window=np.hamming(int(sr * 0.025)))

# =========================    Start Processing   ===============================
for n in range(0, Zxx.shape[1]):
    # Windowing

    spec = Zxx[:, n]
    # compute the magnitude
    sig = abs(spec)

    # save the noisy phase information
    theta = np.angle(spec)
    SNRseg = 10 * ((np.log10(np.linalg.norm(sig, 2) ** 2) - np.log10(np.linalg.norm(noise_mu, 2) ** 2)) / np.log10(
        np.linalg.norm(noise_mu, 2)) ** 2)


    def berouti(SNR):
        if -5.0 <= SNR or SNR <= 20.0:
            a = 4 - SNR * 3 / 20
        else:
            if SNR < -5.0:
                a = 5
            if SNR > 20:
                a = 1
        # print(SNR)
        return a


    def berouti1(SNR):
        if -5.0 <= SNR <= 20.0:
            a = 3 - SNR * 2 / 20
        else:
            if SNR < -5.0:
                a = 4
            if SNR > 20:
                a = 1
        return a


    if Expnt == 1.0:  # 幅度谱
        alpha = berouti1(SNRseg)
    else:  # 功率谱
        alpha = berouti(SNRseg)
    #############
    sub_speech = sig ** Expnt - beta * noise_mu ** Expnt;
    # 当纯净信号小于噪声信号的功率时
    # diffw = sub_speech - beta * noise_mu ** Expnt
    # # beta negative components
    #
    # def find_index(x_list):
    #     index_list = []
    #     for i in range(len(x_list)):
    #         if x_list[i] < 0:
    #             index_list.append(i)
    #     return index_list
    # #
    # z = find_index(diffw)
    # if len(z) > 0:
    #     # 用估计出来的噪声信号表示下限值
    #     sub_speech[z] = beta * noise_mu[z] ** Expnt
    #     # --- implement a simple VAD detector --------------
    #
    # if SNRseg < Thres:  # Update noise spectrum
    #     noise_temp = G * noise_mu ** Expnt + (1 - G) * sig ** Expnt  # 平滑处理噪声功率谱
    #     noise_mu = noise_temp ** (1 / Expnt)  # 新的噪声幅度谱

    # flipud函数实现矩阵的上下翻转，是以矩阵的“水平中线”为对称轴
    # 交换上下对称元素
    # sub_speech[nFFT // 2 + 1:nFFT] = np.flipud(sub_speech[1:nFFT // 2])
    x_phase = (sub_speech ** (1 / Expnt)) * (
                np.array([math.cos(x) for x in theta]) + img * (np.array([math.sin(x) for x in theta])))
    # take the IFFT

    xfinal.append(x_phase)

sub_x = np.array(xfinal)
wav_su = istft(sub_x, fs=sr, nperseg=int(sr * 0.025), noverlap=int(sr * 0.015), nfft=nFFT,
               window=np.hamming(int(sr * 0.025)), time_axis=-2, freq_axis=-1)

sf.write('Data/dataset/wav_test/sample/CHN01/D01-U000043-8k.sub.wav', wav_su[1], samplerate=sr, format='WAV')
print('j')
# 保存文件
# wf = wave.open('en_outfile.wav', 'wb')
# # 设置参数
# wf.setparams(params)
# # 设置波形文件 .tostring()将array转换为data
# wave_data = (winGain * xfinal).astype(np.short)
# wf.writeframes(wave_data.tostring())
# wf.close()
