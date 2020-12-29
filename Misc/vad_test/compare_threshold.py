#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: compare_threshold.py
@Time: 2020/11/29 20:31
@Overview:
"""
import matplotlib.pyplot as plt
from Process_Data.Compute_Feat.compute_vad import ComputeVadEnergy
from Process_Data.audio_processing import Make_Spect
import  numpy as np


# v1 = 'Data/dataset/voxceleb1/8k_radio_v3/id10001/1zcIwhmdeo4/00001.wav'
v1 = 'Data/dataset/aishell-2/data/C0001/IC0001W0001-8k.wav'
a1= 'Data/dataset/wav_test/01-yangxiaokang/tmp_0001-U000013_15s_8k.wav'
r1 = 'Data/dataset/wav_test/00-yangwenhao/tmp_0009-U000000_15s_8k.wav'

v1_spect = Make_Spect(v1, windowsize=0.02, stride=0.01, nfft=320, normalize=False)
a1_spect = Make_Spect(a1, windowsize=0.02, stride=0.01, nfft=320, normalize=False)
r1_spect = Make_Spect(r1, windowsize=0.02, stride=0.01, nfft=320, normalize=False)

v1_energy = np.log(np.exp(v1_spect).sum(axis=1)).reshape(-1, 1)
a1_energy = np.log(np.exp(a1_spect).sum(axis=1)).reshape(-1, 1)
r1_energy = np.log(np.exp(r1_spect).sum(axis=1)).reshape(-1, 1)

v1_vad = ComputeVadEnergy(v1_energy)
a1_vad = ComputeVadEnergy(a1_energy)
r1_vad = ComputeVadEnergy(r1_energy)

v1_vad_071 = ComputeVadEnergy(v1_energy, energy_mean_scale=0.71)
a1_vad_071 = ComputeVadEnergy(a1_energy, energy_mean_scale=0.71)
r1_vad_071 = ComputeVadEnergy(r1_energy, energy_mean_scale=0.71)

v1_vad_071 = ComputeVadEnergy(v1_energy, energy_mean_scale=0.72)
a1_vad_071 = ComputeVadEnergy(a1_energy, energy_mean_scale=0.72)
r1_vad_071 = ComputeVadEnergy(r1_energy, energy_mean_scale=0.72)

for m in [0.25]:
    fig = plt.figure(figsize=(12, 8))

    plt.subplot(331)
    plt.imshow(v1_spect.transpose(), aspect='auto')
    plt.subplot(334)
    plt.ylim(-0.1, 1.1)
    plt.xlim(0, len(v1_spect))
    plt.plot(v1_vad)
    plt.xlim(0, len(v1_spect))
    plt.subplot(337)
    plt.xlim(0, len(v1_spect))
    plt.plot(ComputeVadEnergy(v1_energy, energy_mean_scale=m))

    plt.subplot(332)
    plt.imshow(a1_spect.transpose(), aspect='auto')
    plt.subplot(335)
    plt.xlim(0, len(a1_spect))
    plt.ylim(-0.1, 1.1)
    plt.plot(a1_vad)
    plt.subplot(338)
    plt.xlim(0, len(a1_spect))
    plt.plot(ComputeVadEnergy(a1_energy, energy_mean_scale=m))

    plt.subplot(333)
    plt.imshow(r1_spect.transpose(), aspect='auto')
    plt.subplot(336)
    plt.ylim(-0.1, 1.1)
    plt.xlim(0, len(r1_spect))
    plt.plot(r1_vad)
    plt.subplot(339)
    plt.xlim(0, len(r1_spect))
    plt.plot(ComputeVadEnergy(r1_energy, energy_mean_scale=m))

    # plt.suptitle('Energy_Mean_Scale is %.4f ' % m)
    plt.show()






