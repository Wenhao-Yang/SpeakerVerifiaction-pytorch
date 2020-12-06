#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: vad_test.py
@Time: 2019/11/16 下午6:52
@Overview:
"""
import pdb
from scipy import signal
import numpy as np
from scipy.io import wavfile
import torch
import torch.nn as nn
import torch.optim as optim

from Process_Data.compute_vad import ComputeVadEnergy
from python_speech_features import fbank

import matplotlib.pyplot as plt
torch.manual_seed(0)


# def preemphasis(signal,coeff=0.95):
#     """perform preemphasis on the input signal.
#
#     :param signal: The signal to filter.
#     :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
#     :returns: the filtered signal.
#     """
#     return np.append(signal[0],signal[1:]-coeff*signal[:-1])
#
# audio_pre = preemphasis(audio, 0.97)
#
# f, t, Zxx = signal.stft(audio_pre,
#                         fs,
#                         window=signal.hamming(int(fs*0.025)),
#                         noverlap=fs * 0.015,
#                         nperseg=fs * 0.025,
#                         nfft=512)

# energy = 1.0/512 * np.square(np.absolute(Zxx))
# energy = np.sum(energy,1)

class Vad_layer(nn.Module):
    def __init__(self, feat_dim):
        super(Vad_layer, self).__init__()
        self.linear1 = nn.Linear(feat_dim, feat_dim)
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=1,
                               kernel_size=(5, 1),
                               padding=(2, 0))

        self.linear2 = nn.Linear(feat_dim, 1)
        self.avg = nn.AdaptiveAvgPool3d((1, None, 26))
        self.max = nn.MaxPool2d(kernel_size=(5, 5),
                                stride=1,
                                padding=(2,2))
        self.relu2 = nn.ReLU()
        self.acti = nn.ReLU()
        # nn.init.eye(self.linear1.weight.data)
        # nn.init.zeros_(self.linear1.bias.data)
        # nn.init.eye(self.linear2.weight.data)
        nn.init.normal_(self.linear2.bias.data, mean=-16, std=1)
        nn.init.normal_(self.linear2.weight.data, mean=1, std=0.1)

        nn.init.normal_(self.conv1.weight.data, mean=1, std=0.1)

    def forward(self, input):
        input = input.float()
        # vad_fb = self.weight1.mm(torch.log(input))

        vad_fb = self.conv1(input)
        vad_fb = self.max(vad_fb)

        vad_fb = self.avg(vad_fb)
        # vad_fb = self.linear1(vad_fb)
        vad_fb = self.linear2(vad_fb)
        vad_fb = self.relu2(vad_fb)
        #vad_fb = torch.sign(vad_fb)
        return vad_fb

    def en_forward(self, input):
        input = torch.log(input.float())
        vad_fb = self.conv1(input)
        vad_fb = self.linear2(vad_fb)
        vad_fb = self.max(vad_fb)
        # vad_fb = self.linear2(input)
        vad_fb = self.acti(vad_fb)
        vad_fb = torch.sign(vad_fb)
        # vad_fb = self.acti(vad_fb)
        return vad_fb

def vad_fb(filename):
    fs, audio = wavfile.read(filename)

    fb, ener = fbank(audio, samplerate=fs, nfilt=64, winfunc=np.hamming)
    fb[:, 0] = np.log(ener)
    # log_ener = np.log(ener)
    ener = ener.reshape((ener.shape[0], 1))

    ten_fb = torch.from_numpy(fb).unsqueeze(0)
    ten_fb = ten_fb.unsqueeze(0)
    vad_lis = []
    ComputeVadEnergy(ener, vad_lis)
    vad = np.array(vad_lis)
    print(float(np.sum(vad)) / len(vad))
    ten_vad = torch.from_numpy(vad)
    ten_vad = ten_vad.unsqueeze(1).float()

    return ten_fb, ten_vad, torch.from_numpy(ener).unsqueeze(0)

def main():
    filename2 = 'Data/voxceleb1/vox1_dev_noise/id10001/1zcIwhmdeo4/00001.wav'
    filename1 = 'Data/Aishell/data_aishell/wav/train/S0002/BAC009S0002W0128.wav'
    filename3 = 'Data/voxceleb1/vox1_dev_noise/id10001/1zcIwhmdeo4/00002.wav'

    fb1, vad1, ener1 = vad_fb(filename1)
    fb2, vad2, ener2 = vad_fb(filename2)
    fb3, vad3, ener3 = vad_fb(filename3)

    if fb1.shape[2]<=fb2.shape[2]:
        fb2 = fb2[:,:,:fb1.shape[2],:]
        vad2 = vad2[:fb1.shape[2], :]
        ener2 = ener2[:, :fb1.shape[2], :]

    fb = torch.cat((fb1, fb2), dim=0)
    vad = torch.cat((vad1.unsqueeze(0), vad2.unsqueeze(0)), dim=0)
    ener = torch.cat((ener1.unsqueeze(0), ener2.unsqueeze(0)), dim=0)
    vad1 = vad1.unsqueeze(0)
    input = fb1
    vad = vad1
    # input = ener.view(ener.shape[0], ener.shape[1], ener.shape[3], ener.shape[2])
    # vad_fb = ten_vad.mul(ten_fb).float()


    model = Vad_layer(input.shape[3])
    optimizer = optim.Adagrad(model.parameters(), lr=0.01,
                          weight_decay=1e-3)
    # optimizer = optim.SGD(model.parameters(), lr=0.001,
    #                       momentum=0.99, dampening=0.9,
    #                       weight_decay=1e-4)
    ce1 = nn.L1Loss(reduction='mean')
    ce2 = nn.MSELoss(reduction='mean')
    ce3 = nn.BCELoss()
    sm = nn.Softmax(dim=1)

    epochs = 501
    loss_va = []
    accuracy = []

    for epoch in range(1, epochs):

        vad_out = model.en_forward(input)
        loss = ce2(vad_out, vad) # +ce1(vad_out, vad) # +  #+ (vad_out.mean()) #
        #loss = ce3(vad_out.view(-1), vad.view(-1))
        # loss = ce3(vad_out, vad)
        if epoch % 4==3:
            print('Loss is {}'.format(loss.item()))
        loss_va.append(loss.item())


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # pdb.set_trace()
        # acc_num = torch.max(sm(vad_out.view(-1)), dim=1)[1]
        acc_num = vad_out.view(-1)
        for i in range(len(acc_num)):
            if acc_num[i]>-0.5:
                acc_num[i]=1

        acc = float((acc_num.long() == vad.view(-1).long()).sum())
        accuracy.append(acc/len(vad_out.squeeze().view(-1)))

    vad_out = model.en_forward(input)
    print(vad_out)
    plt.plot(loss_va)
    plt.show()
    plt.plot(accuracy)
    plt.show()




if __name__ == '__main__':
    main()
