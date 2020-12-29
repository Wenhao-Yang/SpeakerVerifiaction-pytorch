#!/usr/bin/env python
# encoding: utf-8
"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: test_model_load.py
@Time: 2020/12/6 19:27
@Overview:
"""

from collections import OrderedDict

import sys
import torch
from torch import nn
from torch.autograd import Variable
import os
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import traceback
import pynvml

sys.path.append("C:/Users/WILLIAM/Documents/Project/SpeakerVerification-pytorch/Define_Model")
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

FRAMES=300
FRAMES_SHIFT=150
file_loader = np.load

class TruncatedVarInput(object):  # 以固定长度截断语音，并进行堆叠后输出
    def __init__(self, frames=FRAMES, frame_shift=FRAMES_SHIFT, remove_vad=True):
        super(TruncatedVarInput, self).__init__()
        self.frames = frames
        self.frame_shift = frame_shift
        self.remove_vad = remove_vad

    def __call__(self, frames_features):
        network_inputs = []

        if self.remove_vad:
            output = frames_features[:, 1:]
        else:
            output = frames_features

        while len(output) < self.frames:
            output = np.concatenate((output, frames_features[:, 1:]), axis=0)

        input_per_file = np.ceil(len(frames_features)/self.frame_shift)
        input_per_file = input_per_file.astype(np.int)

        for i in range(input_per_file):
            if i*self.frame_shift+self.frames >= len(output):
                network_inputs.append(output[-self.frames:, :])
            else:
                network_inputs.append(output[i*self.frame_shift:(i*self.frame_shift+self.frames), :])

        return torch.tensor(network_inputs, dtype=torch.float32)

transform = transforms.Compose([
            TruncatedVarInput()
        ])

class ExtractDataset(Dataset):  # 定义pytorch提取数据vector的类
    def __init__(self, uid2feat_path, transform=transform, file_loader=file_loader):
        self.transform = transform  # 变换函数
        self.file_loader = file_loader

        self.uid2feat_dict = {}
        feat_scp = uid2feat_path

        if not os.path.exists(feat_scp):
            print(" %s not exist...exit.." % feat_scp)
            raise FileExistsError(feat_scp)

        with open(feat_scp, 'r') as f:
            uid2feat = f.readlines()
            for l in uid2feat:
                try:
                    uid, feat_path, sid = l.split()
                except:
                    uid, feat_path = l.split()
                    # sid = None
                if os.path.exists(feat_path):
                    self.uid2feat_dict[uid] = (feat_path, uid)

        self.uids = list(self.uid2feat_dict.keys())
        self.uids.sort()

    def __getitem__(self, index):  # 获取样本的方式
            uid = self.uids[index]
            y = self.file_loader(self.uid2feat_dict[uid][0])
            # sid = self.uid2feat_dict[uid][1]
            feature = self.transform(y)
            return uid, feature# , sid

    def __len__(self):  #  数据集大小
        return len(self.uids)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):  # 3x3卷积，输入通道，输出通道，stride
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ReLU20(nn.Hardtanh):  # relu
    def __init__(self, inplace=False):
        super(ReLU20, self).__init__(0, 20, inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str

class BasicBlock(nn.Module):  # 定义block

    expansion = 1

    def __init__(self, in_channels, channels, stride=1, downsample=None):  # 输入通道，输出通道，stride，下采样
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = ReLU20(inplace=True)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out  # block输出

class LocalResNet(nn.Module):
    """
    网络模型
    """

    def __init__(self, embedding_size, num_classes, block=BasicBlock,
                 resnet_size=8, channels=[64, 128, 256], dropout_p=0.,
                 inst_norm=False, alpha=12,
                 avg_size=4, kernal_size=5, padding=2, **kwargs):

        super(LocalResNet, self).__init__()
        resnet_type = { 8: [1, 1, 1, 0],
                       10: [1, 1, 1, 1],
                       18: [2, 2, 2, 2],
                       34: [3, 4, 6, 3],
                       50: [3, 4, 6, 3],
                       101: [3, 4, 23, 3]}

        layers = resnet_type[resnet_size]
        self.alpha = alpha
        self.layers = layers
        self.dropout_p = dropout_p

        self.embedding_size = embedding_size
        # self.relu = nn.LeakyReLU()
        self.relu = nn.ReLU(inplace=True)

        self.inplanes = channels[0]
        self.conv1 = nn.Conv2d(1, channels[0], kernel_size=5, stride=2, padding=2, bias=False)
        if inst_norm:
            self.bn1 = nn.InstanceNorm2d(channels[0])
        else:
            self.bn1 = nn.BatchNorm2d(channels[0])

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, channels[0], layers[0])

        self.inplanes = channels[1]
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=kernal_size, stride=2,
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.layer2 = self._make_layer(block, channels[1], layers[1])

        self.inplanes = channels[2]
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=kernal_size, stride=2,
                               padding=padding, bias=False)
        self.bn3 = nn.BatchNorm2d(channels[2])
        self.layer3 = self._make_layer(block, channels[2], layers[2])

        if layers[3] != 0:
            assert len(channels) == 4
            self.inplanes = channels[3]
            self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=kernal_size, stride=2,
                                   padding=padding, bias=False)
            self.bn4 = nn.BatchNorm2d(channels[3])
            self.layer4 = self._make_layer(block=block, planes=channels[3], blocks=layers[3])

        self.dropout = nn.Dropout(self.dropout_p)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, avg_size))

        self.fc = nn.Sequential(
            nn.Linear(self.inplanes * avg_size, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

        self.classifier = nn.Linear(self.embedding_size, num_classes)

        for m in self.modules():  # 对于各层参数的初始化
            if isinstance(m, nn.Conv2d):  # 以2/n的开方为标准差，做均值为0的正态分布
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):  # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def l2_norm(self, input, alpha=1.0):
        # alpha = log(p * ( class -2) / (1-p))
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output * alpha

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x = self.layer1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.layer3(x)

        if self.layers[3] != 0:
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu(x)
            x = self.layer4(x)

        if self.dropout_p > 0:
            x = self.dropout(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        if self.alpha:
            x = self.l2_norm(x, alpha=self.alpha)

        logits = self.classifier(x)

        return logits, x

class Mean_Norm(nn.Module):
    def __init__(self, dim=-2):
        super(Mean_Norm, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x - torch.mean(x, dim=self.dim, keepdim=True)

    def __repr__(self):
        return "Mean_Norm(dim=%d)" % self.dim

class L2_Norm(nn.Module):

    def __init__(self, alpha=1.):
        super(L2_Norm, self).__init__()
        self.alpha = alpha

    def forward(self, input):
        # alpha = log(p * ( class -2) / (1-p))

        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output * self.alpha

    def __repr__(self):
        return "L2_Norm(alpha=%f)" % self.alpha

class MultiResNet(nn.Module):
    """
    Define the ResNet model with A-softmax and AM-softmax loss.
    Added dropout as https://github.com/nagadomi/kaggle-cifar10-torch7 after average pooling and fc layer.
    """

    def __init__(self, embedding_size, num_classes_a=1951, num_classes_b=1211, block=BasicBlock, input_dim=161,
                 resnet_size=8, channels=[16, 64, 128, 256], dropout_p=0.25, stride=1, fast=False,
                 inst_norm=False, alpha=12, input_norm='Mean', transform=False,
                 avg_size=4, kernal_size=5, padding=2, **kwargs):

        super(MultiResNet, self).__init__()
        resnet_type = {8: [1, 1, 1, 0],
                       10: [1, 1, 1, 1],
                       18: [2, 2, 2, 2],
                       34: [3, 4, 6, 3],
                       50: [3, 4, 6, 3],
                       101: [3, 4, 23, 3]}

        layers = resnet_type[resnet_size]
        self.alpha = alpha
        self.layers = layers
        self.dropout_p = dropout_p
        self.embedding_size = embedding_size
        self.relu = nn.ReLU(inplace=True)
        self.transform = transform
        self.fast = fast
        self.input_norm = input_norm
        self.inst_layer = Mean_Norm()

        self.inplanes = channels[0]
        self.conv1 = nn.Conv2d(1, channels[0], kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])

        self.maxpool = None

        # self.maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=1)
        self.layer1 = self._make_layer(block, channels[0], layers[0])

        self.inplanes = channels[1]
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=kernal_size, stride=2,
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.layer2 = self._make_layer(block, channels[1], layers[1])

        self.inplanes = channels[2]
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=kernal_size, stride=2,
                               padding=padding, bias=False)
        self.bn3 = nn.BatchNorm2d(channels[2])
        self.layer3 = self._make_layer(block, channels[2], layers[2])

        if layers[3] != 0:
            assert len(channels) == 4
            self.inplanes = channels[3]
            self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=kernal_size, stride=2,
                                   padding=padding, bias=False)
            self.bn4 = nn.BatchNorm2d(channels[3])
            self.layer4 = self._make_layer(block=block, planes=channels[3], blocks=layers[3])

        self.dropout = nn.Dropout(self.dropout_p)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, avg_size))

        self.fc = nn.Sequential(
            nn.Linear(self.inplanes * avg_size, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size)
        )
        self.trans_layer = None

        if self.alpha:
            self.l2_norm = L2_Norm(self.alpha)

        self.classifier_a = nn.Linear(self.embedding_size, num_classes_a)
        self.classifier_b = nn.Linear(self.embedding_size, num_classes_b)

        for m in self.modules():  # 对于各层参数的初始化
            if isinstance(m, nn.Conv2d):  # 以2/n的开方为标准差，做均值为0的正态分布
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):  # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.inst_layer != None:
            x = self.inst_layer(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool != None:
            x = self.maxpool(x)

        x = self.layer1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.layer3(x)

        if self.layers[3] != 0:
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu(x)
            x = self.layer4(x)

        if self.dropout_p > 0:
            x = self.dropout(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        embeddings = self.fc(x)

        if self.trans_layer != None:
            embeddings = self.trans_layer(embeddings)

        if self.alpha:
            embeddings = self.l2_norm(embeddings)

        return '', embeddings

    def cls_forward(self, a, b):

        logits_a = self.classifier_a(a)
        logits_b = self.classifier_b(b)

        return logits_a, logits_b


def ExtractVector(data_dir, xvector_dir, model_path):
    """

    :param data_dir: 输入目录
    :param xvector_dir: 输出npy目录
    :param model: 模型文件路径
    :param enroll: 1为建模，0为识别
    :return: 更新的vector npy的数目
    """
    try:
        data_dir = data_dir.replace('\\', '/')
        xvector_dir = xvector_dir.replace('\\', '/')
    except:
        pass

    model = MultiResNet(embedding_size=128, resnet_size=10, num_classes_a=1951, num_classes_b=1211,
                        channels=[16, 64, 128, 256], alpha=12, dropout_p=0.25)
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location='cpu')
    try:

        filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}
        model.load_state_dict(filtered)

    except Exception as e:
        print(e)
        raise e

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    uid2feat_path = os.path.join(data_dir, 'feat.txt') # Todo: uid utterance_vector.npy 文件名
    the_dataset = ExtractDataset(uid2feat_path=uid2feat_path)
    # nj = max(1, int(np.log2(len(the_dataset))))

    nw = int(min(max(1, np.log2(len(the_dataset))), 4))
    print("Try to load with %d workers..." % nw)
    kwargs = {
        'num_workers': nw,
        'pin_memory': False
    }
    dataloader = torch.utils.data.DataLoader(dataset=the_dataset, batch_size=1, shuffle=False, **kwargs)

    if not os.path.exists(xvector_dir):
        os.makedirs(xvector_dir)

    pbar = tqdm(enumerate(dataloader))
    uid2vectors = {}
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 这里的0是GPU id
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

    batch_size = 80 if torch.cuda.is_available() else 32
    batch_size = min(meminfo.free/1024/1024/1024*18, batch_size)

    with torch.no_grad():
        data = torch.tensor([])
        num_seg_tensor = [0]
        uid_lst = []

        for batch_idx, (a_uid, a_data) in pbar:
            vec_shape = a_data.shape

            if vec_shape[1] != 1:
                a_data = a_data.reshape(vec_shape[0] * vec_shape[1], 1, vec_shape[2], vec_shape[3])

            data = torch.cat((data, a_data), dim=0)
            num_seg_tensor.append(num_seg_tensor[-1] + len(a_data))
            uid_lst.append(a_uid[0])

            if data.shape[0] >= batch_size or batch_idx + 1 == len(dataloader):
                try:
                    data = data.cuda() if next(model.parameters()).is_cuda else data
                    _, out = model(data)
                    out = out.data.cpu().float().numpy()

                    # print(out.shape)
                    if len(out.shape) == 3:
                        out = out.squeeze(0)

                    for i, uid in enumerate(uid_lst):
                        uid2vectors[uid] = out[num_seg_tensor[i]:num_seg_tensor[i + 1]]  # , uid[0])

                    data = torch.tensor([])
                    num_seg_tensor = [0]
                    uid_lst = []

                except Exception as e:
                    print(e, uid_lst, data.shape)
                    traceback.print_exc()

    uids = list(uid2vectors.keys())
    # 更新已有uid2vec列表
    sid2vec_scp = []
    scp_file = '/'.join((data_dir, 'utt2xve')) # Todo: uid spk_vector.npy 文件名

    num_update = 0
    for uid in uids:
        try:
            #xve, sid = uid2vectors[uid]
            xve = uid2vectors[uid]
            vec_path = '/'.join((xvector_dir, '%s.npy' % uid))
            np.save(vec_path, xve)
            num_update += 1
            sid2vec_scp.append((uid, vec_path))

        except Exception as e:
            print('Saving %s' % uid, e)

    with open(scp_file, 'w') as scp:
        for sid, vec_path in sid2vec_scp:
            scp.write('%s %s\n' % (sid, vec_path))
    try:
        torch.cuda.empty_cache()
    except:
        pass

    return num_update

if __name__ == '__main__':
    a = []
    for i in range(1, len(sys.argv)):
        a.append((str(sys.argv[i])))
    print("Try to extract: ", a)
    num_update = ExtractVector(a[0], a[1], a[2])
    print("Extracted %d xvectors for utterances." % num_update)



# /Users/yang/PycharmProjects/army_speaker_tmp/sv_1128/shengwen/db/txt
# /Users/yang/PycharmProjects/army_speaker_tmp/sv_1128/shengwen/db/upload/npy2
# /Users/yang/PycharmProjects/army_speaker_tmp/sv_1128/shengwen/pth/model.pth
# resources/models/checkpoint_25.pth