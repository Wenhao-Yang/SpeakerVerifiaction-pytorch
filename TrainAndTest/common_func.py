#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: common_func.py
@Time: 2019/12/16 6:36 PM
@Overview:
"""
import os

import kaldi_io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Define_Model.CNN import AlexNet
from Define_Model.ResNet import LocalResNet, ResNet20, ThinResNet, ResNet, SimpleResNet, DomainResNet, GradResNet, \
    TimeFreqResNet, MultiResNet
from Define_Model.TDNN.DTDNN import DTDNN
from Define_Model.TDNN.ETDNN import ETDNN_v4, ETDNN, ETDNN_v5
from Define_Model.TDNN.FTDNN import FTDNN
from Define_Model.TDNN.TDNN import TDNN_v2, TDNN_v4, TDNN_v5
from Eval.eval_metrics import evaluate_kaldi_eer, evaluate_kaldi_mindcf


def create_optimizer(parameters, optimizer, **kwargs):
    # setup optimizer
    parameters = filter(lambda p: p.requires_grad, parameters)
    if optimizer == 'sgd':
        opt = optim.SGD(parameters,
                        lr=kwargs['lr'],
                              momentum=kwargs['momentum'],
                              dampening=kwargs['dampening'],
                              weight_decay=kwargs['weight_decay'])

    elif optimizer == 'adam':
        opt = optim.Adam(parameters,
                               lr=kwargs['lr'],
                               weight_decay=kwargs['weight_decay'])

    elif optimizer == 'adagrad':
        opt = optim.Adagrad(parameters,
                                  lr=kwargs['lr'],
                                  lr_decay=kwargs['lr_decay'],
                                  weight_decay=kwargs['weight_decay'])
    elif optimizer == 'RMSprop':
        opt = optim.RMSprop(parameters,
                            lr=kwargs['lr'],
                            momentum=kwargs['momentum'],
                            weight_decay=kwargs['weight_decay'])

    return opt


# ALSTM  ASiResNet34  ExResNet34  LoResNet  ResNet20  SiResNet34  SuResCNN10  TDNN

__factory = {
    'AlexNet': AlexNet,
    'LoResNet': LocalResNet,
    'DomResNet': DomainResNet,
    'ResNet20': ResNet20,
    'SiResNet34': SimpleResNet,
    'ThinResNet': ThinResNet,
    'MultiResNet': MultiResNet,
    'ResNet': ResNet,
    'DTDNN': DTDNN,
    'TDNN': TDNN_v2,
    'TDNN_v4': TDNN_v4,
    'TDNN_v5': TDNN_v5,
    'ETDNN': ETDNN,
    'ETDNN_v4': ETDNN_v4,
    'ETDNN_v5': ETDNN_v5,
    'FTDNN': FTDNN,
    'GradResNet': GradResNet,
    'TimeFreqResNet': TimeFreqResNet
}


def create_model(name, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))

    return __factory[name](**kwargs)


class AverageMeter(object):
    """Computes and stores the average and current value.
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def l2_alpha(C):
#     return np.log(0.99 * (C - 2) / (1 - 0.99))

def verification_extract(extract_loader, model, xvector_dir, epoch, ark_num=50000, gpu=True):
    model.eval()

    if not os.path.exists(xvector_dir):
        os.makedirs(xvector_dir)
        # print('Creating xvector path: %s' % xvector_dir)

    # pbar = tqdm(enumerate(extract_loader))
    uid2vectors = {}
    data = torch.tensor([])
    num_seg_tensor = [0]
    uid_lst = []

    batch_size = 128 if torch.cuda.is_available() else 80
    with torch.no_grad():
        for batch_idx, (a_data, a_uid) in enumerate(extract_loader):
            vec_shape = a_data.shape

            if vec_shape[1] != 1:
                a_data = a_data.reshape(vec_shape[0] * vec_shape[1], 1, vec_shape[2], vec_shape[3])

            data = torch.cat((data, a_data), dim=0)
            num_seg_tensor.append(num_seg_tensor[-1] + len(a_data))
            uid_lst.append(a_uid[0])

            if data.shape[0] >= batch_size or batch_idx + 1 == len(extract_loader):

                data = data.cuda() if next(model.parameters()).is_cuda else data
                model_out = model(data)
                try:
                    _, out, _, _ = model_out
                except:
                    _, out = model_out

                out = out.data.cpu().float().numpy()
                # print(out.shape)
                if len(out.shape) == 3:
                    out = out.squeeze(0)

                for i, uid in enumerate(uid_lst):
                    uid2vectors[uid] = out[num_seg_tensor[i]:num_seg_tensor[i + 1]].mean(axis=0)  # , uid[0])

                data = torch.tensor([])
                num_seg_tensor = [0]
                uid_lst = []

    uids = list(uid2vectors.keys())
    # print('There are %d vectors' % len(uids))
    scp_file = xvector_dir + '/xvectors.scp'
    scp = open(scp_file, 'w')

    # write scp and ark file
    # pdb.set_trace()
    for set_id in range(int(np.ceil(len(uids) / ark_num))):
        ark_file = xvector_dir + '/xvector.{}.ark'.format(set_id)
        with open(ark_file, 'wb') as ark:
            ranges = np.arange(len(uids))[int(set_id * ark_num):int((set_id + 1) * ark_num)]
            for i in ranges:
                key = uids[i]
                vec = uid2vectors[key]
                len_vec = len(vec.tobytes())

                kaldi_io.write_vec_flt(ark, vec, key=key)
                # print(ark.tell())
                scp.write(str(uids[i]) + ' ' + str(ark_file) + ':' + str(ark.tell() - len_vec - 10) + '\n')
    scp.close()
    # print('Saving %d xvectors to %s' % (len(uids), xvector_dir))
    torch.cuda.empty_cache()


def verification_test(test_loader, dist_type, log_interval, xvector_dir, epoch):
    # switch to evaluate mode
    labels, distances = [], []
    dist_fn = nn.CosineSimilarity(dim=1).cuda() if dist_type == 'cos' else nn.PairwiseDistance(2)

    # pbar = tqdm(enumerate(test_loader))
    with torch.no_grad():
        for batch_idx, (data_a, data_p, label) in enumerate(test_loader):
            out_a = torch.tensor(data_a).cuda()  # .view(-1, 4, embedding_size)
            out_p = torch.tensor(data_p).cuda()  # .view(-1, 4, embedding_size)
            dists = dist_fn.forward(out_a, out_p).cpu().numpy()

            distances.append(dists)
            labels.append(label.numpy())
            del out_a, out_p  # , ae, pe

        # if batch_idx % log_interval == 0:
        #     pbar.set_description('Verification Epoch {}: [{}/{} ({:.0f}%)]'.format(
        #         epoch, batch_idx * len(data_a), len(test_loader.dataset), 100. * batch_idx / len(test_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])
    # this_xvector_dir = "%s/epoch_%s" % (xvector_dir, epoch)
    with open('%s/scores' % xvector_dir, 'w') as f:
        for d, l in zip(distances, labels):
            f.write(str(d) + ' ' + str(l) + '\n')

    eer, eer_threshold, accuracy = evaluate_kaldi_eer(distances, labels,
                                                      cos=True if dist_type == 'cos' else False,
                                                      re_thre=True)
    mindcf_01, mindcf_001 = evaluate_kaldi_mindcf(distances, labels)

    return eer, eer_threshold, mindcf_01, mindcf_001
