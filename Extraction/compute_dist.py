#!/usr/bin/env python
# encoding: utf-8
'''
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: VS Code
@File: compute_dist.py
@Time: 2022/12/30 11:32
@Overview: 
'''

import argparse
import torch
import kaldiio

parser = argparse.ArgumentParser(
    description='Extract probilities for x-vector during training.')
parser.add_argument('--data-dir', type=str, help='path to dataset')
parser.add_argument('--checkpoint', type=str, help='path to dataset')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


ckp = torch.load(args.checkpoint)
model_dict = ckp['state_dict']
classifier_centers = model_dict['classifier.W']  # torch.Size([256, 797])
classifier_centers = torch.nn.functional.normalize(classifier_centers, dim=0)
classifier_centers = classifier_centers.cpu()

vec_dict = kaldiio.load_scp(args.data_dir + '/xvectors.scp')
sim_ark = args.data_dir + + '/sim.ark.gz'

with kaldiio.WriteHelper('ark:| gzip -c > ' + sim_ark) as writer:

    pbar = tqdm(vec_dict, ncols=100)
    for k in pbar:
        this_vec = torch.tensor(vec_dict[k]).reshape(1, -1)
        this_vec = torch.nn.functional.normalize(this_vec, dim=1)
        similarities = torch.matmul(this_vec, classifier_centers)

        writer(str(k), similarities.squeeze().numpy())
