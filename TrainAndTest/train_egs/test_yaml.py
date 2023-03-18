#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: test_accuracy.py
@Time: 19-8-6 下午1:29
@Overview: Train the resnet 10 with asoftmax.
"""
from __future__ import print_function
import pickle
import random
import argparse
import os
import pdb
import sys
import time
# Version conflict
import warnings
from collections import OrderedDict

import kaldi_io
import kaldiio
import numpy as np
import psutil
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml

# from Define_Model.Loss.SoftmaxLoss import AngleLinear, AdditiveMarginLinear
from Eval.eval_metrics import evaluate_kaldi_eer, evaluate_kaldi_mindcf
from Process_Data.Datasets.KaldiDataset import ScriptTrainDataset, ScriptValidDataset, KaldiExtractDataset, \
    ScriptVerifyDataset
from Process_Data.audio_processing import ConcateOrgInput, ConcateVarInput, MelFbank, mvnormal, read_WaveFloat, read_WaveInt
from TrainAndTest.common_func import create_model, verification_extract, load_model_args, args_model, args_parse
from logger import NewLogger

warnings.filterwarnings("ignore")

import torch._utils

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor

    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

args = args_parse('PyTorch Speaker Recognition: Extraction, Test')

# Set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.multiprocessing.set_sharing_strategy('file_system')

if args.cuda:
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

# load train config file args.train_config
assert os.path.exists(args.train_config)

with open(args.train_config, 'r') as f:
    # config_args = yaml.load(f, Loader=yaml.FullLoader)
    config_args = load_hyperpyyaml(f)

# create logger
# Define visulaize SummaryWriter instance
kwargs = {'num_workers': args.nj, 'pin_memory': False} if args.cuda else {}
assert os.path.isfile(args.resume), print(args.resume)

sys.stdout = NewLogger(os.path.join(os.path.dirname(args.resume), 'test.log'))

l2_dist = nn.CosineSimilarity(dim=-1, eps=1e-6) if args.cos_sim else nn.PairwiseDistance(2)

if args.test_input == 'var':
    transform = transforms.Compose([
        ConcateOrgInput(remove_vad=args.remove_vad),
    ])
    transform_T = transforms.Compose([
        ConcateOrgInput(remove_vad=args.remove_vad),
    ])

elif args.test_input == 'fix':
    transform = transforms.Compose([
        ConcateVarInput(num_frames=args.chunk_size, frame_shift=args.frame_shift, remove_vad=args.remove_vad),
    ])
    transform_T = transforms.Compose([
        ConcateVarInput(num_frames=args.chunk_size, frame_shift=args.frame_shift, remove_vad=args.remove_vad),
    ])
else:
    raise ValueError('input length must be var or fix.')

if args.mvnorm:
    transform.transforms.append(mvnormal())
    transform_T.transforms.append(mvnormal())

if 'trans_fbank' in config_args and config_args['trans_fbank']:
    transform.transforms.append(
        MelFbank(num_filter=config_args['input_dim']))
    transform_T.transforms.append(
        MelFbank(num_filter=config_args['input_dim']))

# pdb.set_trace()
feat_type = 'kaldi'
if config_args['feat_format'] == 'kaldi':
    # file_loader = read_mat
    file_loader = kaldiio.load_mat
    torch.multiprocessing.set_sharing_strategy('file_system')
elif config_args['feat_format'] == 'npy':
    file_loader = np.load
elif config_args['feat_format'] == 'wav':
    file_loader = read_WaveInt if config_args['feat'] == 'int' else read_WaveFloat
    feat_type = 'wav'

if not args.valid:
    args.num_valid = 0

train_dir = ScriptTrainDataset(dir=args.train_dir, samples_per_speaker=args.input_per_spks, loader=file_loader,
                               feat_type=feat_type,
                               transform=transform, num_valid=args.num_valid, verbose=args.verbose)

if args.score_norm != '' and os.path.isdir(args.train_extract_dir):
    train_extract_dir = KaldiExtractDataset(dir=args.train_extract_dir, transform=transform_T, filer_loader=file_loader,
                                            feat_type=feat_type,
                                            verbose=args.verbose, trials_file='')

verfify_dir = KaldiExtractDataset(dir=args.test_dir, transform=transform_T, filer_loader=file_loader,
                                  feat_type=feat_type,
                                  verbose=args.verbose)


def test(test_loader, xvector_dir, test_cohort_scores=None):
    # switch to evaluate mode
    labels, distances = [], []
    l_batch = []
    d_batch = []
    pbar = tqdm(enumerate(test_loader)) if args.verbose > 0 else enumerate(test_loader)
    for batch_idx, this_batch in pbar:
        if test_cohort_scores != None:
            data_a, data_p, label, uid_a, uid_b = this_batch
        else:
            data_a, data_p, label = this_batch

        data_a = torch.tensor(data_a)  # .cuda()  # .view(-1, 4, embedding_size)
        data_p = torch.tensor(data_p)  # .cuda()  # .view(-

        # if out_p.shape[-1] != args.embedding_size:
        #     out_p = out_p.reshape(-1, args.embedding_size)

        # if args.cluster == 'mean':
        #     out_a = out_a.mean(dim=0, keepdim=True)
        #     out_p = out_p.mean(dim=0, keepdim=True)
        #
        # elif args.cluster == 'cross':
        #     out_a_first = out_a.shape[0]
        #     out_a = out_a.repeat(out_p.shape[0], 1)
        #     out_p = out_p.reshape(out_a_first, 1)
        if len(data_a.shape) == 3:
            data_a_dim1 = data_a.shape[1]
            data_a = data_a.repeat_interleave(data_p.shape[1], dim=1)
            data_p = data_p.repeat_interleave(data_a_dim1, dim=1)
        #
        # else:
        # dists = (data_a[:, :, None] - data_p[:]).norm(p=2, dim=-1)

        # print(dists.shape)
        # pdb.set_trace()
        dists = l2_dist(data_a, data_p)

        if len(dists.shape) == 3:
            dists = dists.mean(dim=-1).mean(dim=-1)
        elif len(dists.shape) == 2:
            dists = dists.mean(dim=-1)

        dists = dists.numpy()
        label = label.numpy()

        if test_cohort_scores != None:
            enroll_mean_std = np.array([test_cohort_scores[uid] for uid in uid_a])

            mean_e_c = enroll_mean_std[:, 0]
            std_e_c = enroll_mean_std[:, 1]

            test_mean_std = np.array([test_cohort_scores[uid] for uid in uid_b])

            mean_t_c = test_mean_std[:, 0]
            std_t_c = test_mean_std[:, 1]
            # [test_cohort_scores[uid] for uid in uid_b]

            if args.score_norm == "z-norm":
                dists = (dists - mean_e_c) / std_e_c
            elif args.score_norm == "t-norm":
                dists = (dists - mean_t_c) / std_t_c
            elif args.score_norm in ["s-norm", "as-norm"]:
                score_e = (dists - mean_e_c) / std_e_c
                score_t = (dists - mean_t_c) / std_t_c
                dists = 0.5 * (score_e + score_t)

        if len(dists) == 1:
            d_batch.append(float(dists[0]))
            l_batch.append(label[0])

            if len(l_batch) >= 128 or len(test_loader.dataset) == (batch_idx + 1):
                distances.append(d_batch)
                labels.append(l_batch)

                l_batch = []
                d_batch = []
        else:
            distances.append(dists)
            labels.append(label)

        if args.verbose > 0 and batch_idx % args.log_interval == 0:
            pbar.set_description('Test: [{}/{} ({:.0f}%)]'.format(
                batch_idx * len(data_a), len(test_loader.dataset), 100. * batch_idx / len(test_loader)))

        del data_a, data_p

    # pdb.set_trace()
    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    time_stamp = time.strftime("%Y.%m.%d.%X", time.localtime()) if args.score_suffix == '' else args.score_suffix

    score_file = os.path.join(xvector_dir, '%sscore.' % args.score_norm + time_stamp)
    with open(score_file, 'w') as f:
        for l in zip(labels, distances):
            f.write(" ".join([str(i) for i in l]) + '\n')

    # pdb.set_trace()
    eer, eer_threshold, accuracy = evaluate_kaldi_eer(distances, labels, cos=args.cos_sim, re_thre=True)
    mindcf_01, mindcf_001 = evaluate_kaldi_mindcf(distances, labels)

    dist_type = 'cos' if args.cos_sim else 'l2'
    test_directorys = args.test_dir.split('/')
    test_set_name = '-'
    for i, dir in enumerate(test_directorys):
        if dir == 'data':
            try:
                test_subset = test_directorys[i + 3].split('_')[0]
            except Exception as e:
                test_subset = test_directorys[i + 2].split('_')[0]

            test_set_name = "-".join((test_directorys[i + 1], test_subset))
    if args.score_suffix != '':
        test_set_name = '-'.join((test_set_name, args.score_suffix[:7]))

    result_str = ''
    if args.verbose > 0:
        result_str += 'For %s_distance, %d pairs:\n' % (dist_type, len(labels))
    result_str += '\33[91m'
    if args.verbose > 0:
        result_str += '+-------------------+-------------+-------------+---------------+---------------+-------------------+\n'

        result_str += '|{: ^19s}|{: ^13s}|{: ^13s}|{: ^15s}|{: ^15s}|{: ^19s}|\n'.format('Test Set',
                                                                                         'EER (%)',
                                                                                         'Threshold',
                                                                                         'MinDCF-0.01',
                                                                                         'MinDCF-0.001',
                                                                                         'Date')
    if args.verbose > 0:
        result_str += '+-------------------+-------------+-------------+---------------+---------------+-------------------+\n'

    eer = '{:.4f}'.format(eer * 100.)
    threshold = '{:.4f}'.format(eer_threshold)
    mindcf_01 = '{:.4f}'.format(mindcf_01)
    mindcf_001 = '{:.4f}'.format(mindcf_001)
    date = time.strftime("%Y%m%d %H:%M:%S", time.localtime())

    result_str += '|{: ^19s}|{: ^13s}|{: ^13s}|{: ^15s}|{: ^15s}|{: ^19s}|'.format(test_set_name,
                                                                                   eer,
                                                                                   threshold,
                                                                                   mindcf_01,
                                                                                   mindcf_001,
                                                                                   date)
    if args.verbose > 0:
        result_str += '\n+-------------------+-------------+-------------+---------------+---------------+-------------------+\n'
    result_str += '\33[0m'

    print(result_str)

    result_file = os.path.join(xvector_dir, '%sresult.' % args.score_norm + time_stamp)
    with open(result_file, 'w') as f:
        f.write(result_str)


def cohort(train_xvectors_dir, test_xvectors_dir):
    train_xvectors_scp = os.path.join(train_xvectors_dir, 'xvectors.scp')
    test_xvectors_scp = os.path.join(test_xvectors_dir, 'xvectors.scp')

    assert os.path.exists(train_xvectors_scp)
    assert os.path.exists(test_xvectors_scp)

    train_stats = {}

    train_vectors = []
    train_scps = []
    with open(train_xvectors_scp, 'r') as f:
        for l in f.readlines():
            uid, vpath = l.split()
            train_scps.append((uid, vpath))

    random.shuffle(train_scps)

    if args.n_train_snts < len(train_scps):
        train_scps = train_scps[:args.n_train_snts]

    for (uid, vpath) in train_scps:
        train_vectors.append(file_loader(vpath))

    train_vectors = torch.tensor(train_vectors).cuda()
    if args.cos_sim:
        train_vectors = train_vectors / train_vectors.norm(p=2, dim=1).unsqueeze(1)

    with open(test_xvectors_scp, 'r') as f:
        pbar = tqdm(f.readlines(), ncols=100) if args.verbose > 0 else f.readlines()

        for l in pbar:
            uid, vpath = l.split()

            test_vector = torch.tensor(file_loader(vpath))
            # pdb.set_trace()
            if args.cos_sim:
                test_vector = test_vector.cuda()
                scores = torch.matmul(train_vectors, test_vector / test_vector.norm(p=2))

                if args.score_norm == "as-norm":
                    scores = torch.topk(scores, k=args.cohort_size, dim=0)[0]
            else:
                test_vector = test_vector.repeat(train_vectors.shape[0], 1).cuda()
                scores = l2_dist(test_vector, train_vectors)

                if args.score_norm == "as-norm":
                    scores = -torch.topk(-scores, k=args.cohort_size, dim=0)[0]

            mean_t_c = torch.mean(scores, dim=0).cpu()
            std_t_c = torch.std(scores, dim=0).cpu()

            train_stats[uid] = [mean_t_c, std_t_c]

    with open(test_xvectors_dir + '/cohort_%d_%d.pickle' % (args.n_train_snts, args.cohort_size), 'wb') as f:
        pickle.dump(train_stats, f, protocol=pickle.HIGHEST_PROTOCOL)

    # pickle.dump(train_stats, test_xvectors_dir)

    return train_stats


if __name__ == '__main__':

    # Views the training images and displays the distance on anchor-negative and anchor-positive
    # test_display_triplet_distance = False
    # print the experiment configuration
    if args.verbose > 0:
        print('\nCurrent time is \33[91m{}\33[0m.'.format(str(time.asctime())))
    opts = vars(args)
    keys = list(opts.keys())
    keys.sort()

    options = []
    for k in keys:
        options.append("\'%s\': \'%s\'" % (str(k), str(opts[k])))

    if args.verbose > 1:
        print('Parsed options: \n{ %s }' % (', '.join(options)))
        print('Number of Speakers: {}.\n'.format(train_dir.num_spks))

    start_time = time.time()
    test_xvector_dir = os.path.join(args.xvector_dir, 'test')
    train_xvector_dir = os.path.join(args.xvector_dir, 'train')

    if args.valid or args.extract:
        
        model = config_args['embedding_model']
        model.classifier = config_args['classifier']

        if args.verbose > 0:
            print('=> loading checkpoint {}'.format(args.resume))
        
        checkpoint = torch.load(args.resume)
        # start_epoch = checkpoint['epoch']

        checkpoint_state_dict = checkpoint['state_dict']
        start = checkpoint['epoch'] if 'epoch' in checkpoint else args.start_epoch
        if args.verbose > 0:
            print('Epoch is : ' + str(start))

        if isinstance(checkpoint_state_dict, tuple):
            checkpoint_state_dict = checkpoint_state_dict[0]
        filtered = {k: v for k, v in checkpoint_state_dict.items() if 'num_batches_tracked' not in k}

        # filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}
        if list(filtered.keys())[0].startswith('module'):
            new_state_dict = OrderedDict()
            for k, v in filtered.items():
                name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，去掉module.
                new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
            filtered = new_state_dict

        if 'fc1.1.weight' in filtered:
            model.fc1 = nn.Sequential(
                nn.Linear(model.encoder_output, model.embedding_size),
                nn.BatchNorm1d(model.embedding_size)
            )

        model_dict = model.state_dict()
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        
        # print(model)
        if args.cuda:
            model.cuda()

        del train_dir  # , valid_dir
        if args.verbose > 0:
            print('Memery Usage: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))

        if args.extract:
            if args.score_norm != '':
                train_verify_loader = torch.utils.data.DataLoader(train_extract_dir, batch_size=args.test_batch_size,
                                                                  shuffle=False, **kwargs)
                verification_extract(train_verify_loader, model, xvector_dir=train_xvector_dir, epoch=start,
                                     test_input=args.test_input, ark_num=50000, gpu=True, verbose=args.verbose,
                                     mean_vector=args.mean_vector,
                                     xvector=args.xvector)

            verify_loader = torch.utils.data.DataLoader(verfify_dir, batch_size=args.test_batch_size, shuffle=False,
                                                        **kwargs)

            # extract(verify_loader, model, args.xvector_dir)
            verification_extract(verify_loader, model, xvector_dir=test_xvector_dir, epoch=start,
                                 test_input=args.test_input, ark_num=50000, gpu=True, verbose=args.verbose,
                                 mean_vector=args.mean_vector,
                                 xvector=args.xvector)

    if args.test:
        file_loader = kaldiio.load_mat
        # file_loader = read_vec_flt
        return_uid = True if args.score_norm != '' else False
        test_dir = ScriptVerifyDataset(dir=args.test_dir, trials_file=args.trials, xvectors_dir=test_xvector_dir,
                                       loader=file_loader, return_uid=return_uid)

        test_loader = torch.utils.data.DataLoader(test_dir,
                                                  batch_size=1 if not args.mean_vector else args.test_batch_size * 128,
                                                  shuffle=False, **kwargs)

        train_stats_pickle = os.path.join(test_xvector_dir,
                                          'cohort_%d_%d.pickle' % (args.n_train_snts, args.cohort_size))

        if args.score_norm == '':
            train_stats = None
        elif os.path.isfile(train_stats_pickle):
            with open(train_stats_pickle, 'rb') as f:
                train_stats = pickle.load(f)
        else:
            train_stats = cohort(train_xvector_dir, test_xvector_dir)

        test(test_loader, xvector_dir=args.xvector_dir, test_cohort_scores=train_stats)

    stop_time = time.time()
    t = float(stop_time - start_time)
    if args.verbose > 0:
        print("Running %.4f minutes for testing.\n" % (t / 60))

