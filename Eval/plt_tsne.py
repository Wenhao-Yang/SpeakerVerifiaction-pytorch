#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: plt_tsne.py
@Time: 2020/10/20 15:25
@Overview:
"""
import argparse
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
from kaldiio import ReadHelper
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Training settings
from Process_Data.constants import cValue_1
from matplotlib.font_manager import FontProperties
from matplotlib.font_manager import FontManager
parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
# Data options
parser.add_argument('--scp-file', type=str, default='Data/xvector/LoResNet8/vox1/spect_egs/arcsoft_dp25/xvectors.scp',
                    help='path to scp file for xvectors')
parser.add_argument('--sid-length', default=7, type=int,
                    help='num of speakers to plot (default: 10)')
parser.add_argument('--num-spk', default=7, type=int,
                    help='num of speakers to plot (default: 10)')
parser.add_argument('--pca-dim', default=0, type=int, help='num of speakers to plot (default: 10)')
parser.add_argument('--out-pdf', default='', type=str, help='num of speakers to plot (default: 10)')
parser.add_argument('--distance', default='l2', type=str, help='num of speakers to plot (default: 10)')
parser.add_argument('--hard-vector', type=str, help='num of speakers to plot (default: 10)')
parser.add_argument('--plot-legend', action='store_true', default=False, help='num of speakers to plot (default: 10)')

args = parser.parse_args()


def cos(x, y):
    costh = x * y / (np.sqrt((x ** 2).sum()) * np.sqrt((y ** 2).sum()))
    return 1 - costh.sum()


def l2(x, y):
    return np.sqrt(((x - y) ** 2).sum())


if __name__ == '__main__':

    vects = {}
    with ReadHelper('scp:%s' % args.scp_file) as reader:
        for key, numpy_array in reader:
            vects[key] = numpy_array

    spks = set([])
    for key in vects:
        s = key[:args.sid_length]
        spks.add(s)

    spks = list(spks)
    spks.sort()

    hard_uids = set([])
    if os.path.exists(args.hard_vector):
        with open(args.hard_vector, 'r') as f:
            for uid in f.readlines():
                hard_uids.add(uid.rstrip('\n'))
        print('The number of hard utterances is ', len(hard_uids))
    else:
        print('There is no hard samples: ', args.hard_vector)

    spk2vec = {}
    for s in spks:
        spk2vec[s] = []

    # pdb.set_trace()
    skip = 0
    for key in vects:
        # if key[:args.sid_length] in spks_this:
        this_vec = vects[key]
        vec_len = len(this_vec)
        if (len(hard_uids) > 0 and key in hard_uids) or len(hard_uids) == 0:
            spk2vec[key[:args.sid_length]].append(this_vec.reshape(1, vec_len))
        else:
            skip += 1

    print('Skip vectors: ', skip)
    spks_this = []
    for spk in spks:
        if len(spk2vec[spk]) > 0:
            spks_this.append(spk)
            if len(spks_this) >= args.num_spk:
                break
    # spks_this = spks[:args.num_spk] if len(spks) > args.num_spk else spks
    all_vectors = []
    all_len = [0]

    if args.num_spk > 0:
        for spk in spks_this:
            if len(spk2vec[spk]) > 0:
                spk_con = np.concatenate(spk2vec[spk])
                all_len.append(len(spk_con))
                all_vectors.append(spk_con)

        all_vectors = np.concatenate(all_vectors, axis=0)

        if args.pca_dim > 0:
            print('PCA... dimension is reduced to ', args.pca_dim)
            pca = PCA(n_components=args.pca_dim)
            all_vectors = pca.fit_transform(all_vectors)

        S_embedded = TSNE(n_components=2, metric='cosine').fit_transform(all_vectors)
        emb_group = []
        for i in range(len(all_len) - 1):
            start = np.sum(all_len[:(i + 1)]).astype(np.int32)
            stop = np.sum(all_len[:(i + 2)]).astype(np.int32)
            this_points = S_embedded[start:stop]
            assert len(this_points) > 0, 'start:stop is %s:%s' % (start, stop)
            emb_group.append(this_points)

        plot_legend = args.plot_legend
        fig_width = 6 if plot_legend else 5
        plt.figure(figsize=(fig_width, 4))
        font_manager = FontManager()
        font_manager.addfont(path='/home/yangwenhao/font/TimesNewRoman.ttf')
        plt.rcParams['font.sans-serif'] = 'Times New Roman'

        # FontManager.addfont('/home/yangwenhao/font/TimesNewRoman.ttf')
        #
        # plt.rcParams['font.sans-serif'] = 'Times New Roman'
        # font = FontProperties(fname='/home/yangwenhao/font/TimesNewRoman.ttf')
        # plt.rc('font', family='Times New Roman', weight='semibold')

        leng = []
        for idx, group in enumerate(emb_group):
            if len(group) > 0:
                c = cValue_1[idx]
                leng.append(spks_this[idx])
                plt.scatter(group[:, 0], group[:, 1], color=c, s=25, alpha=0.4)

        # plt.legend(leng, loc="best", fontsize=10, bbox_to_anchor=(1, 1.05), borderaxespad=1)  # , fontproperties=font)
        # plt.xlim([-20, 20])
        # plt.ylim([-20, 20])
        if plot_legend:
            print('Plot legend...')
            plt.legend(leng, loc="best", fontsize=14, bbox_to_anchor=(1, 1.05),
                       borderaxespad=1)  # , fontproperties=font)
            # plt.xlim([-20, 20])
            # plt.ylim([-20, 20])
            plt.subplots_adjust(right=5 / 6)

        # plt.xticks(fontsize=16, fontproperties=font)
        # plt.yticks(fontsize=16, fontproperties=font)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        if args.out_pdf.endswith('pdf'):
            plt.savefig(args.out_pdf, format="pdf")

        plt.show()

    if args.distance == 'l2':
        dist_fn = l2
    elif args.distance == 'cos':
        dist_fn = cos

    within_vars = []
    spk2num_utt = []
    for spk in spk2vec:
        if len(spk2vec[spk]) > 0:
            spk_mean = np.mean(spk2vec[spk], axis=0)
            within_var = 0
            for vec in spk2vec[spk]:
                within_var += dist_fn(vec, spk_mean) ** 2
            within_var /= len(spk2vec[spk])

            # within_var = np.var(spk2vec[spk], axis=0).sum()
            within_vars.append(within_var)
            spk2num_utt.append(len(spk2vec[spk]))

    within_var = np.array(within_vars) * np.array(spk2num_utt)
    within_var = np.sum(within_var) / np.sum(spk2num_utt)

    # between_var = np.var(all_vectors, axis=0).sum() - within_var
    all_vectors = []
    for spk in spk2vec:
        if len(spk2vec[spk]) > 0:
            spk_con = np.concatenate(spk2vec[spk])
            all_vectors.append(spk_con)
    all_vectors = np.concatenate(all_vectors, axis=0)

    overall_mean = np.mean(all_vectors, axis=0)
    between_var = 0
    for spk in spk2vec:
        if len(spk2vec[spk]) > 0:
        # between_var += np.sum((np.mean(spk2vec[spk], axis=0)-overall_mean)**2) * len(spk2vec[spk])
            between_var += dist_fn(np.mean(spk2vec[spk], axis=0), overall_mean) ** 2 * len(spk2vec[spk])

    between_var /= np.sum(spk2num_utt)

    print("Variance Within-Class Between-Class\n", '         {:>7.4f}    {:>7.4f}'.format(within_var, between_var))
