#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: f_ratio.py
@Time: 2020/9/22 15:39
@Overview:
"""
import numpy as np


def fratio(k, all_frames):
    """

    :param k: subband k
    :param all_frames: [spk, k, N] N is the number of frames

    :return:
    """
    try:
        num_spk, num_k, num_fram = all_frames.shape
    except Exception as e:
        print(all_frames.shape)
        raise e
    assert num_k == k

    u_i = np.mean(all_frames, axis=2)  # [spk, k]
    u = u_i.mean(axis=0)  # [k]
    numerator = np.sum(np.power(u_i - u, 2), axis=0)

    in_spk_var = np.power(all_frames - u_i.reshape(num_spk, num_k, 1).repeat(num_fram, 2), 2)  # [spk, k, N]
    denominator = np.sum(np.mean(in_spk_var, axis=2), axis=0)

    return numerator / denominator
