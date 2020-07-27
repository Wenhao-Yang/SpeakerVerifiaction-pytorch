#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: lms_filter.py
@Time: 2020/7/27 12:52
@Overview:
"""

import numpy as np
import scipy


# 定义向量的内积
def multiVector(A, B):
    C = scipy.zeros(len(A))

    for i in range(len(A)):
        C[i] = A[i] * B[i]

    return sum(C)


# 取定给定的反向的个数
# 返回A中从a-->b的值
def inVector(A, b, a):
    D = scipy.zeros(b - a + 1)

    for i in range(b - a + 1):
        D[i] = A[i + a]

    return D[::-1]


def LMS(xn, dn, M, mu, itr):
    """

    :param xn:
    :param dn:
    :param M: Weight 的长度
    :param mu:
    :param itr:
    :return:
    """
    en = scipy.zeros(itr)
    W = [[0] * M for i in range(itr)]

    for k in range(itr)[M - 1:itr]:
        x = inVector(xn, k, k - M + 1)
        d = x.mean()
        y = multiVector(W[k - 1], x)
        en[k] = d - y
        W[k] = np.add(W[k - 1], 2 * mu * en[k] * x)  # 跟新权重

    # 求最优时滤波器的输出序列

    yn = scipy.inf * scipy.ones(len(xn))

    for k in range(len(xn))[M - 1:len(xn)]:
        x = inVector(xn, k, k - M + 1)

        yn[k] = multiVector(W[len(W) - 1], x)

    return (yn, en)
