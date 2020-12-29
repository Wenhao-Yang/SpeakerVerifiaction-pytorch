# !/usr/bin/env python -u
# -*- coding: utf-8 -*-

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: probar.py
@Time: 2019/10/29 下午4:49
@Overview:
"""

from tqdm import tqdm
import time

print('\33[1;34m Current learning rate is {}.\33[0m \n'.format(0.1))
def train():
    pbar = tqdm(enumerate(range(6)))

    for i, j in pbar:
        pbar.set_description('train Step %d'% i)
        #do something
        # time.sleep(1)

    print('\nPython has had awesome string \n')

def test():
    pbar = tqdm(enumerate(range(3)))

    for i, j in pbar:
        pbar.set_description('test Step {}'.format(i))
        # do something
        # time.sleep(1)

    print('Python tqdm test!\n')


if __name__ == '__main__':

    for i in range(0, 2):
        train()
        test()
