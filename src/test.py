#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/14 11:58
# @Author  : uyplayer
# @Site    : uyplayer.pw
# @File    : test.py
# @Software: PyCharm

import torch
import torch.nn as nn
import numpy as np

# 导入包
import argparse
# 创建解析器
parser = argparse.ArgumentParser()

#添加位置参数(positional arguments)
parser = argparse.ArgumentParser()
parser.add_argument('-ne', '--neutral', type=str,default="NEUTRAL")
parser.add_argument('-sq_len', '--sequence_length', type=int, default=300, help='Max sentence length in ''train/test '
                                                                                'data (''Default: 50)')
parser.add_argument('-embed_dim', '--embedding_dim', type=int, default=300, help='word_embedding_dim')
parser.add_argument('-mf', '--model_path',type=str,default='../model_files/Sentiment140 dataset with 1.6 million '
                                                           'tweets/RCNN_tweets.pth', help='model file saving dir')
args = parser.parse_args()
print(args.sequence_length)


