#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/23 20:10
# @Author  : uyplayer
# @Site    : uyplayer.pw
# @File    : Transformer-tweets.py
# @Software: PyCharm

'''
http://jalammar.github.io/illustrated-transformer/
https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04
https://blog.csdn.net/malefactor/article/details/86500387
https://baijiahao.baidu.com/s?id=1651219987457222196&wfr=spider&for=pc
https://zhuanlan.zhihu.com/p/44121378
https://keras.io/examples/nlp/text_classification_with_transformer/
https://medium.com/towards-artificial-intelligence/text-classification-with-simple-transformers-a29d13358135
https://blog.floydhub.com/the-transformer-in-pytorch/
'''

# dependency library
from __future__ import print_function
# system
import time
import numpy as np
# sklearn
from sklearn.preprocessing import LabelEncoder
# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# pre_tools
from pre_tools.load_data_tweets import return_data
# warnings
# import warnings
# warnings.filterwarnings("ignore")


# dependency library
# system
import os
import re
import string
import time
import math
# data
import numpy as np
import pandas as pd
# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
# Pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchsummary import summary
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
# torxhtext
import torchtext
from torchtext import data
from torchtext.vocab import Vectors
# torch model summary
from torchsummary import summary
from string import punctuation
# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# pre_tools
from pre_tools.load_data_tweets import return_data
# warnings
# import warnings
# warnings.filterwarnings("ignore")

#Embedding
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self,x):
        return self.embed(x)

#PositionalEncoder
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model

        # create positional matrix
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0,d_model,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        pe = pe.unsqueeze(0) # 增加维度
        '''
        向模块添加持久缓冲区。
        这通常用于注册不应被视为模型参数的缓冲区。例如，BatchNorm的running_mean不是一个参数，而是持久状态的一部分。
        缓冲区可以使用给定的名称作为属性访问。
        说明：
        应该就是在内存中定义一个常量，同时，模型保存和加载的时候可以写入和读出。
        '''
        self.register_buffer('pe', pe)

    def forward(self,x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1) # 获取x的行数
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).cuda()# self.pe[:, :seq_len] 获取前面的 seq_len个行
        return x

#attention
def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
    scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)


    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output

#MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output



class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

#Normalization
class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(Norm, self).__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm












    



































