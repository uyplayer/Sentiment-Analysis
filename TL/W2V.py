#!/usr/bin/python3
# -*- coding: UTF-8 -*-
"""
@Time     : 2/14/21 2:28 PM
@Author   : uyplayer
@Site     : uyplayer.pw
@contact  : uyplayer@qq.com
@File     : W2V.py
@Software : PyCharm
"""



# dependency library
from __future__ import print_function
import argparse
# sklearn
from sklearn.preprocessing import LabelEncoder
# Pytorch
import torch
from torch.utils.data import TensorDataset, DataLoader
# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# pre_tools
from pre_tools.load_data_tweets import return_data
# params
from TL.Params import args
# warnings
import warnings
warnings.filterwarnings("ignore")


def wv2_loader():
    # get data
    train_x, train_y, test_x, test_y = return_data(rate=0.03)
    '''
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)
    (38400,)
    (38400,)
    (9600,)
    (9600,)
    '''

    # convert to numpy
    # train_x = np.array(train_x)
    # train_y = np.array(train_y)
    # test_x = np.array(test_x)
    # test_y = np.array(test_y)

    # keras tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_x)
    vocab_size = len(tokenizer.word_index) + 1
    x_train = pad_sequences(tokenizer.texts_to_sequences(train_x), maxlen=args.sequence_length)  # 300  word to index
    x_test = pad_sequences(tokenizer.texts_to_sequences(test_x), maxlen=args.sequence_length)  # 300

    # Label Encoder ; process labels
    labels = train_y.unique().tolist()

    labels.append(args.neutral)

    encoder = LabelEncoder()
    encoder.fit(train_y.tolist())

    y_train = encoder.transform(train_y.tolist())
    y_test = encoder.transform(test_y.tolist())

    # print(y_train) # [1 1 1 ... 0 0 0]

    # y_train = y_train.reshape(-1,1)
    # y_test = y_test.reshape(-1,1)
    # print(y_train) # [[1][1][1]...[0][0][0]]
    # print(y_test) # [[0][0][0]...[1][1][0]]

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    valid_data = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    # dataloaders
    batch_size = args.batch_size

    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

    # dataiter = iter(train_loader)
    # sample_x, sample_y = dataiter.next()
    #
    # print('Sample input size: ', sample_x.size()) # batch_size, seq_length
    # print('Sample input: \n', sample_x)
    # print('Sample input: \n', sample_y)

    return train_loader,valid_loader,vocab_size


def evelute_w2v(text):
    # text : list
    # get data
    train_x, train_y, test_x, test_y = return_data(rate=0.03)

    # keras tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_x)
    vocab_size = len(tokenizer.word_index) + 1
    test = pad_sequences(tokenizer.texts_to_sequences(text), maxlen=args.sequence_length)  # 300  word to index
    test = torch.from_numpy(test)

    return test
