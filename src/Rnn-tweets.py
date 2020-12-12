#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/2 15:17
# @Author  : uyplayer
# @Site    : uyplayer.pw
# @File    : Rnn-tweets.py
# @Software: PyCharm

# dependency library
# system
import os
import re
import string
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
import torch.nn.functional as F
import torch.optim as optim
import torchvision
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

# PARAMS
# WORD2VECgensim
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10
# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 50
BATCH_SIZE = 1024
# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)
# MODEL PATH
MODEL_PATH = "./model_files/Sentiment140 dataset with 1.6 million tweets/Rnn_tweets.pth"

# GPU check
# device = torch.device("cpu")
is_available = torch.cuda.is_available()
device = torch.device("cuda:0" if is_available else "cpu")
print(" device is ",device)
if is_available:
    print(" GPU is avaliable")
    num_gpu = torch.cuda.device_count()
    print(" number of GPU is ",num_gpu)
    current_gpu = torch.cuda.current_device()
    print(" current_gpu is ", current_gpu)

else:
    print(" GPU is not avaliable ")

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
x_train = pad_sequences(tokenizer.texts_to_sequences(train_x), maxlen=SEQUENCE_LENGTH) # 300  word to index
x_test = pad_sequences(tokenizer.texts_to_sequences(test_x), maxlen=SEQUENCE_LENGTH) # 300

# Label Encoder ; process labels
labels = train_y.unique().tolist()

labels.append(NEUTRAL)

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
batch_size = BATCH_SIZE

# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

# dataiter = iter(train_loader)
# sample_x, sample_y = dataiter.next()
#
# print('Sample input size: ', sample_x.size()) # batch_size, seq_length
# print('Sample input: \n', sample_x)
# print('Sample input: \n', sample_y)

# my model
class Model(nn.Module):
    def __init__(self,input_size,hidden_dim,batch_size,vocab_size,output_dim):
        super(Model, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.output_dim = output_dim

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.rnn = nn.RNN(self.input_size, self.hidden_dim)
        self.fc = nn.Linear(self.input_size, output_dim)

    def forward(self, text):
        # text [sentence length, batch_size]
        embedded = self.embedding(text)
        # embedded = [sentence length, batch_size, emb dim]
        # what are the output and hidden  ? https://www.pianshen.com/article/6959294754/
        # hidden is last hiden state ; output is the rnn output
        output, hidden = self.rnn(embedded)
        # output = [sent len, batch_size, hid dim]
        # hidden = [1, batch_size, hid dim]
        assert torch.equal(output[-1, :, :], hidden.squeeze(0))
        return self.fc(output[:, -1, :])

# training
def train():

    input_size = 300
    hidden_dim = 300
    batch_size = 100
    epochs = EPOCHS
    vocab_size = len(tokenizer.word_index) + 1
    output_dim = 1

    # model  input_size,hidden_dim,batch_size,vocab_size,output_dim)
    model = Model(input_size,hidden_dim,batch_size,vocab_size,output_dim)

    # device
    model = model.to(device)

