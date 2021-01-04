#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/14 11:03
# @Author  : uyplayer
# @Site    : uyplayer.pw
# @File    : FastText_tweets.py
# @Software: PyCharm

'''
fastText for sentiment analysis
paper : https://arxiv.org/pdf/1607.01759.pdf
https://zhuanlan.zhihu.com/p/32965521
https://keras-zh.readthedocs.io/examples/imdb_fasttext/

1. 训练词向量时，我们使用正常的word2vec方法，而真实的fastText还附加了字符级别的n-gram作为特征输入；

2. 我们的输出层使用简单的softmax分类，而真实的fastText使用的是Hierarchical Softmax。
'''

# dependency library
from __future__ import print_function
# system
import os
import re
import string
import time
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
# warnings
import warnings
warnings.filterwarnings("ignore")

# params
NGRAN_RANGE = 2
MAX_LEN = 300 # n_grams max lenght
MAX_FEATURES = 20000
EMBEDDING_DIM = 300
OUTPUT_DIM = 1
BATCH_SIZE = 64
SEQUENCE_LENGTH = 500
EPOCHS = 5
# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"


# MODEL PATH
MODEL_PATH = "../model_files/Sentiment140 dataset with 1.6 million tweets/FastText_tweets.pth"

# GPU check
# device = torch.device("cpu")
is_available = torch.cuda.is_available()
device = torch.device("cuda:0" if is_available else "cpu")
print(" device is ",device)
if is_available:
    print(" GPU is avaliable")
    num_gpu = torch.cuda.device_count()
    print(" number of GPU is : ",num_gpu)
    current_gpu = torch.cuda.current_device()
    print(" current_gpu is : ", current_gpu)

else:
    print(" GPU is not avaliable ")

# get data
train_x, train_y, test_x, test_y = return_data(rate=0.03)

# keras tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_x)
# x_train = pad_sequences(tokenizer.texts_to_sequences(train_x), maxlen=SEQUENCE_LENGTH) # 300  word to index
# x_test = pad_sequences(tokenizer.texts_to_sequences(test_x), maxlen=SEQUENCE_LENGTH) # 300
x_train = tokenizer.texts_to_sequences(train_x)
x_test = tokenizer.texts_to_sequences(test_x)

# n-gram processing
def create_ngram_set(input_list, ngram_value=2):
    """
    从整数列表中提取一组 n 元语法。

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    通过附加 n-gram 值来增强列表（序列）的输入列表。

    示例：添加 bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

    示例：添加 tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

# processing ngram
token_indice = 0
indice_token = 0
ngram_set = set()

assert NGRAN_RANGE > 1
if NGRAN_RANGE > 1:
    print('Adding {}-gram features'.format(NGRAN_RANGE))
    # 从训练集中创建一组唯一的 n-gram。
    for input_list in x_train:
        for i in range(2, NGRAN_RANGE + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # 将 n-gram token 映射到唯一整数的字典。
    # 整数值大于 max_features，
    # 以避免与现有功能冲突。
    start_index = MAX_FEATURES + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # MAX_FEATURES 是可以在数据集中找到的最大整数。
    MAX_FEATURES = np.max(list(indice_token.keys())) + 1

    # 使用 n-grams 功能增强 x_train 和 x_test
    x_train = add_ngram(x_train, token_indice, NGRAN_RANGE)
    x_test = add_ngram(x_test, token_indice, NGRAN_RANGE)
    print('Average train sequence length: {}'.format(
        np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(
        np.mean(list(map(len, x_test)), dtype=int)))



x_train = pad_sequences(x_train, maxlen=SEQUENCE_LENGTH) # 500  word to index
x_test = pad_sequences(x_test, maxlen=SEQUENCE_LENGTH) # 500
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

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
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=2)



# vocab_size
vocab_size = len(tokenizer.word_index) + 1

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

# batch of dataloader
batch_size = BATCH_SIZE

# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=2)

dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x)
print('Sample input: \n', sample_y)



# my model
class Model(nn.Module):

    def __init__(self, input_size, batch_size, hidden_dim, output_dim):
        super(Model, self).__init__()
        self.vocab_size = input_size
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # embedding
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        # 我们添加了 GlobalAveragePooling1D，它将对文档中所有单词执行平均嵌入
        # self.GlobalAveragePooling1D = torch.mean(x,dim=1)
        self.fc1 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, text):
        text = text.clone().detach()
        text = text.to(device).long()
        embedding = self.embedding(text)
        GlobalAveragePooling1D = torch.mean(embedding,dim=1)
        out = self.fc1(GlobalAveragePooling1D)
        return out


# training
def train():
    # input_size, embedding_dims, batch_size, vocab_size, output_dim
    hidden_dim = SEQUENCE_LENGTH
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    input_size = MAX_FEATURES
    output_dim = OUTPUT_DIM

    # model  input_size, batch_size, hidden_dim, output_dim
    model = Model(input_size, batch_size, hidden_dim, output_dim)

    # device
    model = model.to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    # validate loss
    loss_val = np.ones((epochs, 1)) * np.inf

    # train
    for epoch in range(epochs):

        running_loss = 0.0
        running_accuracy = 0.0
        t = time.time()

        for i, data in enumerate(train_loader):

            sample_x, sample_y = data

            # LongTensor
            train_x = sample_x.type(torch.LongTensor)
            # FloatTensor
            train_y = sample_y.type(torch.FloatTensor)

            # device
            train_x = train_x.to(device).long()
            train_y = train_y.to(device)

            # output
            output = model(train_x)

            # change demintion
            train_y = train_y.view(-1, 1)

            # loss
            loss = criterion(output, train_y)
            accuracy = ((output > 0.5).type(torch.uint8) == train_y).float().mean().item()

            # running
            running_loss += loss.item()
            running_accuracy += accuracy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elapsed = time.time() - t

            print("Errors for epoch %d, batch %d loss: %f, accuracy : %f ,  took time: %f" % (epoch, i, loss.item(),
                                                                                              accuracy, elapsed))

            # each 20
            if i == 20:
                print("[%d, %5d] epoch_loss : %.3f  epoch_accuracy : %.3f" % ((epoch + 1, i + 1, running_loss / 20,
                                                                               running_accuracy / 20)))
                running_loss = 0.0
                running_accuracy = 0.0

        # validate  valid_loader
        va_len = len(valid_loader)
        loss_test = np.ones((va_len, 1))

        for i,data in enumerate(valid_loader):
            sample_x, sample_y = data

            # LongTensor
            test_x = sample_x.type(torch.LongTensor)
            # FloatTensor
            test_y = sample_y.type(torch.FloatTensor)

            # device
            test_x = test_x.to(device).long()
            test_y = test_y.to(device)

            # output
            output = model(test_x)

            # change demintion
            test_y = test_y.view(-1, 1)

            # loss
            loss = criterion(output, test_y)
            loss_test[i] = loss.detach().cpu().numpy()

        loss_val[epoch] = np.mean(loss_test)
        # save model if it reduces the loss
        if loss_val[epoch] == np.min(loss_val):
            torch.save(model.state_dict(), MODEL_PATH)

        print("Validation errors for epoch %d: %f ,  took time: %f" % (epoch, loss_val[epoch], elapsed))

# evaluate
def evaluate(text_list,ngram=1):

    assert ngram>1

    # model params
    hidden_dim = SEQUENCE_LENGTH
    batch_size = BATCH_SIZE
    input_size = MAX_FEATURES
    output_dim = OUTPUT_DIM

    # load model  input_size, batch_size, hidden_dim, output_dim
    model = Model(input_size,batch_size,hidden_dim,output_dim)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)

    # tokinizer
    test = tokenizer.texts_to_sequences(text_list)
    # ngram
    test = add_ngram(test, token_indice, ngram)
    test = pad_sequences(test, maxlen=SEQUENCE_LENGTH)  # 500
    test = torch.from_numpy(test)
    # batch generate
    def get_batches(X, n_batches=20):
        batch_size = len(X) // n_batches
        for i in range(0, n_batches * batch_size, batch_size):
            if i != (n_batches - 1) * batch_size:
                x = X[i:i + n_batches]
            else:
                x = X[i:]
            yield x
    # evaluate
    if len(test)//batch_size > 0:
        for i, x in enumerate(get_batches(test, batch_size)):
            outputs = model(x)
            for i in range(len(outputs)):
                print(f"{text_list[i]}   :  {float(outputs[i])}")

    else:
        outputs = model(test)
        for i in range(len(outputs)):
            print(f"{text_list[i]}   :  {float(outputs[i])}")

# Main
if __name__ == "__main__":
    train()
    evaluate(["I love you","I want to hit someone"], ngram=2)











