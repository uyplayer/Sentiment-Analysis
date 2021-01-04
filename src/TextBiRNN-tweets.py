#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/16 16:24
# @Author  : uyplayer
# @Site    : uyplayer.pw
# @File    : TextBiRNN-tweets.py
# @Software: PyCharm

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

# PARAMS
# WORD2VEC
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10
# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 100
BATCH_SIZE = 1024
# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)
# MODEL PATH
MODEL_PATH = "../model_files/Sentiment140 dataset with 1.6 million tweets/TextBiRNN-tweets.pth"

# GPU check
# device = torch.device("cpu")
is_available = torch.cuda.is_available()
device = torch.device("cuda:0" if is_available else "cpu")
print(" device is : ",device)
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
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=2)

# dataiter = iter(train_loader)
# sample_x, sample_y = dataiter.next()
#
# print('Sample input size: ', sample_x.size()) # batch_size, seq_length
# print('Sample input: \n', sample_x)
# print('Sample input: \n', sample_y)


# my model
class Model(nn.Module):
    def __init__(self,vocab_size,hidden_dim,input_size,hidden_size,num_layers=2,bidirectional=True):
        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional


        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                            batch_first=True)
        self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=300)
        self.fc2 = nn.Linear(self.hidden_dim, 1)
        self.sig = nn.Sigmoid()

    def forward(self,text):
        text = text.clone().detach()
        text = text.to(device).long()
        embedding = self.embedding(text)
        out,(hidden, cell) = self.lstm(embedding)
        li = self.fc1(out[:, -1, :]) # get last line
        l2 = self.fc2(li)
        sig = self.sig(l2)
        return sig

# training
def train():

    vocab_size = len(tokenizer.word_index) + 1
    hidden_dim = 300
    input_size = 300
    hidden_size = 300
    num_layers = 2
    bidirectional = True
    dropout = 0.3
    epochs = EPOCHS
    # epochs = 1

    # model  vocab_size,hidden_dim,input_size,hidden_size,num_layers=2,bidirectional=True
    model = Model(vocab_size,hidden_dim,input_size,hidden_size,num_layers,bidirectional)

    # device
    model = model.to(device)

    criterion = nn.MSELoss()
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
                                                                               running_accuracy/20)))
                running_loss = 0.0
                running_accuracy =0.0


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
def evaluate(text_list):
    vocab_size = len(tokenizer.word_index) + 1
    hidden_dim = 300
    input_size = 300
    hidden_size = 300
    num_layers = 2
    bidirectional = True
    dropout = 0.3
    epochs = EPOCHS
    # epochs = 1

    # model  vocab_size,hidden_dim,input_size,hidden_size,num_layers=2,bidirectional=True
    model = Model(vocab_size, hidden_dim, input_size, hidden_size, num_layers, bidirectional)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)

    # tokinizer
    test = pad_sequences(tokenizer.texts_to_sequences(text_list), maxlen=SEQUENCE_LENGTH)
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
    if len(test) // batch_size > 0:
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
    evaluate(["I love you","I want to hit someone"])