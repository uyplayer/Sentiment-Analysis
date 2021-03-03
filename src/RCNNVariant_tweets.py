#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/23 9:53
# @Author  : uyplayer
# @Site    : uyplayer.pw
# @File    : RCNNVariant_tweets.py
# @Software: PyCharm

# dependency library
from __future__ import print_function
# system
import os
import re
import string
import time
import argparse
import json
# data
import numpy as np
import pandas as pd
# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
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
#warnings
# import warnings
# warnings.filterwarnings("ignore")

# model PARAMS
parser = argparse.ArgumentParser()
parser.add_argument('-ne', '--neutral', type=str,default="NEUTRAL")
parser.add_argument('-sq_len', '--sequence_length', type=int, default=100, help='Max sentence length in ''train/test '
                                                                                'data (''Default: 50)')
parser.add_argument('-embed_dim', '--embedding_dim', type=int, default=200, help='word_embedding_dim')
parser.add_argument('-mf', '--model_path',type=str,default='./model_files/Sentiment140 dataset with 1.6 million tweets/RCNNVariant_tweets.pth', help='model file saving dir')
parser.add_argument('-ct', '--cell_type',type=str,default='LSTM', help='cell type RNN LSTM GRU')
parser.add_argument("--hidden_size", type=int, default=200, help="Size of hidden layer (Default: 512)")
parser.add_argument("--input_size", type=int,default=200, help="input_size of cell")
parser.add_argument("--dropout_keep_prob", type=float, default=0.7, help="Dropout keep probability (Default: 0.7)")
parser.add_argument("--l2_reg_lambda", type=float, default=0.5, help="L2 regularization lambda (Default: 0.5)")
parser.add_argument('-num_classes', '--num_classes', type=int, default=2, help='number of classes')
parser.add_argument('-num_layers', '--num_layers', type=int, default=2, help='number of classes')
parser.add_argument('-bidirectional', '--bidirectional', type=bool, default=True, help='cell bidirectional')
parser.add_argument('-dropout', '--dropout', type=float, default=0.2, help='dropout')
parser.add_argument('-batch_first', '--batch_first', type=bool, default=True, help='batch_first')
parser.add_argument('-bias', '--bias', type=bool, default=True, help='bias')
# Training parameters
parser.add_argument("--batch_size", type=int, default=64, help="Batch Size (Default: 64)")
parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs (Default: 10)")
parser.add_argument("--display_every", type=int, default=10, help="Number of iterations to display training info.")
parser.add_argument("--evaluate_every", type=int, default=100, help="Evaluate model on dev set after this many steps")
parser.add_argument("--checkpoint_every", type=int, default=100, help="Save model after this many steps")
parser.add_argument("--num_checkpoints", type=int, default=5, help="Number of checkpoints to store")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Which learning rate to start with. (Default: 1e-3)")
args = parser.parse_args()

# GPU check
print(" checking  cuda  ")
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

# Get data
print(" loadding data ")
train_x, train_y, test_x, test_y = return_data(rate=0.03)

# Label Encoder ; process labels
print(" processing labels ")
labels = train_y.unique().tolist()
# NEUTRAL
labels.append(args.neutral)
encoder = LabelEncoder()
encoder.fit(train_y.tolist())
y_train = encoder.transform(train_y.tolist())
y_test = encoder.transform(test_y.tolist())
# convert to [] mode
# y_train = [[0, 1] if item == 1 else [1, 0] for item in y_train]
# y_test = [[0, 1] if item == 1 else [1, 0] for item in y_test]
# convert to numpy
y_train = np.array(y_train)
y_test = np.array(y_test)

# keras tokenizer train and test data
print(" processing train and test ")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_x)
vocab_size = len(tokenizer.word_index) + 1
x_train = pad_sequences(tokenizer.texts_to_sequences(train_x), maxlen=args.sequence_length) # 100
x_test = pad_sequences(tokenizer.texts_to_sequences(test_x), maxlen=args.sequence_length) # 100

# create Tensor datasets for train and test
train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
valid_data = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

# dataloaders
batch_size = args.batch_size

# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

# vocab size
vocab_size = len(tokenizer.word_index) + 1

# save_models
class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, cell_type, input_size, hidden_size, num_layers, bidirectional,
                 dropout, batch_first, bias):
        super(Model, self).__init__()

        # params
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.cell_type = cell_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.batch_first = batch_first
        self.bias = bias

        # embedding
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)

        # cell type : LSTM GRU
        assert self.cell_type in ["LSTM", "GRU"], "cell type is wrong"
        if self.cell_type == "LSTM":
            self.cell = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                               bias=bias, batch_first=batch_first, dropout=self.dropout,
                               bidirectional=self.bidirectional)
        elif self.cell_type == "GRU":
            self.cell = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                               bias=bias, batch_first=batch_first, dropout=self.dropout,
                               bidirectional=self.bidirectional)

        # convolution input_channel =1 ,output_channel =1; kernel_size =3;stride=3
        self.conv2d_layer = nn.Conv2d(1, 2, 3, stride=3)

        # maxpooling
        self.pool = nn.MaxPool2d(2, 2)

        # AvgPool1d
        self.avpool = nn.AvgPool2d(2, 2)

        # fullconnect
        self.full = nn.Linear(6400, 1)

        # softmax
        self.sf = nn.Sigmoid()

    def forward(self, text):

        # embedding
        text = text.clone().detach()
        text = text.to(device).long()

        # torch.Size([64, 100, 200])
        embeding = self.embedding(text)

        # cell_type Bidirectional
        if self.cell_type == "LSTM":
            output, (hn, cn) = self.cell(embeding)
        elif self.cell_type == "GRU":
            output, hn = self.cell(embeding)
        # torch.Size([64, 100, 400])
        # torch.Size([4, 64, 200])
        # print(output.shape)
        # print(hn.shape)

        # concat
        concat = torch.cat([output, embeding], 2)
        # torch.Size([64, 100, 600])
        # print(concat.shape)
        # reshape
        concat = concat.view(concat.shape[0], -1, concat.shape[1], concat.shape[2])

        # convolution
        conv2 = self.conv2d_layer(concat)
        conv5 = self.conv2d_layer(concat)
        conv = conv2*conv5
        # torch.Size([64, 2, 33, 200])
        # print(conv.shape)

        # maxpooling
        pool = self.pool(conv)
        # torch.Size([64, 2, 16, 100])
        # print(pool.shape)

        # averagepooling
        avpool = self.avpool(conv)
        # torch.Size([64, 2, 16, 100])
        # print(avpool.shape)

        # concat
        cat = torch.cat([pool, avpool], 2)
        # torch.Size([64, 2, 32, 100])
        # print(cat.shape)

        # fullconnect
        cat = cat.view(-1, 2*32*100)
        full = self.full(cat)
        # torch.Size([64, 2])
        # print(full.shape)

        # sigmoid/sofrtmax
        y = self.sf(full)

        return y

# test model
# dataiter = iter(train_loader)
# sample_x, sample_y = dataiter.next()
# print(" sample_x.shape : ", sample_x.shape)
# print(" sample_y.shape : ", sample_y.shape)
#
# sequence_length = args.sequence_length
# num_classes = args.num_classes
# vocab_size = vocab_size
# embedding_dim = args.embedding_dim
# cell_type = args.cell_type
# input_size = args.input_size
# hidden_size = args.hidden_size
# num_layers = args.num_layers
# bidirectional = args.bidirectional
# dropout = args.dropout
# batch_first = args.batch_first
# bias = args.bias
#
# # vocab_size, embedding_dim, cell_type, input_size, hidden_size, num_layers, bidirectional,dropout, batch_first, bias
# model = save_models(vocab_size=vocab_size, embedding_dim=embedding_dim, cell_type=cell_type, input_size=input_size,
#               hidden_size=hidden_size, num_layers=num_layers,
#               bidirectional=bidirectional, dropout=dropout,
#               batch_first=batch_first, bias=bias)
#
# model.to(device)
#
# print(model(sample_x))

# training
def train():
    sequence_length = args.sequence_length
    num_classes = args.num_classes
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = args.embedding_dim
    cell_type = args.cell_type
    input_size = args.input_size
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    bidirectional = args.bidirectional
    dropout = args.dropout
    batch_first = args.batch_first
    bias = args.bias
    epochs = args.num_epochs

    # vocab_size, embedding_dim, cell_type, input_size, hidden_size, num_layers, bidirectional,dropout, batch_first, bias
    model = Model(vocab_size=vocab_size, embedding_dim=embedding_dim, cell_type=cell_type, input_size=input_size,
                  hidden_size=hidden_size, num_layers=num_layers,
                  bidirectional=bidirectional, dropout=dropout,
                  batch_first=batch_first, bias=bias)

    # device
    model = model.to(device)

    criterion = nn.SmoothL1Loss()
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
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
            train_y = train_y.view(-1, 1)

            # output
            output = model(train_x)

            # loss
            loss = criterion(output, train_y)

            # _, out_index = torch.max(output, 1)
            # _, train_y_index = torch.max(train_y, 1)

            # out_index = out_index.detach().cpu().numpy()
            # train_y_index = train_y_index.detach().cpu().numpy()

            # accuracy
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

            # display_every
            if i == 20:
                print("[%d, %5d] epoch_loss : %.3f  epoch_accuracy : %.3f" % ((epoch + 1, i + 1, running_loss / 21,
                                                                               running_accuracy / 21)))
                epoch_accuracy = accuracy_score(train_y.cpu(), ((output > 0.5).type(torch.uint8).cpu()))
                precision = precision_score(train_y.cpu(), ((output > 0.5).type(torch.uint8).cpu()))
                recall = recall_score(train_y.cpu(), ((output > 0.5).type(torch.uint8)).cpu(), average='micro')
                f1 = f1_score(train_y.cpu(), ((output > 0.5).type(torch.uint8)).cpu())
                dictio = {"epoch": epoch + 1, "epoch_loss": running_loss / 21,
                          "epoch_accuracy": running_accuracy / 21, "accuracy": epoch_accuracy, "precision": precision,
                          "recall": recall, "f1": f1}
                print(dictio)
                with open("./results/RCNNVariant.txt", "a+") as file:
                    file.write(json.dumps(dictio) + "\n")

                running_loss = 0.0
                running_accuracy = 0.0


        # validate  valid_loader
        va_len = len(valid_loader)
        loss_test = np.ones((va_len, 1))

        for i, data in enumerate(valid_loader):
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

            # loss
            loss = criterion(output, test_y)
            loss_test[i] = loss.detach().cpu().numpy()

        loss_val[epoch] = np.mean(loss_test)
        # save model if it reduces the loss
        if loss_val[epoch] == np.min(loss_val):
            torch.save(model.state_dict(), args.model_path)

        print("Validation errors for epoch %d: %f ,  took time: %f" % (epoch, loss_val[epoch], elapsed))


# evaluate
def evaluate(text_list):
    sequence_length = args.sequence_length
    num_classes = args.num_classes
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = args.embedding_dim
    cell_type = args.cell_type
    input_size = args.input_size
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    bidirectional = args.bidirectional
    dropout = args.dropout
    batch_first = args.batch_first
    bias = args.bias

    # vocab_size, embedding_dim, cell_type, input_size, hidden_size, num_layers, bidirectional,dropout, batch_first, bias
    model = Model(vocab_size=vocab_size, embedding_dim=embedding_dim, cell_type=cell_type, input_size=input_size,
                  hidden_size=hidden_size, num_layers=num_layers,
                  bidirectional=bidirectional, dropout=dropout,
                  batch_first=batch_first, bias=bias)

    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    # tokinizer
    test = pad_sequences(tokenizer.texts_to_sequences(text_list), maxlen=args.sequence_length)
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
            _, outputs = torch.max(outputs,1)
            outputs = outputs.detach().cpu().numpy()
            for i in range(len(outputs)):
                print(f"{text_list[i]}   :  {float(outputs[i])}")

    else:
        outputs = model(test)
        _, outputs = torch.max(outputs,1)
        outputs = outputs.detach().cpu().numpy()
        for i in range(len(outputs)):
            print(f"{text_list[i]}   :  {float(outputs[i])}")


# Main
if __name__ == "__main__":
    train()
    evaluate(["I love you", "I want to hit someone", "fuck you bitch"])