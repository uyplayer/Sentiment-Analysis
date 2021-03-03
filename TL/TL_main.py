#!/usr/bin/python3
# -*- coding: UTF-8 -*-
"""
@Time     : 2/14/21 2:28 PM
@Author   : uyplayer
@Site     : uyplayer.pw
@contact  : uyplayer@qq.com
@File     : TL_main.py
@Software : PyCharm
"""

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
# params
from TL.Params import args
# w2v
from TL.W2V import wv2_loader
# model
from TL.Models import Embeddings, MultiHeadAttention,PositionalEncoding,Model
#warnings
import warnings
warnings.filterwarnings("ignore")


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



# get word to vec
train_loader,valid_loader,vocab_size = wv2_loader()
# dataiter = iter(train_loader)
# sample_x, sample_y = dataiter.next()
# # print('sample_x input size: ', sample_x.size()) # batch_size, seq_length
# # print('sample_y input size: ', sample_y.size()) # batch_size, seq_length
# # print('Sample input: \n', sample_x)
# print('Sample input: \n', sample_y)



# Embedding to get word embedding and position embedding
Embeds = Embeddings(vocab_size)
# embeddings,word_embeddings,position_embeddings = Embeds(sample_x)
# print(embeddings.size())
# print(word_embeddings.size())
# print(position_embeddings.size())

# Model
model = Model()
# out = model(embeddings,word_embeddings)
# print(out)

# ops paramsters
criterion = nn.SmoothL1Loss()
# optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) 1e-2
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# train
def train():
    print(" Train is started ")
    model.train()  # Turn on the train mode

    # validate loss
    loss_val = np.ones((args.epochs, 1)) * np.inf

    for epoch in range(args.epochs):

        running_loss = 0.0
        running_accuracy = 0.0
        t = time.time()

        for i, data in enumerate(train_loader):

            sample_x, sample_y = data
            # LongTensor
            train_x = sample_x.type(torch.LongTensor)
            # FloatTensor
            train_y = sample_y.type(torch.FloatTensor)
            embeddings, word_embeddings, position_embeddings = Embeds(sample_x)
            output = model(embeddings, word_embeddings)
            sample_y = sample_y.view(-1,1)
            sample_y = sample_y.type(torch.FloatTensor)
            # loss
            loss = criterion(output, sample_y)
            accuracy = ((output > 0.5).type(torch.uint8) == sample_y).float().mean().item()
            # loss.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            # optimizer.step()
            # # optimizer.zero_grad()
            # # loss.backward(retain_graph=True)
            # # optimizer.step()

            # running
            running_loss =running_loss + loss.item()
            running_accuracy =running_accuracy + accuracy

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elapsed = time.time() - t

            print("Errors for epoch %d, batch %d loss: %f, accuracy : %f ,  took time: %f" % (epoch, i, loss.item(),
                                                                                              accuracy, elapsed))

            # each 20
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
                with open("./results/TL.txt", "a+") as file:
                    file.write(json.dumps(dictio) + "\n")

                running_loss = 0.0
                running_accuracy = 0.0

        # validate  valid_loader
        va_len = len(valid_loader)
        loss_test = np.ones((va_len, 1))

        for i,data in enumerate(valid_loader):

            sample_x, sample_y = data

            # LongTensor
            sample_x = sample_x.type(torch.LongTensor)

            # FloatTensor
            sample_y = sample_y.view(-1, 1)
            sample_y = sample_y.type(torch.FloatTensor)

            # output
            embeddings, word_embeddings, position_embeddings = Embeds(sample_x)
            output = model(embeddings, word_embeddings)

            # loss
            loss = criterion(output, sample_y)
            loss_test[i] = loss.detach().numpy()

        loss_val[epoch] = np.mean(loss_test)
        # save model if it reduces the loss
        if loss_val[epoch] == np.min(loss_val):
            torch.save(model.state_dict(), args.model_path)

        print("Validation errors for epoch %d: %f ,  took time: %f" % (epoch, loss_val[epoch], elapsed))

# evaluate
# def evaluate(text_list):
#
#     model = Model()
#     model.load_state_dict(torch.load(args.model_path))
#
#
#     # tokinizer
#     test = evelute_w2v(text_list)
#
#     # batch generate
#     batch_size = args.batch_size
#
#     def get_batches(X, n_batches=20):
#         batch_size = len(X) // n_batches
#         for i in range(0, n_batches * batch_size, batch_size):
#             if i != (n_batches - 1) * batch_size:
#                 x = X[i:i + n_batches]
#             else:
#                 x = X[i:]
#             yield x
#     # evaluate
#     if len(test) // batch_size > 0:
#         for i, x in enumerate(get_batches(test, batch_size)):
#             embeddings, word_embeddings, position_embeddings = Embeds(x)
#             outputs = model(embeddings, word_embeddings)
#             for i in range(len(outputs)):
#                 print(f"{text_list[i]}   :  {float(outputs[i])}")
#
#     else:
#         embeddings, word_embeddings, position_embeddings = Embeds(test)
#         outputs = model(embeddings, word_embeddings)
#         for i in range(len(outputs)):
#             print(f"{text_list[i]}   :  {float(outputs[i])}")
# Main
if __name__ == "__main__":
    train()














