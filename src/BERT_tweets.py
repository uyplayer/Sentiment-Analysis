#!/usr/bin/python3
# -*- coding: UTF-8 -*-
"""
@Time    : 2021-01-03 2:57
@Author  : uyplayer
@Site    : uyplayer.pw
@contact : uyplayer@qq.com
@File    : BERT_tweets.py
@Software: PyCharm
"""

# https://pytorch.org/hub/huggingface_pytorch-transformers/
# https://huggingface.co/transformers/model_doc/bert.html#transformers.BertForSequenceClassification
# http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
# https://huggingface.co/transformers/preprocessing.html
# https://stackoverflow.com/questions/45648668/convert-numpy-array-to-0-or-1
# https://huggingface.co/transformers/training.html

# dependency library
from __future__ import print_function
# system
import os
import re
import string
import time
import random
import argparse
# data
import numpy as np
import pandas as pd
# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
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
# bert
import transformers
from transformers import BertModel, BertConfig,BertTokenizer
from transformers import AdamW
from transformers import BertForSequenceClassification
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
from pre_tools.load_data_tweets import return_bert_data
# warnings
# import warnings
# warnings.filterwarnings("ignore")


# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"

# model PARAMS
parser = argparse.ArgumentParser()
parser.add_argument('-bc', '--batch_size', type=int,default=100,help='bact size of input')
parser.add_argument('-sl', '--sequence_length', type=int,default=300,help='sequence lenght of a sentense')
parser.add_argument('-sd', '--seed', type=int,default=60,help='param of random seed')
parser.add_argument('-ts', '--train_size', type=float,default=0.8,help='size of train data set')
parser.add_argument('-epos', '--epochs', type=int,default=30,help='training epochs')
parser.add_argument('-mn', '--model_name', type=str,default='bert-base-uncased',help=' model name we are going to use ')
parser.add_argument('-tosa', '--token_save_path', type=str,default='../model_files/Sentiment140 dataset with 1.6 million '
                                                           'tweets/BERT_tweets_tokens.pth',help='path of saving token files')
args = parser.parse_args()

# random seed
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

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
data = return_bert_data()
text = data.text
target = data.target


# Label Encoder ; process labels
labels = target.unique().tolist()

labels.append(NEUTRAL)

encoder = LabelEncoder()
encoder.fit(target.tolist())

labels = encoder.transform(target.tolist())



# tokenizer
tokenizer = BertTokenizer.from_pretrained(args.model_name)

# model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# pandas to list
text = text.tolist()

# split data
train_X,test_X,train_y,test_y = train_test_split(text,labels,test_size=1-args.train_size,random_state=40)

# masking
# masking = np.where(x != 0,1,0) # this means  1 if x!=0 else 0

# batch
def get_batches(X,Y, n_batches=20):
    batch_size = len(X) // n_batches
    for i in range(0, n_batches * batch_size, batch_size):
        if i != (n_batches - 1) * batch_size:
            x = X[i:i + n_batches]
            y = Y[i:i + n_batches]
        else:
            x = X[i:]
            y = Y[i:]
        yield x, y

def run_only():

    # opt

    optimizer = AdamW(model.parameters(), lr=1e-5)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

    #Epoch
    for epoch in range(args.epochs):

        # shuffle
        shuf_train_X, shuf_train_y = shuffle(train_X, train_y,random_state=0)

        # only using one bert model , because we using only bert model and at the end do not have external model , so we donot need calcilating gred
        for i, (x, y) in enumerate(get_batches(shuf_train_X, shuf_train_y, args.batch_size)):
            # shuffle
            x_shuf, y_shuf = shuffle(x, y, random_state=0)
            output_token = tokenizer(x_shuf, padding=True, truncation=True, return_tensors="pt")
            input_ids = output_token['input_ids']
            attention_mask = output_token['attention_mask']
            y_shuf = torch.tensor(y_shuf).unsqueeze(0)
            outputs = model(input_ids, attention_mask=attention_mask, labels=y_shuf)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(loss.item())

# using bert model and adding cusiimized one at end ; it needs to calcilating gred , because we using external model at the end
def train():
    pass

if __name__ == "__main__":
    run_only()