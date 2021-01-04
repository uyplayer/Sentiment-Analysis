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

# dependency library
from __future__ import print_function
# system
import os
import re
import string
import time
import random
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
# bert
import transformers
from transformers import BertModel, BertConfig,BertTokenizer
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
import warnings
warnings.filterwarnings("ignore")

#Params
BATCH_SIZE = 100
SEQUENCE_LENGTH = 300
# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SEED = 60
TRAIN_SIZE = 0.8

# random seed
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
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

y = encoder.transform(target.tolist())


# tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# config
config = BertConfig.from_pretrained("bert-base-cased")
config.is_decoder = True
# model
model = BertModel.from_pretrained('bert-base-uncased',config=config)

# tokenizer
x = tokenizer(text, padding=True, truncation=True, return_tensors="pt") # pt is pytorch ;np is numpy

# masking
# masking = np.where(x != 0,1,0) # this means  1 if x!=0 else 0
input_ids = x['input_ids']
attention_mask = x['attention_mask']


# only using one bert model , because we using only bert model and at the end do not have external model , so we donot need calcilating gred
with torch.no_grad():
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    outputs = model(input_ids, attention_mask=attention_mask)
    loss = F.cross_entropy(outputs.logits, labels)
    loss.backward()
    optimizer.step()

# using bert model and adding cusiimized one at end ; it needs to calcilating gred , because we using external model at the end
def train():
    pass

