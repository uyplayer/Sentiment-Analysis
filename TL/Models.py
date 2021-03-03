#!/usr/bin/python3
# -*- coding: UTF-8 -*-
"""
@Time     : 2/14/21 2:28 PM
@Author   : uyplayer
@Site     : uyplayer.pw
@contact  : uyplayer@qq.com
@File     : Models.py
@Software : PyCharm
"""

# dependency library
from __future__ import print_function
# system
import math
# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# params
from TL.Params import args
# warnings
import warnings
warnings.filterwarnings("ignore")

#Embedding : word embedding and position embedding
class Embeddings(nn.Module):

    def __init__(self,vocab_size):
        super(Embeddings, self).__init__()
        self.embedding_dim = args.embedding_dim
        self.dropout = args.dropout
        self.max_len = args.pos_max_len
        self.word_embeddings = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=self.dropout)
        pe = torch.zeros(self.max_len, self.embedding_dim)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * (-math.log(10000.0) / self.embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)
        # print(pe)
        # print(pe.shape)
        self.position_embeddings = nn.Embedding.from_pretrained(pe)

    def forward(self, input_ids):

        non_zero = torch.count_nonzero(input_ids, dim=1)
        max_len = input_ids.size(1)
        input_pos = torch.tensor(
            [list(range(1, len + 1)) + [0] * (max_len - len) for len in non_zero])

        # seq_length = input_ids.size(1)
        # position_ids = torch.arange(seq_length)  # (max_seq_length)
        #
        # position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)
        input_ids = input_ids.long()
        input_pos = input_pos.long()
        word_embeddings = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        position_embeddings = self.position_embeddings(input_pos)  # (bs, max_seq_length, dim)
        embeddings = word_embeddings + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)

        return embeddings,word_embeddings,position_embeddings


class PositionalEncoding(nn.Module):

    def __init__(self):
        super(PositionalEncoding, self).__init__()

        self.embedding_dim = args.embedding_dim
        self.dropout = args.dropout
        self.max_len = args.pos_max_len

        self.dropout = nn.Dropout(p=self.dropout)

        pe = torch.zeros(self.max_len, self.embedding_dim)
        # torch.arange returns a 1-D tensor with values from start (0) to end (max_len)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(
            1)  # unsqueeze add 1 to the axis 1 (from (4,) to (4, 1))
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * (-math.log(10000.0) / self.embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        # register buffer means the variable pe is not learnable (pe is not a model parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

#attention
def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
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

    def forward(self, q, k, v):
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
        scores = attention(q, k, v, self.d_k, self.dropout)

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

        self.size = args.d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


# CNN
class Model_CNN(nn.Module):

    def __init__(self):
        super(Model_CNN, self).__init__()

        self.input_channel_1 = args.input_channel_1
        self.output_channel_1 = args.output_channel_1
        self.input_channel_2 = args.input_channel_2
        self.output_channel_2 = args.output_channel_2

        # archetecture
        self.conv1 = nn.Conv2d(self.input_channel_1, self.output_channel_1, 3, stride=3) # kernel_size =3;stride=3
        # ;padding=0;dilation=1
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(self.input_channel_2, self.output_channel_2, 3, stride=3)
        self.fc1 = nn.Linear(100, 1)
        self.r = nn.ReLU(inplace=True)

    def forward(self, x):
        # x:torch.Size([1, 300])
        x = x.clone().detach()
        x = x.view(x.shape[0],-1,x.shape[1],x.shape[1]) # torch.Size([1, 1, 300, 300])

        x = self.conv1(x)  # (1024,2, 100,100)

        x = self.pool(F.relu(x))  # (1024,2, 50,50)

        x = self.conv2(x)  # (1024,4, 16,18)

        x = F.dropout(x, p=args.dropout)
        x = self.pool(x)  #(1024,4, 8,8)
        x = self.r(x)
        x = x.view(-1, 4 * 5 * 5) # 256

        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x

# LSTM model
class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()

        # Hyperparameters
        self.hidden_dim = args.hidden_size
        self.LSTM_layers = args.num_layers
        self.input_size = args.input_size
        self.batch_size = args.batch_size

        self.dropout = nn.Dropout(args.dropout)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_dim, num_layers=self.LSTM_layers,
                            batch_first=True)
        self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim * 2)
        self.fc2 = nn.Linear(self.hidden_dim * 2, 1)  # blstm

    def forward(self, x):
        # Hidden and cell state definion
        h = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim))
        c = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim))

        # Initialization fo hidden and cell states
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)
        # Each sequence "x" is passed through an embedding layer
        x = x.clone().detach()
        # Feed LSTMs
        out, (hidden, cell) = self.lstm(x, (h, c))
        # The last hidden state is taken
        # out = torch.relu_(self.fc1(out[:, -1, :]))
        # out = self.dropout(out)
        # out = torch.sigmoid(self.fc2(out))
        return out[:, -1, :]

# Transfoemr Encoder
class Trasformer_Encoder(nn.Module):

    def __init__(self):
        super(Trasformer_Encoder, self).__init__()
        self.MultiHeads = MultiHeadAttention(args.multiheads, args.multiheads_dim, args.dropout)
        self.FeedForwards = FeedForward(args.d_model, args.d_ff, args.dropout)
        self.Norms = Norm(args.d_model)
        # 100*200
        self.W_q = nn.Parameter(torch.randn(args.embedding_dim, args.multiheads_dim))
        self.W_k = nn.Parameter(torch.randn(args.embedding_dim, args.multiheads_dim))
        self.W_v = nn.Parameter(torch.randn(args.embedding_dim, args.multiheads_dim))

    def forward(self,x):
        # q, k, v
        # print(x.size())
        # print(self.W_q.size())
        q = torch.matmul(x, self.W_q)
        k = torch.matmul(x, self.W_k)
        v = torch.matmul(x, self.W_v)

        # MultiHeads
        MultiHeads = self.MultiHeads(q, k, v) # [64, 100, 100]
        x_atten = x+MultiHeads

        # Norm
        Norms = self.Norms(x_atten)

        # Feedforwrd
        Feeds = self.FeedForwards(x_atten)

        # Norm
        out = Norms + Feeds
        return out


# Model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hidden_dim = args.hidden_size

        self.Trasformer_Encoders = Trasformer_Encoder()
        self.LSTMs1 = LSTM()
        self.LSTMs2 = LSTM()

        self.Norms = Norm(args.d_model)

        self.dropout = nn.Dropout(args.dropout)

        self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim * 2)
        self.fc2 = nn.Linear(self.hidden_dim * 2, 1)  # blstm


    def forward(self,embeddings,word_embeddings):

        # Encoder
        Encode_outs = self.Trasformer_Encoders(embeddings)

        # LSTMs
        # wd = self.Norms(word_embeddings)
        Lstm_outs1 = self.LSTMs1(word_embeddings)

        Lstm_outs2 = self.LSTMs2(Encode_outs)
        lstm_out = Lstm_outs1+Lstm_outs2

        out = torch.relu_(self.fc1(lstm_out))
        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out))
        return out





