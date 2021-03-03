#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/14 14:43
# @Author  : uyplayer
# @Site    : uyplayer.pw
# @File    : Lstm_tweets.py.py
# @Software: PyCharm

'''
this script is LSTM model for tweets
https://www.kaggle.com/paoloripamonti/twitter-sentiment-analysis
https://realpython.com/python-keras-text-classification/
'''

# dependency library
# data
import time
import numpy as np
import pandas as pd
import json
# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM,Input
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
# Word2vec
import gensim
# pre_tools
from pre_tools.load_data_tweets import return_data
# warnings
import warnings
warnings.filterwarnings("ignore")

# RuntimeError: cuDNN error: CUDNN_STATUS_MAPPING_ERROR
# torch.backends.cudnn.enabled = False

# PARAMS
# WORD2VECgensim
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10
# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 30
BATCH_SIZE = 1024
# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)
# MODEL PATH
MODEL_PATH = "./model_files/Sentiment140 dataset with 1.6 million tweets/Lstm_tweets.pth"

# GPU check
# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(" device is ",device)
if torch.cuda.is_available():
    print(" GPU is avaliable")
    num_gpu = torch.cuda.device_count()
    print(" number of GPU is ",num_gpu)
    current_gpu = torch.cuda.current_device()
    print(" current_gpu is ", current_gpu)

else:
    print(" GPU is not avaliable")




# get data
train_x, train_y, test_x, test_y = return_data(rate=0.03)

# documents
def documents(data):
    # data : list
    documents = [_text.split() for _text in data]
    return documents
# Word2Vec
def w2v_model():
    # Initialize https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial
    # W2V_SIZE = 300
    w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE,
                                                window=W2V_WINDOW,
                                                min_count=W2V_MIN_COUNT,
                                                workers=8)
    return w2v_model

# setup data to Word2Vec
documents = documents(train_x)
w2v_model = w2v_model()
w2v_model.build_vocab(documents)

# words
words = w2v_model.wv.vocab.keys()
vocab_size = len(words)
# print(words)
# print(vocab_size)

# w2v_model to train
w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)
# print(w2v_model.most_similar("love"))

# Tokenize Text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_x)

vocab_size = len(tokenizer.word_index) + 1
# print("Total words", vocab_size)
x_train = pad_sequences(tokenizer.texts_to_sequences(train_x), maxlen=SEQUENCE_LENGTH) # 300  word to index
x_test = pad_sequences(tokenizer.texts_to_sequences(test_x), maxlen=SEQUENCE_LENGTH) # 300

# Label Encoder
labels = train_y.unique().tolist()

labels.append(NEUTRAL)

encoder = LabelEncoder()
encoder.fit(train_y.tolist())

y_train = encoder.transform(train_y.tolist())
y_test = encoder.transform(test_y.tolist())

# print(y_train) # [1 1 1 ... 0 0 0]

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
# print(y_train) # [[1][1][1]...[0][0][0]]
# print(y_test) # [[0][0][0]...[1][1][0]]

print("x_train",x_train.shape) # (1280000, 300)
print("x_test",x_test.shape) # (320000, 300)
print("y_train",y_train.shape) # (1280000, 1)
print("y_test",y_test.shape) # (320000, 1)

# convet to tensor
x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

# Embedding layer
embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
# print(embedding_matrix.shape) # (290419, 300)



# LSTM model
class Model(nn.Module):

    def __init__(self, input_size,hidden_dim,LSTM_layers,batch_size,vocab_size):
        super(Model, self).__init__()

        # Hyperparameters
        self.hidden_dim = hidden_dim
        self.LSTM_layers = LSTM_layers
        self.input_size = input_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size

        self.dropout = nn.Dropout(0.5)
        self.embedding = nn.Embedding(self.vocab_size , self.hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_dim, num_layers=self.LSTM_layers,
                            batch_first=True)
        self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim * 2)
        self.fc2 = nn.Linear(self.hidden_dim * 2, 1) # blstm

    def forward(self, x):
        # Hidden and cell state definion
        h = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim)).to(device)
        c = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim)).to(device)

        # Initialization fo hidden and cell states
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)
        # Each sequence "x" is passed through an embedding layer
        x = x.clone().detach()
        x = x.to(device).long()
        out = self.embedding(x)
        # Feed LSTMs
        out, (hidden, cell) = self.lstm(out, (h, c))
        # The last hidden state is taken
        out = torch.relu_(self.fc1(out[:, -1, :]))
        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out))
        return out

# batch generate
def get_batches(X,Y,n_batches = 20):
    batch_size = len(X) // n_batches
    for i in range(0,n_batches*batch_size,batch_size):
        if i != (n_batches-1)*batch_size:
            x,y = X[i:i+n_batches],Y[i:i+n_batches]
        else:
            x,y = X[i:],Y[i:]
        yield x,y



# training
def train():

    input_size = 300
    hidden_dim = 300
    LSTM_layers = 2
    batch_size = 100
    epochs = EPOCHS
    # epochs = 1
    vocab_size = len(tokenizer.word_index) + 1

    model = Model(input_size,hidden_dim,LSTM_layers,batch_size,vocab_size)

    # device
    model = model.to(device)

    criterion = nn.SmoothL1Loss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # validate loss
    loss_val = np.ones((epochs,1)) * np.inf

    # train
    for epoch in range(epochs):

        running_loss = 0.0
        running_accuracy = 0.0
        t = time.time()

        for i, (train_x, train_y) in enumerate(get_batches(x_train, y_train, batch_size)):

            # LongTensor
            train_x = train_x.type(torch.LongTensor)
            # FloatTensor
            train_y = train_y.type(torch.FloatTensor)

            # device
            train_x = train_x.to(device).long()
            train_y = train_y.to(device)
            output = model(train_x)

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
                with open("./results/Lstm.txt", "a+") as file:
                    file.write(json.dumps(dictio) + "\n")

                running_loss = 0.0
                running_accuracy = 0.0


        # validate x_test,y_test
        va_len = len(y_test)
        loss_test = np.ones((va_len, 1))

        # iter each sample in test , we do not use the get_batches fuction
        for n_val in range(va_len):
            x_t = x_test[n_val].view(-1,len(x_test[n_val]))
            x_t = x_t.to(device)
            y_out = model(x_t)
            y_t = y_test[n_val]
            y_t = y_t.to(device)
            result = criterion(y_out, y_t)
            loss_test[n_val] = result.detach().cpu().numpy()
        loss_val[epoch] = np.mean(loss_test)
        # save model if it reduces the loss
        if loss_val[epoch] == np.min(loss_val):
            torch.save(model.state_dict(), MODEL_PATH)

        print("Validation errors for epoch %d: %f ,  took time: %f" % (epoch, loss_val[epoch], elapsed))


# evaluate
def evaluate(text_list):
    # model params
    input_size = 300
    hidden_dim = 300
    LSTM_layers = 2
    batch_size = 100

    # load model
    model = Model(input_size, hidden_dim, LSTM_layers, batch_size, vocab_size)
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
    if len(test)//batch_size > 0:
        for i, x in enumerate(get_batches(test, batch_size)):
            outputs = model(x)

    else:
        outputs = model(test)
    for i in range(len(outputs)):
        print(f"{text_list[i]}   :  {float(outputs[i])}")



# Main
if __name__ == "__main__":
    train()
    # evaluate(["I love you","I want to hit someone","go to hill shit","I will kill you"])


