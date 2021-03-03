#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2021/3/2 2:12 چۈشتىن كېيىن
@Author  : uyplayer
@Site    : uyplayer.pw
@File    : vis.py.py
@contact : uyplayer@qq.com
@Software: PyCharm
"""

import numpy as np
import matplotlib.pyplot as plt
import json

'''

'''
# files  BERT Cnn Lstm Rnn
Files = ["TextRnn","HAN","RCNN","RCNNVariant","TextAttBiRNN","TextBiRNN","TL"]

# get data from files
def get_data():
    data = {}
    for filename in Files:
        collect = []
        with open(filename+".txt","r") as F:
            for line in F.readlines():
                line = line.strip()
                line = json.loads(line)
                collect.append(line)
        data[filename]=collect
    return data

# get items from data
def get_items():
    data = get_data()
    data_items = {}
    for item in Files:
        items = data[item]
        epoch =[]
        epoch_loss = []
        epoch_accuracy = []
        accuracy = []
        for each in items:
            epoch.append(each['epoch'])
            epoch_loss.append(each['epoch_loss'])
            epoch_accuracy.append(each['epoch_accuracy'])
            accuracy.append(each['accuracy'])
        data_items[item] = {"epoch":epoch,"epoch_loss":epoch_loss,"epoch_accuracy":epoch_accuracy,"accuracy":accuracy}
    return data_items

# vis
data = get_items()

# Epoch loss
def epoch_loss():
    fig, ax = plt.subplots()
    plt.ylim(0, 0.2)

    # ["BERT","Cnn","FastText","HAN","Lstm","RCNN","RCNNVariant","Rnn","TextAttBiRNN","TextBiRNN","TL"]

    # line_BERT, = ax.plot(data['BERT']['epoch'], data['BERT']['epoch_loss'], label='BERT')
    # line_Cnn, = ax.plot(data['Cnn']['epoch'], data['Cnn']['epoch_loss'], label='Cnn')
    # line_Lstm, = ax.plot(data['Lstm']['epoch'], data['Lstm']['epoch_loss'], label='Lstm')
    # line_FastText, = ax.plot(data['FastText']['epoch'], data['FastText']['epoch_loss'], label='FastText')
    # line_RCNN, = ax.plot(data['RCNN']['epoch'], data['RCNN']['epoch_loss'], label='RCNN')
    # line_RCNNVariant, = ax.plot(data['RCNNVariant']['epoch'], data['RCNNVariant']['epoch_loss'], label='RCNNVariant')






    # line_TextRnn, = ax.plot(data['TextRnn']['epoch'], data['TextRnn']['epoch_loss'],marker="*", label='TextRnn')
    # line_TextBiRNN, = ax.plot(data['TextBiRNN']['epoch'], data['TextBiRNN']['epoch_loss'],marker="+", label='TextBiRNN')
    # line_TextAttBiRNN, = ax.plot(data['TextAttBiRNN']['epoch'], data['TextAttBiRNN']['epoch_loss'],marker="x", label='TextAttBiRNN')
    line_HAN, = ax.plot(data['HAN']['epoch'], data['HAN']['epoch_loss'],marker="3", label='HAN')
    line_TL, = ax.plot(data['TL']['epoch'], data['TL']['epoch_loss'],marker="h", label='EL')

    plt.xlabel('Epoch ')
    plt.ylabel('Loss')
    plt.title('Epoch Loss')
    ax.legend()
    plt.show()

# Epoch accuracy
def epoch_accuracy():
    fig, ax = plt.subplots()
    plt.ylim(0.4, 1)

    # ["BERT","Cnn","FastText","HAN","Lstm","RCNN","RCNNVariant","Rnn","TextAttBiRNN","TextBiRNN","TL"]

    # line_BERT, = ax.plot(data['BERT']['epoch'], data['BERT']['epoch_accuracy'], label='BERT')
    # line_Cnn, = ax.plot(data['Cnn']['epoch'], data['Cnn']['epoch_accuracy'], label='Cnn')
    # line_Lstm, = ax.plot(data['Lstm']['epoch'], data['Lstm']['epoch_accuracy'], label='Lstm')
    # line_FastText, = ax.plot(data['FastText']['epoch'], data['FastText']['epoch_accuracy'], label='FastText')
    # line_RCNN, = ax.plot(data['RCNN']['epoch'], data['RCNN']['epoch_accuracy'], label='RCNN')
    # line_RCNNVariant, = ax.plot(data['RCNNVariant']['epoch'], data['RCNNVariant']['epoch_accuracy'], label='RCNNVariant')

    # line_TextRnn, = ax.plot(data['TextRnn']['epoch'], data['TextRnn']['epoch_accuracy'],marker="*", label='TextRnn')
    # line_TextBiRNN, = ax.plot(data['TextBiRNN']['epoch'], data['TextBiRNN']['epoch_accuracy'],marker="+", label='TextBiRNN')
    # line_TextAttBiRNN, = ax.plot(data['TextAttBiRNN']['epoch'], data['TextAttBiRNN']['epoch_accuracy'],marker="x", label='TextAttBiRNN')
    line_HAN, = ax.plot(data['HAN']['epoch'], data['HAN']['epoch_accuracy'],marker="3",label='HAN')
    TL, = ax.plot(data['TL']['epoch'], data['TL']['epoch_accuracy'],marker="h", label='EL')


    plt.xlabel('Epoch ')
    plt.ylabel('Accuracy')
    plt.title('Epoch Accuracy')
    # ax.legend()
    ax.legend(loc='lower right', shadow=True)
    plt.show()

epoch_loss()
epoch_accuracy()





