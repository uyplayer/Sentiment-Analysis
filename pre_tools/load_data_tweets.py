#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/14 13:50
# @Author  : uyplayer
# @Site    : uyplayer.pw
# @File    : load_data_tweets.py
# @Software: PyCharm

'''
this script is for  loadding data from training.1600000.processed.noemoticon.csv
datasets/English/Sentiment140 dataset with 1.6 million tweets/training.1600000.processed.noemoticon.csv
Dataset details
target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
ids: The id of the tweet ( 2087)
date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
flag: The query (lyx). If there is no query, then this value is NO_QUERY.
user: the user that tweeted (robotickilldozr)
text: the text of the tweet (Lyx is cool)
'''

# dependency library
# system
import os,re,math,random
# data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
# nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
# Scikit-learn
from sklearn.model_selection import train_test_split

# root path
FILE = "./datasets/English/Sentiment140 dataset with 1.6 million tweets/training.1600000.processed.noemoticon.csv"

# params  DATASET
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
TRAIN_SIZE = 0.8

# read *.csv file
print(" Open *.csv file")
tweets = pd.read_csv(FILE, encoding="ISO-8859-1", names=DATASET_COLUMNS)
# print("Dataset size:", len(tweets)) # Dataset size: 1600000
# print(tweets.head(5))
# handle the label
decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
def decode_sentiment(label):
    return decode_map[int(label)]
# print(tweets.target.apply(lambda x:decode_sentiment(x)))

target_cnt = Counter(tweets.target)
# print(target_cnt) #Counter({0: 800000, 4: 800000})

# Pre-Process dataset
print(" Pre-Process dataset ")
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

def preprocess(text, stem=False):
    # remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

tweets.text = tweets.text.apply(lambda x: preprocess(x))
# print(tweets.text)

print(" Split train and test ")
tweets_train, tweets_test = train_test_split(tweets, test_size=1-TRAIN_SIZE, random_state=42)
# print("TRAIN size:", len(tweets_train)) # TRAIN size: 1280000
# print("TEST size:", len(tweets_test)) # TEST size: 320000

# return data
def return_data(rate=1):
    # labels  0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"
    train_x = tweets_train.text
    train_y = tweets_train.target
    test_x = tweets_test.text
    test_y = tweets_test.target

    len_train = int(len(train_x)*rate)
    len_test = int(len(test_x)*rate)

    train_x = train_x[:len_train]
    train_y = train_y[:len_train]
    test_x = test_x[:len_test]
    test_y = test_y[:len_test]

    return train_x, train_y, test_x, test_y

