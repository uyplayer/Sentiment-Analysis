#!/usr/bin/python3
# -*- coding: UTF-8 -*-
"""
@Time     : 2/14/21 7:20 PM
@Author   : uyplayer
@Site     : uyplayer.pw
@contact  : uyplayer@qq.com
@File     : Params.py
@Software : PyCharm
"""
# dependency library
from __future__ import print_function
# system
import argparse

# model PARAMS
parser = argparse.ArgumentParser()
parser.add_argument('-ne', '--neutral', type=str,default="NEUTRAL")
parser.add_argument('-sq_len', '--sequence_length', type=int, default=100, help='Max sentence length in ''train/test '
                                                                                'data (''Default: 50)')
parser.add_argument('-embed_dim', '--embedding_dim', type=int, default=200, help='word_embedding_dim')
parser.add_argument('-multiheads', '--multiheads', type=int, default=4, help='multi_head_dim')
parser.add_argument('-multiheads_dim', '--multiheads_dim', type=int, default=200, help='multi_head_dim')
parser.add_argument('-pos_max_len', '--pos_max_len', type=int, default=5000, help='position embedding max len of row')
parser.add_argument('-mf', '--model_path',type=str,default='./TL/save_models/TCL_tweets.pth', help='model file saving dir')
parser.add_argument('-d_model', '--d_model', type=int, default=200, help='demintion')
parser.add_argument('-d_ff', '--d_ff', type=int, default=200, help='demintion')

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
# CNN
parser.add_argument("--input_channel_1", type=int, default=1, help="INPUT_CHANNEL_1")
parser.add_argument("--output_channel_1", type=int, default=2, help="OUTPUT_CHANNEL_1")
parser.add_argument("--input_channel_2", type=int, default=2, help="INPUT_CHANNEL_2")
parser.add_argument("--output_channel_2", type=int, default=4, help="OUTPUT_CHANNEL_2")
# Training parameters
parser.add_argument("--batch_size", type=int, default=64, help="Batch Size (Default: 64)")
parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs (Default: 10)")
parser.add_argument("--display_every", type=int, default=10, help="Number of iterations to display training info.")
parser.add_argument("--evaluate_every", type=int, default=100, help="Evaluate model on dev set after this many steps")
parser.add_argument("--checkpoint_every", type=int, default=100, help="Save model after this many steps")
parser.add_argument("--num_checkpoints", type=int, default=5, help="Number of checkpoints to store")
parser.add_argument("--learning_rate", type=float, default=1e-2, help="Which learning rate to start with. (Default: 1e-3)")
parser.add_argument("--epochs", type=int, default=30, help="epochs")
args = parser.parse_args()



