#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/14 11:58
# @Author  : uyplayer
# @Site    : uyplayer.pw
# @File    : test.py
# @Software: PyCharm

from sklearn import metrics

y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]

y_true_1 = [0, 1, 2, 0, 1, 2]
y_pred_1 = [0, 2, 3, 0, 0, 1]

y_true_2 = [0, 1, 2, 0, 1, 2]
y_pred_2 = [0, 1, 1, 0, 0, 1]


print(metrics.precision_score(y_true, y_pred, average='macro'))
print(metrics.precision_score(y_true_1, y_pred_1, average='macro'))
print(metrics.precision_score(y_true_2, y_pred_2, average='macro'))






