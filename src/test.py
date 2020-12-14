#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/14 11:58
# @Author  : uyplayer
# @Site    : uyplayer.pw
# @File    : test.py
# @Software: PyCharm

import torch
import torch.nn as nn

input_shape = (2, 3, 4)
x = torch.randn(input_shape)
print(x)
m = torch.mean(x,dim=1)
print(m.shape)
print(m)