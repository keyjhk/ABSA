# -*- coding: utf-8 -*-
# file: squeeze_embedding.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.


import torch
import torch.nn as nn
import numpy as np


class SqueezeEmbedding(nn.Module):
    """
    Squeeze sequence embedding length to the longest one in the batch
    """

    def __init__(self, batch_first=True):
        super(SqueezeEmbedding, self).__init__()
        self.batch_first = batch_first

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack -> unpack ->unsort
        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        # x: batch,MAX_LEN,embed_dim  ;x_len: batch
        out_test = test(x, x_len)

        x_sort_idx = torch.sort(-x_len)[1].long()  # batch 降序 ，取索引
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()  # 记录第0行、第1行的索引位置
        x_len = x_len[x_sort_idx]  # 等效于 x_len = torch.sort(x_len,descending=True)[0]
        x = x[x_sort_idx]  # 按照样本长度的顺序 重新排列 最长的在第一行 次之第二行 以此类推
        """pack"""
        # enforce= False 其实也可以 但是 上面完成了排序操作
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len.cpu(), batch_first=self.batch_first)
        """unpack: out"""
        out = torch.nn.utils.rnn.pad_packed_sequence(x_emb_p, batch_first=self.batch_first)  # (sequence, lengths)
        out = out[0]
        """unsort"""
        # 按照样本一开始的顺序恢复
        # unsorted_index 记录了原来的索引序号
        out = out[x_unsort_idx]

        return out


def test(x, x_len):
    padx = torch.nn.utils.rnn.pack_padded_sequence(x, x_len.cpu(), enforce_sorted=False,batch_first=True)
    return torch.nn.utils.rnn.pad_packed_sequence(padx,batch_first=True)[0] # 这是等效的 ！！！
