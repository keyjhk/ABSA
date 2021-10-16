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


class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM'):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len.to('cpu'), batch_first=self.batch_first)

        # process using the selected RNN
        if self.rnn_type == 'LSTM':
            # LSTM 网络会返回3个 相比rnn、gu
            # ct 是最后时间步的隐藏状态
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else:
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[
            x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)  # 这一步是为了恢复 原来的顺序

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]  #
            out = out[x_unsort_idx]
            """unsort: out c"""
            if self.rnn_type == 'LSTM':
                ct = torch.transpose(ct, 0, 1)[
                    x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
                ct = torch.transpose(ct, 0, 1)

            return out, (ht, ct)


def test(x, x_len):
    padx = torch.nn.utils.rnn.pack_padded_sequence(x, x_len.cpu(), enforce_sorted=False,batch_first=True)
    return torch.nn.utils.rnn.pad_packed_sequence(padx,batch_first=True)[0] # 这是等效的 ！！！



