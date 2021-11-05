# -*- coding: utf-8 -*-
# file: attention.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim: 没有进行wq/wk映射前的原始向量维度
        :param hidden_dim: wq/wk 映射后的向量维度 ，q与k向量维度是统一的
        :param out_dim: 映射scores*v的维度
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q, v=None, mask=None):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)

        # default head=1
        # q:batch,q_len,embed_dim ;  k:batch,seq_len,embed_dim
        mb_size = k.shape[0]  # batch_size
        k_len = k.shape[1]  # key_seq_len
        q_len = q.shape[1]  # query_seq_len

        # head * hidden_dim == embed_dim
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)  # mb_size,k_len,head,hidden_dim
        # 将head、batch等维度提前 便于计算注意力(矩阵乘) 因为它只和后两个维度有关
        # permute: head,mb_size,k_len,hidden_dim  ;view: head*mb_size,k_len,hidden_dim
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        # qx 同理 head*mb_size ,q_len,hidden_dim
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)

        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)  # head*mb_size,hidden_dim,k_len
            # head*mb_size,q_len,hidden_dim [BMM] head*mb_size,hidden_dim,k_len
            # ==> head*mb_size,q_len,k_len
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            # concat
            # unsqueeze:head*mb_size,1,k_len,hidden_dim  ;
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)  # head*mb_size,q_len,k_len,hidden_dim
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)  # head*mb_size,q_len,k_len,hidden_dim
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            # general  : (q*w)*k^T w矩阵添加是为了非线性操作
            # head*mb_size ,q_len,hidden_dim [BMM] hidden_dim,hidden_dim
            # ==> head*mb_size,q_len,hidden_dim
            qw = torch.matmul(qx, self.weight)  # head*mb_size,q_len,hidden_size ==> head*mb_size,q_len,hidden_size
            kt = kx.permute(0, 2, 1)  # head*mb_size,hidden_dim,k_len
            # head*mb_size,q_len,hidden_dim [BMM] head*mb_size,hidden_dim,k_len
            # head*mb_size,q_len,k_len
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')

        if mask is not None:
            # mask: q_len,k_len
            score = score.masked_fill(mask, value=torch.tensor(-1e9))
        score = F.softmax(score, dim=-1)  # head*mb_size,q_len,k_len
        # 到此为止 已经得到了可以作为权重的score了 以下操作在于融合 attention 下的value(key)

        # 注意力融合 value
        # head*mb_size,q_len,k_len [BMM] head*mb_size,k_len,hidden_dim
        # ==> head*mb_size,q_len,hidden_dim
        output = torch.bmm(score, kx) if v is None else torch.bmm(score, v)  # (n_head*?, q_len, hidden_dim)
        # split:  List[(mb_size,q_len,hidden_dim)]* head
        # concat: mb_size,q_len,hidden_dim*head
        # 拼接所有头 Head= concat(head1,head2,...)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (batch, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (mb_size, q_len, out_dim) 做一次线性变化
        output = self.dropout(output)  # mb_size, q_len, out_dim ; outdim在默认情况下等于embed_dim

        # out: 注意力融合的 value
        # score： 注意力权重矩阵
        return output, score


class NoQueryAttention(Attention):
    '''q is a parameter
    no query 的意思就是 query 没有显式提供 只有key
    其实是 w权重矩阵作为隐式的query   score=tanh(w*(k))
    init里的q 就是该权重矩阵
    '''

    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', q_len=1,
                 dropout=0):
        super(NoQueryAttention, self).__init__(embed_dim, hidden_dim, out_dim, n_head, score_function, dropout)
        self.q_len = q_len  # q_len=1 means 1 group score ;
        # embed_dim == key's embed_dim ,for attention
        self.q = nn.Parameter(torch.Tensor(q_len, embed_dim))
        self.reset_q()

    def reset_q(self):
        stdv = 1. / math.sqrt(self.embed_dim)
        self.q.data.uniform_(-stdv, stdv)

    def forward(self, k, **kwargs):
        mb_size = k.shape[0]  # minibatch
        q = self.q.expand(mb_size, -1, -1)  # minibatch,q_len,embed_dim
        return super(NoQueryAttention, self).forward(k, q)


def squeeze_embedding(embeeding, len, batch_first=True,padding_value=0):
    if len.device != 'cpu':
        len = len.to('cpu')
    pad_embed = pack_padded_sequence(embeeding, len, batch_first=batch_first, enforce_sorted=False)
    return pad_packed_sequence(pad_embed, batch_first=batch_first,padding_value=padding_value)[0]
