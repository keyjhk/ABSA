import torch
import torch.nn as nn
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """
        实现层归一化
        """
        super(LayerNorm, self).__init__()
        # 注册为parameter表示系数和偏置都为可训练的量
        self.a_2 = nn.Parameter(torch.ones(features))  # 线性放缩因子
        self.b_2 = nn.Parameter(torch.zeros(features)) # 偏置
        self.eps = eps # 保证归一化分母不为0

    def forward(self, x):
        """
        :param x: 输入size = (batch , L , d_model)
        :return: 归一化后的结果，size同上
        """
        # laynorm 是在x的特征集上求归一化
        mean = x.mean(-1, keepdim=True) # 最后一个维度求均值
        std = x.std(-1, keepdim=True)  # 最后一个维度求方差
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2   #归一化并线性放缩+偏移