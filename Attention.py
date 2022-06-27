from turtle import forward
from typing import ValuesView
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from d2l import torch as d2l
from Masked_Softmax import masked_softmax


# Additive Attention 加性注意力，可学习的参数为Wk（h×k），Wq（h×q），value维度为h
# 等价于将query和key合并起来后放入一个隐藏层大小为h，输出大小为1的单隐藏层MLP中得到一个注意力分数，激活函数为tanh
class Additive_Attention(nn.Module):
    def __init__(self, key_size, query_size, hidden_size, dropout, **kwargs):
        super(Additive_Attention, self).__init__(**kwargs)
        self.Wk = nn.Linear(key_size, hidden_size, bias=False)     # 公式中是不需要bias的
        self.Wq = nn.Linear(query_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, 1, bias=False)          # 输出为一个打分值
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, querys, keys, values, valid_lens):
        querys, keys = self.Wq(querys), self.Wk(keys)
        # 在维度扩展后，
        # `queries` 的形状：(`batch_size`, 查询的个数, 1, `num_hidden`)
        # `key` 的形状：(`batch_size`, 1, “键－值”对的个数, `num_hiddens`)
        # 使⽤⼴播⽅式进⾏求和
        features = querys.unsqueeze(2) + keys.unsqueeze(1)       # 这里我觉得还是看自己query和key给的数据形式是啥样的
        features = F.tanh(features)
        # `self.w_v` 仅有⼀个输出，因此从形状中移除最后那个维度。
        # `scores` 的形状：(`batch_size`, 查询的个数, “键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)        # 以batch的形式做矩阵乘法


# Dot-product Attention 点积注意力，就是transformer里面用的那种，query和key的长度都为d，
# 则可以让它们两个做点积后除以根号d再经过softmax之后，与value值相乘，拓展到多维最后输出得到的维度是n×v，v即value的长度
# 点积即两向量对应元素相乘再相加，得到的是一个值
class DotProd_Attention(nn.Module):
    def __init__(self, dropout, **kwargs) -> None:
        super(DotProd_Attention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, querys, keys, values, valid_lens=None):
        d = querys.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(querys, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)



if __name__ == '__main__':
    # 测试加性注意力分数
    # 这里用两个batch
    # 查询、键和值的形状为（批量⼤⼩，步数或词元序列⻓度，特征⼤⼩），实际输出为(2, 1, 20)、(2, 10, 2)和(2, 10, 4)
    # 注意⼒汇聚输出的形状为（批量⼤⼩，查询的步数，值的维度）
    query, key = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
    # 这里设置不同batch使用相同values，两个矩阵是相同的
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1) 
    valid_lens = torch.tensor([2, 6])
    # 貌似这个hidden_size的维度并不会影响输出打分
    add_attention_test = Additive_Attention(key_size=2, query_size=20, hidden_size=100, dropout=0.1)
    add_attention_test.eval()
    data = add_attention_test(query, key, values, valid_lens)
    print(data, "\n")

    # 测试点积注意力评分函数
    # 我们使⽤与先前加性注意⼒例⼦中相同的键、值和有效⻓度
    # 对于点积操作，我们令查询的特征维度与键的特征维度⼤⼩相同
    query2 = torch.normal(0, 1, (2, 1, 2))   # 保证最后一个维度相同
    dot_attention_test = DotProd_Attention(dropout=0.5)
    dot_attention_test.eval()
    x = dot_attention_test(query2, key, values, valid_lens)
    print(x)

    # 与加性注意⼒演⽰相同，由于键包含的是相同的元素，⽽这些元素⽆法通过任何查询进⾏区分，
    # 因此获得了均匀的注意⼒权重