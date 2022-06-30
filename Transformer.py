import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
from .Attention import Multihead_attention

# 基于位置的前馈网络（FFN，其实就是线性层，名字叫的好听点）
class PositionWiseFFN(nn.Module):
    """基于位置的前馈⽹络"""
    # 因为⽤同⼀个多层感知机对所有位置上的输⼊进⾏变换，所以当所有这些位置的输⼊相同时，它们的输出也是相同的
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
    **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

# ffn = PositionWiseFFN(4, 4, 8)
# ffn.eval()
# print(ffn(torch.ones((2, 3, 4)))[0])  # 可以观察到同一位置的输出值相等

# 残差连接与layernorm实现
class AddNorm(nn.Module):
    """残差连接后进⾏层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


# 编码器block的实现，transformer是要叠好几个encoderblock和decoderblock
class EncoderBlock(nn.Module):
    """transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = Multihead_attention(
        key_size, query_size, value_size, num_hiddens, num_heads, dropout,
        use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
        ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)
    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))