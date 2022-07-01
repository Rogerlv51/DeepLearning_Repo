import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
from Attention import Multihead_attention

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
# 测试下前馈网络
# ffn = PositionWiseFFN(4, 4, 8)
# ffn.eval()
# print(ffn(torch.ones((2, 3, 4)))[0])  # 可以观察到同一位置的输出值相等

# layerNorm和batchNorm的区别在于说，layernorm是针对于一个样本的所有特征来做归一化的，使得从一个样本上看过去是均值为0方差为1
# 而batchnorm则是对当前一个batch内所有样本的同一列特征来做归一化，也就是说两者处理的维度不同

# 残差连接与layernorm实现
class AddNorm(nn.Module):
    """残差连接后进⾏层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
# 测试下残差连接层，两个输入维度要一致
# add_norm = AddNorm([3, 4], 0.5)
# add_norm.eval()
# print(add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape)

# 编码器block的实现，transformer是要叠好几个encoderblock和decoderblock
class EncoderBlock(nn.Module):
    """transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = Multihead_attention(
            num_heads, dropout, query_size, key_size, value_size, 
            num_hiddens)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
        ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)
    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
# 可以看到，transformer编码器中的任何层都不会改变其输⼊的形状
# 测试编码器block
# X = torch.ones((2, 100, 24))
# valid_lens = torch.tensor([3, 2])   # 分别设置两个样本的有效长度
# encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
# encoder_blk.eval()
# print(encoder_blk(X, valid_lens).shape)

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建⼀个⾜够⻓的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
        -1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

# 叠加encoder_block
class TransformerEncoder(nn.Module):
    """transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
    num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
    num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
            EncoderBlock(key_size, query_size, value_size, num_hiddens,
            norm_shape, ffn_num_input, ffn_num_hiddens,
            num_heads, dropout, use_bias))
    def forward(self, X, valid_lens):
    # 因为位置编码值在-1和1之间，
    # 因此嵌⼊值乘以嵌⼊维度的平⽅根进⾏缩放，
    # 然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X