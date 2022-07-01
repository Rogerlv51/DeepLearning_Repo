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
# Transformer编码器输出的形状是（批量⼤⼩，时间步数⽬，num_hiddens）
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

# 测试Transformer_Encoder，两个block
# encoder = TransformerEncoder(200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
# encoder.eval()
# valid_lens = torch.tensor([3, 2])
# print(encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape)

# transformer解码器也是由多个相同的层组成。在DecoderBlock类中实现的每个层包含了三个⼦层:
# 解码器⾃注意⼒、“编码器-解码器”注意⼒和基于位置的前馈⽹络。这些⼦层也都被和紧随的layernorm围绕
# 在掩蔽多头解码器⾃注意⼒层（第⼀个⼦层）中，查询、键和值都来⾃上⼀个解码器层的输出
# 为了在解码器中保留⾃回归的属性，其掩蔽⾃注意⼒设定了参数dec_valid_lens，以便任何查询都只会与解码器中所有已经⽣成词元的位置（即直到该查询位置为⽌）进⾏注意⼒计算
class DecoderBlock(nn.Module):
    """解码器中第i个块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        
        self.attention1 = Multihead_attention(num_heads, dropout, query_size, key_size, value_size, num_hiddens)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = Multihead_attention(num_heads, dropout, query_size, key_size, value_size, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)
    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同⼀时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元⼀个接着⼀个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表⽰
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            # 其中每⼀⾏是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        # ⾃注意⼒
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器－解码器注意⼒。
        # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

# 测试decoder-block
valid_lens = torch.tensor([3, 2])
encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
encoder_blk.eval()
decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
decoder_blk.eval()
X = torch.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
print(decoder_blk(X, state)[0].shape)




