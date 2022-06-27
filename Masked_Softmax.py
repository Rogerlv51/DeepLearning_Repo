import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from d2l import torch as d2l


'''
def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.

    Defined in :numref:`sec_seq2seq_decoder`"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X
'''

# 掩码softmax操作即masked-softmax
def masked_softmax(x, valid_lens):   # valid_lens指定我们要保留的有效长度，其余部分作掩码处理，置为一个很大的负数，可以用numpy中的inf
    if valid_lens == None:    # 没指定的话，就直接softmax即可
        return F.softmax(x, dim=-1)      
        # dim=-1表示对最后一维作softmax计算，把取值范围限制在[0,1]，这里是每一行输出值的和应为1
    else:
        shape = x.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])   # 如果你只是输入了一个维度的valid_lens显然要把它复制到每一行上
            # 对于不同的样本也可以分别指定有效长度，这就是做repeat操作的目的
        else:
            valid_lens = valid_lens.reshape(-1)     # 不是1维的话，拉成1维的
        # 在最后的轴上，被遮蔽的元素使⽤⼀个⾮常⼤的负值替换，从⽽其softmax (指数)输出为0
        x = d2l.sequence_mask(x.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return F.softmax(x.reshape(shape), dim=-1)


if __name__ == '__main__':
    # 考虑由两个2×4矩阵表⽰的样本，这两个样本的有效⻓度分别为2 和3。经过遮蔽softmax 操作，超出有效⻓度的值都被遮蔽为0
    print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])), '\n')
    # 我们也可以使⽤⼆维张量为矩阵样本中的每⼀⾏指定有效⻓度
    print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))







