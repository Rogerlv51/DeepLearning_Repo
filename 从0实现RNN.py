import math
import torch.nn as nn
import torch
from d2l import torch as d2l
from torch.nn import functional as F

# 读取数据集，在时光机器数据集上训练
batch_size, num_steps = 32, 35     # 定义批量大小和时间步长度
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 在train_iter中，每个词元都表示为一个数字索引，将这些索引直接输入神经网络可能会使学习变得困难。
# 我们通常将每个词元表示为更具表现力的特征向量，最简单的方式就是one-hot编码，向量的长度显然是（不重复）词元的个数

'''
我们每次采样的小批量数据形状是二维张量：（批量大小，时间步数）。
one_hot函数将这样一个小批量数据转换成三维张量，张量的最后一个维度等于词表大小（len(vocab)）。
我们经常转换输入的维度，以便获得形状为（时间步数，批量大小，词表大小）的输出
'''

## 初始化模型参数
def get_params(vocab_size, num_hiddens, device): 
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens)) 
    W_hh = normal((num_hiddens, num_hiddens)) 
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs)) 
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度，允许参数可以被学习，可求导
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params: 
        param.requires_grad_(True)
    return params

## 为了定义循环神经网络模型，我们首先需要一个init_rnn_state函数在初始化时返回隐状态。
## 这个函数的返回是一个张量，张量全用0填充，形状为（批量大小，隐藏单元数）
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)    # 以元组的形式返回

## 定义函数计算隐状态和输出
def rnn(inputs, state, params):
    # inputs 的形状：（时间步数量， 批量大小，词表大小）
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状：(批量大小，词表大小)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)   # 使用tanh激活函数
        Y = torch.mm(H, W_hq) + b_q 
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)    

## 定义一个类来封装上述所有函数
class RNNModelScratch: 
    "从零开始实现的循环神经网络模型"
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn 
    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self. forward_fn(X, state, self.params)
    def begin_state(self, batch_size, device):    # 存储参数
        return self.init_state(batch_size, self.num_hiddens, device)

