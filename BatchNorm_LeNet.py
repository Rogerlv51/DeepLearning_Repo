
import torch
from torch import nn
from d2l import torch as d2l

# 哈德玛积：矩阵对应位置元素相乘，又称基本积，符号是一个圆圈里面一个点

# 批标准化，batchnormalize，对一个小批量里面的样本做归一化使其均值0，方差为1，
# 在算方差的时候一般会在后面加个eps很小的一个正数，来保证标准差始终大于0，这样归一化不会出现除以0的情况
# 训练模式下的batchnormalization和预测模式下是有一些区别的，训练的时候使用实际算出来的均值和标准差
# 预测的时候常用的一种方法是：通过移动平均估算整个训练数据集的样本均值和⽅差，并在预测时使⽤它们得到确定的输出
# 通常批标准化层要放在全连接层（或卷积层）之后，激活函数层之前
# 批标准化的公式： gamma 哈德玛积 ((x-均值)/标准差) + beta，gamma和beta是拉伸参数和平移参数要放到模型里学习

# 下面从0实现批标准化，当然现在框架中都有API直接调用，训练速度会比自己写函数快得多
def batch_normal(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 使用is_grad_enabled判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用传入移动平均所得的moving_mean和moving_var
        X_hat = (X-moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)   # 判断一下要处理的数据是否从全连接或卷积层输出的
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X-mean)**2).mean(dim=0)
        else:
            # 二维卷积过后一般输出维度是4维，输出通道数在第2个维度上
            # 使⽤⼆维卷积层的情况，计算通道维上（axis=1）的均值和⽅差。
            # 这⾥我们需要保持X的形状以便后⾯可以做⼴播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下直接使用当前计算出的mean和var算标准化
        X_hat = (X-mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean 
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta    # 平移和缩放 
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    # num_features：全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表⽰全连接层，4表⽰卷积层
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))   # 设置成网络内部参数参与训练
        self.beta = nn.Parameter(torch.zeros(shape))
        # ⾮模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
    def forward(self, X):
        
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var，因为这两个参数要参与求梯度优化，所以才要记录下来
        Y, self.moving_mean, self.moving_var = batch_normal(X, self.gamma, self.beta, self.moving_mean,
        self.moving_var, eps=1e-5, momentum=0.9)
        return Y

# 在手动设计批标准化流程的时候，学习到一个很重要的设计思想，这种思路在以后设计网络时十分重要：
# 我们⽤⼀个单独的函数定义其数学原理，⽐如说batch_normal, 然后将此功能集成到⼀个⾃定义层中，
# 其代码主要处理数据移动到训练设备（如GPU）、分配和初始化任何必需的变量、跟踪移动平均线（此处为均值和⽅差）等问题
if __name__ == "__main__":
    # 使用LeNet测试我们写的BatchNorm
    net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
            nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
            nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
            nn.Linear(84, 10))
    lr, num_epochs, batch_size = 0.5, 20, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())