# 从VGG网络开始，引入了启发式网络，出现了块(block)的概念，重复层的操作来加深网络，从而才是深度学习

from turtle import forward
import torch
import torch.nn as nn
from d2l import torch as d2l

# 经典卷积神经网络 = 卷积层 + 线性ReLU激活 + 最大池化层
# block的实现，在VGG中作者使⽤了带有3×3卷积核、padding为1（保持⾼度和宽度）的卷积层，
# 和带有2×2池化窗⼝、stride为2（每个块后的分辨率减半）的最⼤池化层
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
        kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)    # sequential接收一个class对象，所以这里要加*


'''
原始VGG⽹络有5个卷积块, 其中前两个块各有⼀个卷积层, 后三个块各包含两个卷积层。第⼀个模块有64个
输出通道, 每个后续模块将输出通道数量翻倍, 直到该数字达到512。由于该⽹络使⽤8个卷积层和3个全连接
层, 因此它通常被称为VGG-11。
'''
class VGG_11(nn.Module):
    def __init__(self, conv_arch):
        super(VGG_11, self).__init__()
        conv_blks = []
        in_channels = 1
        # 卷积层部分
        for (num_convs, out_channels) in conv_arch:   # 其实就把那些块循环添加到Sequential里面就行了
            conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        self.Conv_layer = nn.Sequential(*conv_blks, nn.Flatten())
        self.linear1 = nn.Linear(out_channels*7*7, 4096)    # 因为前面一层做了flatten拉成一维的数据了
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(4096, 10)
    def forward(self, x):
        x = self.Conv_layer(x)
        x = self.dropout1(self.relu1(self.linear1(x)))
        x = self.dropout2(self.relu2(self.linear2(x)))
        x = self.linear3(x)
        return x
        

if __name__ == "__main__":
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    ratio = 4
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    vggtest = VGG_11(small_conv_arch)
    # print(vggtest)   # 打印网络结构

    
    lr, num_epochs, batch_size = 0.05, 2, 64   # 这里电脑用的cpu就把epoch这些改小点为了速度
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    d2l.train_ch6(vggtest, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    
    
