from matplotlib.pyplot import cla
import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F


'''
ResNet沿⽤了VGG完整的3×3卷积层设计。残差块⾥⾸先有2个有相同输出通道数的3×3卷积层。每个卷积
层后接⼀个批量规范化层和ReLU激活函数。然后我们通过跨层数据通路，跳过这2个卷积运算，将输⼊直接
加在最后的ReLU激活函数前。这样的设计要求2个卷积层的输出与输⼊形状⼀样，从⽽使它们可以相加。如
果想改变通道数，就需要引⼊⼀个额外的1×1卷积层来将输⼊变换成需要的形状后再做相加运算
'''

# 实现残差模块以达到x+[f(x)-x] = f(x)
# 通过kernelsize为1的卷积层和调整stride大小，我们可以任意设置自己想要的输出通道，以及输出的高宽
class Residual(nn.Module):
    def __init__(self, in_channels, num_channels, use_11conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_channels,
                kernel_size=3, stride=strides, padding=1)   
        # 如果步长为1的话，显然padding=(kernel_size-1)/2，这个公式记住吧，向上取整
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels,
                kernel_size=3, padding=1)
        if use_11conv:
            self.conv3 = nn.Conv2d(in_channels, num_channels, 1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
    def forward(self, x):
        Y = F.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            x = self.conv3(x)   # 确保输入形状与输出保持一致，主要是通道数目channels
        Y += x
        return F.relu(Y)

# ResNet模型搭建，这里简单点就用ResNet-18
# 在输出通道数为64、步幅为2的7×7卷积层后，接步幅为2的3×3的Maxpooling，每个卷积层后增加BatchNorm
# 在接入这一个卷积层之后，ResNet则使⽤4个由残差块组成的模块，每个模块使⽤若⼲个同样输出通道数的残差块
# 第⼀个模块的输出通道数同输⼊通道数保持⼀致，利用前面1×1卷积。由于之前已经使⽤了步幅为2的Maxpooling，
# 所以⽆须减⼩⾼和宽。之后的每个模块在第⼀个残差块⾥将上⼀个模块的通道数翻倍，并将⾼和宽减半。

# 第一个残差块要做特殊处理，定义残差块
def resnet_block(in_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for num in range(num_residuals):
        if num == 0 and not first_block:   # 判断第一个残差连接是否需要做特殊处理
            blk.append(Residual(in_channels, num_channels, use_11conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk



if __name__ == "__main__":
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    # 最后ResNet中加⼊全局平均汇聚层，以及全连接层输出
    net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(), nn.Linear(512, 10))
    print(net, "\n")
    lr, num_epochs, batch_size = 0.05, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
