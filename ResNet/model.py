import torch.nn as nn
import torch

# 实现基础残差连接块，注意在论文中有两种残差连接块，一种是基础的，一种是带有下采样的，下采样stride=2，且要升维
class BasicBlock(nn.Module):  # 对应18层和34层的残差块
    expansion = 1  # 表示前后卷积层的卷积核个数是没有变化的，倍数为1
    # downsample对应虚线残差连接的1×1卷积层
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(BasicBlock.self).__init__()
        # 注意所有的卷积层都不用偏置
        # 残差块第一个卷积层，stride=1为实线连接，stride=2为虚线连接
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        # 残差块第二个卷积层，实线虚线都一样，且stride为1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # 记录下输入，方便残差连接
        if self.downsample is not None:
            identity = self.downsample(x)   # 虚连接的情况，要对输入下采样加升维才能执行后续加法操作
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # 残差连接
        out = self.relu(out)
        return out