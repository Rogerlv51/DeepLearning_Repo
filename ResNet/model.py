import torch.nn as nn
import torch

# 实现基础残差连接块，注意在论文中有两种残差连接块，一种是基础的，一种是带有下采样的，下采样stride=2，且要升维
class BasicBlock(nn.Module):
    # downsample对应虚线残差连接的1×1卷积层
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(BasicBlock.self).__init__()
        # 注意所有的卷积层都不用偏置
        # 残差块第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        # 残差块第二个卷积层，两个卷积层通道数都一样
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)