import torch.nn as nn
import torch

# 实现基础残差连接块，注意在论文中有两种残差连接块，一种是基础的，一种是带有下采样的，下采样stride=2，且要升维
class BasicBlock(nn.Module):  # 对应18层和34层的残差块
    expansion = 1  # 表示前后卷积层的卷积核个数是没有变化的，倍数为1
    # downsample对应虚线残差连接的1×1卷积层
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
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

# 接下来定义50层以后的残差结构
class Bottleneck(nn.Module):
    expansion = 4  # 表示前后卷积层的卷积核个数是变化的，倍数为4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.Relu = nn.ReLU(inplace=True)  # 节约内存
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.Relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.Relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.Relu(out)
        return out

# 定义主干网络
class ResNet(nn.Module):
    # block指定要使用那种残差块，block_num指定每个残差块的个数，num_classes指定分类的类别数，include_top方便我们在这个网络架构上进行微调
    def __init__(self, block, block_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  # 进入残差块的输入通道数为64
        # 112下采样到56，stride=2，步长的计算方式为：图像输出大小 = (图像输入大小-kernel_size+2*padding)/stride + 1，pytorch采用向下取整
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    # 构建残差块的叠加
    def _make_layer(self,block,channel,block_num,stride=1):
        pass

