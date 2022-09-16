# 从VGG网络开始，引入了启发式网络，出现了块(block)的概念，重复层的操作来加深网络，从而才是深度学习

import torch
import torch.nn as nn
#from d2l import torch as d2l
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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
    
# 参数说明：
# train_iter和test_iter都需要是Dataloader实例化之后的对象
# loss为损失函数，optimizer为优化器，device指定是否在GPU上训练
# 需要说明的是这个自定义train函数可能在不同的训练任务上需要做修改，这里只是针对pytorch自带的数据集进行训练
def train(epoch, train_iter, test_iter, train_net, loss, optimizer, device):    # 自定义训练函数，不用都d2l本身包里面自带的，锻炼工程能力       
    train_net.to(device)
    last_loss = [100]
    num_loss = 0
    for i in range(epoch):
        train_net.train()
        for batch, (data, label) in enumerate(train_iter):
            data, label = data.to(device), label.to(device)
            y_hat = train_net(data)
            real_loss = loss(y_hat, label)
            optimizer.zero_grad()
            real_loss.backward()
            optimizer.step()
            with torch.no_grad():
                if batch == batch_size - 1:
                    train_net.eval()
                    test_x, test_y = next(iter(test_iter))
                    test_out = train_net(test_x.float().to(device))
                    pred_y = torch.max(test_out, 1)[1].to(device).data.squeeze()
                    accurency = (test_y.to(device)==pred_y).sum().item() / len(test_y.to(device)) 
                    print("Epoch: ", i+1, "---------Train Loss: %.4f" % real_loss.item(), 
                      "---------Accurency: %.3f" % accurency)
                    last_loss.append(real_loss.item())
                    print(last_loss[i+1])
        
                    # 设置early_stop防止过拟合，当然也可以直接调pytorch的API
                    if last_loss[i+1] > last_loss[i]:
                        num_loss += 1
                    else:
                        num_loss = 0 
                    if num_loss >= 5:  
                        print("The train process is done")
                        break
        if num_loss >= 5:   # 这里必须break两次，不然好像跳不出去
            break     
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Now Train on:  ", device, "\n")
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    ratio = 4
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    # 这里电脑用的cpu就把epoch这些改小点为了速度，GPU显存不够话也调小batch_size
    lr, num_epochs, batch_size = 0.01, 30, 128 
    vggtest = VGG_11(small_conv_arch)
    # print(vggtest)   # 打印网络结构
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(vggtest.parameters(), lr=lr)
    test_data = torchvision.datasets.FashionMNIST(
        root='data',
        train=False,
        # transform参数一般是我们用来对数据进行预处理的修饰，通常跟torchvision.transforms中的函数相结合使用
        transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), ]),    
        # 把数据转化成tensor形式，且灰度值会从0-255调整到0-1
        download=True)    # 本地有就设置成False，即使是True也不会重复下载
    test_iter = DataLoader(test_data, num_workers=1, batch_size=batch_size, shuffle=False)
    training_data = torchvision.datasets.FashionMNIST(
        root="data", train=True, download=True,
        transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),]))
    train_iter = DataLoader(training_data, num_workers=1, batch_size=batch_size, shuffle=True)
    # print(test_data.data[0].shape)    # 看下图片数据大小
    # print(test_data.targets[0]) # 类别标签

    ##
    # 注意一个问题，使用pytorch自带的数据集时一定要导入到dataloader里面才能resize成功
    ##

    train(epoch=num_epochs, train_iter=train_iter, test_iter=test_iter, 
          train_net=vggtest, loss=loss, optimizer=optimizer, device=device)
    
    # train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    # d2l.train_ch6(vggtest, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
