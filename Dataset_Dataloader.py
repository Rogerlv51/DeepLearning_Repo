import os
import pandas as pd
import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2 as cv
import matplotlib.pyplot as plt

# 通常使用Compose把你要做的transform全都包起来比如：
trans = transforms.Compose([
    transforms.PILToTensor(),    # 这一步transforms是必须要有的，不然pytorch读取不了
    transforms.Resize((224, 224)),
    transforms.ToTensor(),])

class mydataset(Dataset):
    def __init__(self, label_name, img_dir, transform=None, target_transform=None):
        super().__init__()
        self.dataset = pd.read_csv(label_name)
        self.img_dir = img_dir
        self.transforms = transform
        self.target_transforms = target_transform

    # The __len__ function returns the number of samples in our dataset
    def __len__(self):   # 返回数据集的长度，这个看你init里面如何定义读取数据集的方式
        return len(self.dataset)
    
    # The __getitem__ function loads and returns a sample from the dataset at the given index
    # 我们通过这个函数把image和label返回成pytorch能够识别的方式，注意这里处理时仅针对一个样本
    def __getitem__(self, index):  # index默认是行索引
        # 拼接图片路径，一般img_dir只是我们存放一些列图片的文件夹地址
        # 由于fashionmnist数据集里面的文件名称为tshirt1.jpg, 0；所以这里采用了这种方式读取
        # 针对自己的数据集有所不同，我们应当灵活处理数据
        image_path = os.path.join(self.img_dir, self.dataset.iloc[index, 0])
        # 通常来讲要把图片转化成tensor形式系统才能识别，当使用你自己收集的数据集时
        image = cv.imread(image_path)   # 这里我用opencv库读取图片地址，别的方式也可以
        label = self.dataset.iloc[index,1]
        if self.transforms:
            image = self.transforms(image)
        if self.target_transforms:
            label = self.target_transforms(label)
        return image, label

if __name__ == "__main__":
    training_data = datasets.FashionMNIST(root="data", train=True, download=False,
    transform=trans)

    test_data = datasets.FashionMNIST(root="data", train=False, download=False,
    transform=transforms.ToTensor())

    training_dataloader = DataLoader(training_data, num_workers=1, batch_size=64, shuffle=True)

    test_dataloader = DataLoader(test_data, num_workers=1, batch_size=64, shuffle=True)
    print(test_data.data[0].shape)
    print(test_data.targets[0])     # 类别是9即'Ankle boot'
    plt.imshow(training_data.data[0], cmap='gray')   # 数据显示一下
    plt.show()