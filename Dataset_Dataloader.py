import os
import pandas as pd
import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

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
    def __getitem__(self, index):
        image = os.path.join(self.img_dir, )
        return image, label