'''
检查一下苗师兄backward反向传播不同参数的代码
'''
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torchvision
from torch.utils.data import DataLoader
from model import getResNet, AutoEncoder_Miao,ResNet18_sigmoid
from utils import transform_only_tensor
import numpy as np


def getLoader(dataset, batch_size, shuffle=True, num_workers=2, set_only_two_classes=False, c1=0,
              c2=1):
    '''
    获得dataloader,cifar10类,只取两类
    :param dataset:
    :param batch_size:
    :param shuffle:
    :param num_workers:
    :param set_only_two_classes: True则cifar10类,只取两类
    :param c1:
    :param c2:
    :return:
    '''
    print(f"原始数据集大小:{len(dataset)}", end=",")
    y = np.array(dataset.targets)
    if set_only_two_classes:
        index = np.where((y == c1) | (y == c2))[0]
        x = dataset.data[index]
        y = np.array(dataset.targets)[index]
        dataset.data = x
        dataset.targets = y
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    print(
        f"新数据集大小{len(dataset)},loader数量:{len(dataloader)},data形状:{dataset.data.shape},targects形状{len(dataset.targets)}")
    return dataloader


discriminator = ResNet18_sigmoid().cuda()
generator = AutoEncoder_Miao().cuda()

trainset = torchvision.datasets.CIFAR10(root="../data", train=True, transform=transform_only_tensor)
trainloader = getLoader(trainset, 128, True, 2, True, 0, 1)
# criterion = torch.nn.
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
epochs = 10
# one=torch.FloatTensor([1])
for epoch in range(epochs):
    for batch, (data, label) in enumerate(trainloader):
        data = data.cuda()
        label = label.cuda()
        pred = discriminator(data)
        loss = pred
        print(loss.shape,loss.mean())
        optimizer_d.zero_grad()
        loss.backward(-torch.ones_like(loss))
        optimizer_d.step()
        # optimizer_d.zero_grad()
        # loss.backward(torch.ones_like())

