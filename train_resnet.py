'''
使用CIFAR10数据集训练一个正常的ResNet
keras指出ResNet32 200轮	acc=92.46 %	论文官方acc=92.49 %
数据集描述:
cifar10 有10类,图像大小为32*32
**实际训练效果**训练的效果不错,resnet18 epoch200时候test acc=0.956,loss=0.1652
https://github.com/kuangliu/pytorch-cifar
'''
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import sys
# from resnet_10 import resnet32
from model import ResNet18, ResNet34
from utils import *
writer = SummaryWriter(f"../runs/trainresnet{time.time():.2f}")
writer.add_text("实验描述", "训练resnet18,transform 只有totensor,一定要训练出来啊")
os.makedirs("../weights", exist_ok=True)
# 超参数
batch_size = 128
epochs = 200
lr = 0.1
# 数据集 # [, 3, 32, 32]
trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=False, transform=transform_only_tensor_train)
testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=False, transform=transform_only_tensor)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# 模型
model = ResNet18().cuda()
model = torch.nn.DataParallel(model)

cudnn.benchmark = True
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)  # 5的10的-4次方,0.0005
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def train_and_test():
    # 训练 测试
    for epoch in range(epochs):
        # 训练
        model.train()
        loss_train = 0
        acc_train = 0
        for batch, (data, label) in enumerate(tqdm(trainloader)):
            data = data.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            outputs = model(data)
            outputs_label = torch.argmax(outputs, dim=1)
            acc_train += (outputs_label == label).int().sum().item()
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        loss_train = loss_train / len(trainloader)
        acc_train = acc_train / len(trainset)
        print(f"train{epoch},acc={acc_train},loss={loss_train}")
        # 测试
        model.eval()
        acc_test = 0
        loss_test = 0
        with torch.no_grad():
            for batch, (data, label) in enumerate(tqdm(testloader)):
                data = data.cuda()
                label = label.cuda()
                outputs = model(data)
                outputs_label = torch.argmax(outputs, dim=1)
                loss = criterion(outputs, label)
                acc_test += (outputs_label == label).int().sum().item()
                loss_test += loss.item()
        acc_test /= (len(testset))
        loss_test /= len(testloader)
        print(f"test epoch{epoch},acc={acc_test},loss={loss_test}")
        writer.add_scalars("loss", {"loss_train": loss_train, "loss_test": loss_test}, epoch)
        writer.add_scalars("acc", {"acc_train": acc_train, "acc_test": acc_test}, epoch)
        # 保存
        state = {
            "model": model.state_dict(),
            "acc": acc_test,
            "epoch": epoch
        }
        torch.save(state, f"../weights/resnet18--transform_onlyToTensor--epoch{epoch}.pth")
        scheduler.step()


# train_and_test()

def test(epoch=1):
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=False, transform=transform_tensor_norm)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=False, transform=transform_tensor_norm)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                              shuffle=True, num_workers=0)

    testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                             shuffle=True, num_workers=0)
    model.load_state_dict(torch.load("../betterweights/mymodel__epoch30__transform_tensor_norm.pth")["model"])
    model.eval()
    acc_test = 0
    loss_test = 0
    with torch.no_grad():
        for batch, (data, label) in enumerate(tqdm(testloader)):
            data = data.cuda()
            label = label.cuda()
            outputs = model(data)
            outputs_label = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, label)
            acc_test += (outputs_label == label).int().sum().item()
            loss_test += loss.item()
    acc_test /= (len(testset))
    loss_test /= len(testloader)
    print(f"test epoch{epoch},acc={acc_test},loss={loss_test}")


# test()
if __name__ == "__main__":
    train_and_test()
    # test()