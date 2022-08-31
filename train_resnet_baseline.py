'''
苗师兄 baseline 代码
'''
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.utils import save_image
# import torch.backends.cudnn as cudnn
import sys
from model import *
from utils import *
from models import resnet_orig

root_result = "../result/train_resnet_basline"
os.makedirs(root_result, exist_ok=True)
writer = SummaryWriter("./baseline_runs")
batch_size = 128
epochs = 200
cifar10_train = torchvision.datasets.CIFAR10(root="../data/cifar10", train=True, download=False,
                                             transform=transform_train_cifar_miao)
cifar10_test = torchvision.datasets.CIFAR10(root="../data/cifar10", train=False, download=False,
                                            transform=transform_test_cifar_miao)
cifar100_train = torchvision.datasets.CIFAR100(root="../data/cifar100", train=True, download=False,
                                               transform=transform_train_cifar_miao)
cifar100_test = torchvision.datasets.CIFAR100(root="../data/cifar100", train=False, download=False,
                                              transform=transform_test_cifar_miao)
svhn_train = torchvision.datasets.SVHN(root="../data/svhn", split="train", download=False,
                                       transform=transform_train_cifar_miao)
svhn_test = torchvision.datasets.SVHN(root="../data/svhn", split="test", download=False,
                                      transform=transform_test_cifar_miao)

trainloader_cifar10 = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=2)
testloader_cifar10 = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=True, num_workers=2)
trainloader_cifar100 = torch.utils.data.DataLoader(cifar100_train, batch_size=batch_size, shuffle=True, num_workers=2)
testloader_cifar100 = torch.utils.data.DataLoader(cifar100_test, batch_size=batch_size, shuffle=True, num_workers=2)
trainloader_svhn = torch.utils.data.DataLoader(svhn_train, batch_size=batch_size, shuffle=True, num_workers=2)
testloader_svhn = torch.utils.data.DataLoader(svhn_test, batch_size=batch_size, shuffle=True, num_workers=2)

net = resnet_orig.ResNet18(num_classes=10).cuda()
net = torch.nn.DataParallel(net)
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)


def base_train(epoch, net):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # 观察量
    acc_cifar10_all = 0
    mmc_cifar10_all = 0
    mmc_cifar100_all = 0
    mmc_svhn_all = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader_cifar10):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        # 正常样本损失
        loss = criterion(outputs, targets)
        # 梯度清零
        optimizer.zero_grad()
        # 损失回传
        loss.backward()
        # 更新参数
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc_cifar10_all = get_acc(net, testloader_cifar10)
    print(f"epoch[{epoch}/{epochs}] : acc_cifar10={acc_cifar10_all}")
    writer.add_scalar("acc_cifar10", acc_cifar10_all, epoch)
    if True and epoch > 50:
        mmc_cifar10_all = get_mmc(net, testloader_cifar10)
        mmc_cifar100_all = get_mmc(net, testloader_cifar100)
        mmc_svhn_all = get_mmc(net, testloader_svhn)
        print(
            f"epoch[{epoch}/{epochs}] : mmc_cifar10={mmc_cifar10_all},mmc_cifar100={mmc_cifar100_all},mmc_svhn={mmc_svhn_all}")
        writer.add_scalars("mmc",
                           {"mmc_cifar10": mmc_cifar10_all, "mmc_cifar100": mmc_cifar100_all, "mmc_svhn": mmc_svhn_all},
                           epoch)


# train
def train_baseline():
    '''
    苗代码训练resnet baseline
    :return:
    '''
    for epoch in range(0, epochs):
        # train_gen, dev_gen = mycifar_load(128, data_dir=opt.set_dir, key=opt.keylabel)
        lr = 1e-1
        # if epoch > 30:
        #     lr = 1e-2
        # if epoch > 60:
        #     lr = 1e-3
        # if epoch > 90:
        #     lr = 1e-4
        # if epoch >= 50:
        #     lr = 1e-2
        # if epoch >= 75:
        #     lr = 1e-3
        # if epoch >= 90:
        #     lr = 1e-4
        # if epoch >= 90:
        #     lr = 2e-2
        # if epoch >= 180:
        #     lr = 4e-3
        # if epoch >= 240:
        #     lr = 8e-4
        lr = 1e-1
        if epoch >= 60:
            lr = 2e-2
        if epoch >= 120:
            lr = 4e-3
        if epoch >= 160:
            lr = 8e-4
        adjust_learning_rate(optimizer, lr)
        base_train(epoch, net)
        torch.save(net.state_dict(), root_result + f"/baseline_resnet--epoch{epoch}.pth")


if __name__ == "__main__":
    pass
    train_baseline()  # 苗代码训练resnet18 baseline
