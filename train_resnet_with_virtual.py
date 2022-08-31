'''
描述: 使用训练好的ae压制训练
---
说明:
    老师说,严格按照苗师兄代码来.
    epochs200 lr0.1以及后续调整策略 优化器
'''
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="这个就不调整了,就默认200")
parser.add_argument("--gpus", default="0")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--use_scheduler", type=str, default="False", help="使用virtual训练discriminator的时候使用动态学习scheduler")
parser.add_argument("--loss_virtual_weight", type=float, default=1, help="压制训练时候,loss_virtual的权重")
# 压制训练时生成多少倍数的虚假图像
parser.add_argument("--virtual_scale", type=int, default=2, help="要多少个0.5倍正常样本数量的 virtual example")
# v5版本专用 选择ae参数
parser.add_argument("--crossweight", type=int, default=3, help="3,4,5对应crossloss权重1e -3-4-5")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

if args.use_scheduler == "False":
    args.use_scheduler = False
elif args.use_scheduler == "True":
    args.use_scheduler = True
else:
    print("use_scheduler参数错误")
    exit()
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

print(str(args))
writer = SummaryWriter()
writer.add_text("args", str(args))
# v0是chamferloss和wloss训练的ae的压制训练
# v1用了苗师兄sgd优化器
# v2用了 3loss训练的
# v4 transform 按照transform_train_cifar_miao
# v5 选用的ae是使用3loss训练出的,观察该ae对resnet18的压制训练效果
results_root = "../results/train_discriminator_by_virtual_v5"
os.makedirs(results_root, exist_ok=True)
str_args = get_args(args)
results_root = results_root + f"/{str_args}"
os.makedirs(results_root, exist_ok=True)
results_root_pic = results_root + "/pic"
os.makedirs(results_root_pic, exist_ok=True)
results_root_pth = results_root + "/pth"
os.makedirs(results_root_pth, exist_ok=True)


def get_virtual_example(model_g=None, data=None, scale=2):
    '''
    压制训练时,通过gan生成virtual example,virtual label为0.1,0.1,0.1....
    :return:virtual_data,virtual_label
    '''
    list = []
    for i in range(scale):
        data = data[torch.randperm(len(data))].detach()
        data_virtual = model_g.module.generate_virtual(data).detach()
        list.append(data_virtual)
        # 压制训练时候,虚假样本的label应该都是0.1,我设置错了.
    data_virtual = torch.concat(list, dim=0).detach()
    label_virtual = (torch.ones([len(data_virtual), 10]) * 0.1).cuda().detach()
    return data_virtual, label_virtual


# 模型
model_d = getResNet("resnet" + "18").cuda()
model_d = torch.nn.DataParallel(model_d)
# 不应该拿训练好的模型来训练,应该是重头训练.
# state_d = torch.load("../betterweights/resnet18--transform_onlyToTensor--epoch199--acc095--loss017.pth")
# model_d.load_state_dict(state_d["model"])
model_d.apply(weights_init)
model_g = AutoEncoder_Miao().cuda()
model_g = torch.nn.DataParallel(model_g)
# 原来的state_g
# state_g = to|rch.load("../betterweights/pthmodel_chamfer_and_wloss--epoch299--crossweight0.001.pth")  # chamferloss
# v5的state_g
state_g = torch.load(
    f"../betterweights/ae_trained_by3loss_v0/ae_chamfer_and_wloss_and_crossloss--crossweight1e-{args.crossweight}_epoch799.pth")
# state_g = torch.load("../betterweights/ae_miao__epoch234__w_weight0.001__entroloss1.97.pth")  # 引入wgan训练的ae
model_g.load_state_dict(state_g["model"])
model_g.eval()
# optimizer_d = torch.optim.Adam(model_d.parameters(), lr=args.lr)#

# 数据集
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

trainloader_cifar10 = torch.utils.data.DataLoader(cifar10_train, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=2)
testloader_cifar10 = torch.utils.data.DataLoader(cifar10_test, batch_size=args.batch_size, shuffle=True, num_workers=2)
trainloader_cifar100 = torch.utils.data.DataLoader(cifar100_train, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=2)
testloader_cifar100 = torch.utils.data.DataLoader(cifar100_test, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=2)
trainloader_svhn = torch.utils.data.DataLoader(svhn_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader_svhn = torch.utils.data.DataLoader(svhn_test, batch_size=args.batch_size, shuffle=True, num_workers=2)

# 优化器
criterion = torch.nn.CrossEntropyLoss().cuda()
lr = 1e-1
optimizer_d = torch.optim.SGD(model_d.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)  # 苗lr0.1
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=200)
# 记录起初的mmc
mmc_cifar100 = get_mmc(model_d, testloader_cifar100)
mmc_svhn = get_mmc(model_d, testloader_svhn)
mmc_cifar10 = get_mmc(model_d, testloader_cifar10)
acc = get_acc(model_d, testloader_cifar10)
writer.add_scalars("loss", {"mmc_cifar100": mmc_cifar100, "mmc_svhn": mmc_svhn, "mmc_cifar10": mmc_cifar10}, 0)
writer.add_scalar("cifar10_acc", acc, 0)
print(f"压制训练前:acc:{acc},mmc_cifar100={mmc_cifar100},mmc_svhn={mmc_svhn},mmc_cifar10={mmc_cifar10}")

# 压制训练
acc_std = 0.80
num_classes = 10
for epoch in range(args.epochs):
    # lr 2e-2 提高的多 但是只能到0.86
    # 苗调整学习率
    if epoch >= 60:
        lr = 2e-2
    if epoch >= 120:
        lr = 4e-3
    if epoch >= 160:
        lr = 8e-4
    adjust_learning_rate(optimizer_d, lr)

    # 训练
    model_d.train()
    loss_train_containv = 0
    for batch_idx, (data, label) in enumerate(trainloader_cifar10):
        data = data.cuda()
        label = label.cuda()
        data_normal = data.detach()
        label_normal = F.one_hot(label, num_classes).detach().float()
        # data_virtual = model_g.module.generate_virtual(data).detach()
        # 压制训练时候,虚假样本的label应该都是0.1,我设置错了.
        # label_virtual = (torch.ones([int(len(label) / 2), 10]) * 0.1).cuda().detach()
        data_virtual, label_virtual = get_virtual_example(model_g=model_g, data=data, scale=args.virtual_scale)
        pred_normal = model_d(data_normal)
        loss_normal = criterion(pred_normal, label_normal)
        pred_virtual = model_d(data_virtual)
        loss_virtual = criterion(pred_virtual, label_virtual)
        loss = (loss_virtual + loss_normal).mean()
        optimizer_d.zero_grad()
        loss_normal.backward()
        (args.loss_virtual_weight * loss_virtual).backward()
        optimizer_d.step()

        if args.use_scheduler:
            scheduler.step()
        loss_train_containv += loss.item()
        # 没啥用就是看一下ae生成看来咋样的虚假图像用于压制训练
        if batch_idx == 0 and epoch % 40 == 0:
            pic = torch.concat([data, data_virtual], dim=0)
            save_image(pic, results_root_pic + f"/virtualpic--epoch{epoch}.jpg")

    loss_train_containv /= len(trainloader_cifar10)
    # 测试
    mmc_cifar100 = get_mmc(model_d, testloader_cifar100)
    mmc_svhn = get_mmc(model_d, testloader_svhn)
    mmc_cifar10 = get_mmc(model_d, testloader_cifar10)
    acc = get_acc(model_d, testloader_cifar10)
    if acc > acc_std:
        acc_std = acc
        state_d = {"model": model_d.state_dict()}
        torch.save(state_d,
                   f"{results_root}/resnet18_yazhixunlian__acc{acc:.2f}__cimmc{mmc_cifar100:.2f}__svhnmmc{mmc_svhn}.pth")
    writer.add_scalars("loss", {"mmc_cifar10": mmc_cifar10, "mmc_cifar100": mmc_cifar100,
                                "mmc_svhn": mmc_svhn}, epoch + 1)
    writer.add_scalar("cifar10_acc", acc, epoch + 1)
    writer.add_scalar("train_containv", loss_train_containv, epoch + 1)

    print(f"epoch[{epoch}/{args.epochs}] : cifar10_test_acc={acc} , ", "mmc_test_cifar10=", mmc_cifar10)
    print("loss_train_containv=", loss_train_containv, "mmc_cifar100=", mmc_cifar100, " , mmc_svhn=", mmc_svhn)
    print("-" * 40)
