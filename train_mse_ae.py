'''
先训练一个能正常重构图像的ae
使用mse
'''

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0006)
parser.add_argument("--optimizer", default="Adam", help="Adam SGD")
parser.add_argument("--begin_epoch", type=int, default=0)
parser.add_argument("--epoch", type=int, default=300)
parser.add_argument("--gpus", default="0")
args = parser.parse_args()
import os

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
from utils import *
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# import torch.backends.cudnn as cudnn
import sys
from model import *

root_weights = "../weights_bce_ae"
os.makedirs(root_weights, exist_ok=True)
writer = SummaryWriter()
writer.add_text("实验描述", f"使用bce训练自己的ae,lr={args.lr},optimizer={args.optimizer}")
print(str(args))
torch.set_printoptions(profile="full")
# 数据集
num_classes = 10
batch_size = 128
epochs = 200
lr = args.lr
# [, 3, 32, 32]

trainset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=True,
                                        download=False, transform=transform_only_tensor)
testset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=False,
                                       download=False, transform=transform_only_tensor)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# 模型

model_g = simple_ae().cuda()
model_g = torch.nn.DataParallel(model_g)

# 只有从0开始训练的时候,才需要初始化权重
if args.begin_epoch == 0:
    model_g.apply(weights_init)
else:
    model_g.load_state_dict(torch.load(f"../weights_ae/ae__epoch{args.begin_epoch}.pth")["model"])
    print(f"ae加载路径:../weights_ae/ae__epoch{args.begin_epoch}.pth")
if args.optimizer == "SGD":
    optimizer_g = torch.optim.SGD(model_g.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)  # 5的10的-4次方,0.0005
elif args.optimizer == "Adam":
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=lr)
scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=200)
# criterion = torch.nn.MSELoss().cuda()  # 使得生成的图像更像两个类别
criterion = torch.nn.BCEWithLogitsLoss().cuda()  # github上用的这个
## mse训练和测试
for epoch in range(args.begin_epoch, args.epoch, 1):
    model_g.train()
    loss_train, loss_test = 0, 0
    # acc_train,acc_test=0,0
    for batch, (data, label) in enumerate(trainloader):
        optimizer_g.zero_grad()
        data = data.cuda()
        label = label.cuda()
        outs = model_g(data)
        loss = criterion(outs, data)
        loss.backward()
        optimizer_g.step()
        loss_train += loss.item()
    loss_train /= len(trainloader)
    for batch, (data, label) in enumerate(testloader):
        model_g.eval()
        with torch.no_grad():
            data = data.cuda()
            label = label.cuda()
            outs = model_g(data)
            loss = criterion(outs, data)
            loss_test += loss.item()
    loss_test /= len(testloader)
    writer.add_scalars("mse_ae_loss", {"train": loss_train, "test": loss_test}, epoch)
    print(f"epoch{epoch},train_loss={loss_train},test_loss={loss_test}")
    state = {"model": model_g.state_dict(), "loss": loss_test, "epoch": epoch}
    torch.save(state, root_weights + f"/ae__epoch{epoch}__lr{lr}__mseloss{loss_test}.pth")
    scheduler_g.step()
