'''
查看ae重构效果
'''

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.00006)
parser.add_argument("--optimizer", default="Adam", help="Adam SGD")
parser.add_argument("--begin_epoch", type=int, default=0)
parser.add_argument("--epoch", type=int, default=300)
parser.add_argument("--gpus", default="0,1")
args = parser.parse_args()
import os

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
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
from utils import *

root_weights = "../weights_mse_ae"
os.makedirs(root_weights, exist_ok=True)
writer = SummaryWriter()
writer.add_text("实验描述", f"使用mse训练苗师兄的ae,lr={args.lr},optimizer={args.optimizer}")
torch.set_printoptions(profile="full")
# 数据集

num_classes = 10
batch_size = 5
epochs = 200
lr = args.lr
# [, 3, 32, 32]

trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=False, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=False, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# 模型
model_d = getResNet("resnet" + "18").cuda()
model_d = torch.nn.DataParallel(model_d)
state = torch.load(
    f"../betterweights/resnet18--transform_onlyToTensor--epoch199--acc095--loss017.pth")  # model acc epoch
model_d.load_state_dict(state["model"])

model_g = AutoEncoder().cuda()
# model_g = AutoEncoder().cuda()
model_g = torch.nn.DataParallel(model_g)


def show_mse_ae():
    '''
    观察mse的ae的重构图像的能力
    :return:
    '''
    # for lr in [0.1,0.001,0.0001,0.00001,0.000001,0.000001]:
    model_g.eval()
    state = torch.load(f"../betterweights/ae__miaoae__epoch299__lr0.001.pth")
    # state = torch.load(f"../weights_mse_ae/ae__epoch299__lr{lr}.pth")
    model_g.load_state_dict(state["model"])
    for data, label in trainloader:
        data = data.cuda()
        label = label.cuda()
        outs = model_g(data)
        pred = model_d(outs).softmax(dim=1)
        # print(torch.topk(pred, k=2, dim=1)[0])
        outs = torch.concat([data, outs], dim=0)
        outs = torchvision.utils.make_grid(outs)
        outs = transform_convert(outs, transform_train)
        outs = outs.cpu().detach().numpy()
        outs = np.transpose(outs, [1, 2, 0])
        plt.imshow((outs * 255).astype(np.uint8))
        print("a")
        plt.show()


# show_mse_ae()
state = torch.load(f"../betterweights/ae_mine_goodreconstruct--OnlyToTensor.pth")
model_g.load_state_dict(state["model"], strict=False)



def show_2classes_ae_mine(path, loader, transforms):
    state = torch.load(path)
    model_g = Autoencoder().cuda()
    # model_g=torch.nn.DataParallel(model_g)

    model_g.load_state_dict(state, strict=False)
    i = 0
    for data, label in loader:
        data, label = data.cuda(), label.cuda()
        outs = model_g(data)
        pred = model_d(outs).softmax(dim=1)
        print(torch.topk(pred, 2, 1)[0])
        imgs = torch.concat([data, outs], dim=0)
        imgs = torchvision.utils.make_grid(imgs)
        imgs = transform_convert(imgs, transforms)
        imgs = imgs.cpu().detach().numpy()
        imgs = np.transpose(imgs, [1, 2, 0])
        plt.imshow((imgs * 255).astype(np.uint8))
        plt.show()
        i += 1
        if (i >= 5):
            break


show_2classes_ae_mine(path="./weights/autoencoder.pkl", loader=trainloader,
                      transforms=transform_train)
show_2classes_ae_mine(path="./weights/autoencoder.pkl", loader=testloader,
                      transforms=transform_test)
