'''
检测下 3loss约束下的ae 生成虚假样本 能不能像两个类.
'''
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr_g", type=float, default=0.0002, help="model_g的lr")
parser.add_argument('--lr_dis', type=float, default=0.0002, help='wgan discrinator lr, default=0.0002')
parser.add_argument('--lr_scale', type=float, default=1e4, help='wgan discrinator lr, default=0.0002')
parser.add_argument("--optimizer", default="Adam", help="Adam SGD")
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--gpus", default="0")
parser.add_argument("--batch_size", type=int, default=128)
# parser.add_argument("--loss", default="crossentropyloss", help="l1loss crossentropyloss")
# crossentropyloss 和 mseloss 联合训练 ae
# wgan的 discriminator
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--w_loss_weight', type=float, default=1e-5, help='wloss上加的权重,苗师兄wgan是1e-5')
parser.add_argument('--cross_loss_weight', type=float, default=1e-5, help='cross_loss上加的权重')
parser.add_argument("--set_sigmoid", type=str, default='False', help="wgan的dis是否需要sigmoid,不能要sigmoid!")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

if args.set_sigmoid == "True":
    args.set_sigmoid = True
elif args.set_sigmoid == "False":
    args.set_sigmoid = False
else:
    print(f"--set_sigmoid传参为{args.set_sigmoid},错误!")
    exit()

if args.lr_dis == 0 or args.lr_dis < 0:
    args.lr_dis = args.lr_g / args.lr_scale
print(str(args))

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import chamfer3D.dist_chamfer_3D
from tqdm import tqdm
# import torch.backends.cudnn as cudnn
import sys
from model import *
from utils import *

# v0貌似忘了保存权重 v1保存了权重
results_root = "../results/train_ae_with_3loss_chamfer_w_cross"
os.makedirs(results_root, exist_ok=True)
results_root = "../results/train_ae_with_3loss_chamfer_w_cross" + f"/{args}"
os.makedirs(results_root, exist_ok=True)
file = open(results_root + "/args.txt", "w")
file.write(f"{args}")
file.close()
results_pic_root = results_root + "/pic"
results_pth_root = results_root + "/pth"
os.makedirs(results_pic_root, exist_ok=True)
os.makedirs(results_pth_root, exist_ok=True)

num_classes = 10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
trainset = torchvision.datasets.CIFAR10(root="../data/cifar10", train=True, download=False, transform=transform_only_tensor)
testset = torchvision.datasets.CIFAR10(root="../data/cifar10", train=False, download=False, transform=transform_only_tensor)
trainloader = DataLoader(trainset, args.batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(testset, args.batch_size, shuffle=True, num_workers=2)

# 模型
discriminator = Discriminator_WGAN_miao_cifar10(set_sigmoid=args.set_sigmoid).cuda()  # set_sigmoid=False
# discriminator = simple_discriminator().cuda()
discriminator = torch.nn.DataParallel(discriminator)
discriminator.apply(weights_init)

model_g = AutoEncoder_Miao().cuda()
model_g = torch.nn.DataParallel(model_g)

model_g.apply(weights_init)
state_g = torch.load("../betterweights/ae_miao_OnlyToTensor--sigmoid--epoch348--loss0.03.pth")
model_g.load_state_dict(state_g["model"])

model_d = getResNet("resnet" + "18").cuda()
model_d = torch.nn.DataParallel(model_d)
state = torch.load("../betterweights/resnet18--transform_onlyToTensor--epoch199--acc095--loss017.pth")
model_d.load_state_dict(state["model"])

# 优化器
criterion_cross = nn.CrossEntropyLoss().cuda()
criterion_bce = nn.BCEWithLogitsLoss().cuda()
chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()

optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=args.lr_dis, betas=(args.beta1, 0.999))
optimizer_g = torch.optim.Adam(model_g.parameters(), lr=args.lr_g, betas=(args.beta1, 0.999))

origin_data, origin_label = next(iter(testloader))
origin_data, origin_label = origin_data.cuda(), origin_label.cuda()
save_image(origin_data, results_pic_root + "/origin_data" + ".jpg")

for batch_idx, (data, label) in enumerate(testloader):
    data = data.cuda()
    label = label.cuda()
