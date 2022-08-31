''''
wgan+chamfer 的结构 但是鉴别器换成crossentropy十分类器
'''
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr_g", type=float, default=0.0002, help="model_g的lr")
parser.add_argument('--lr_dis', type=float, default=0.0002, help='wgan discrinator lr, default=0.0002')
parser.add_argument('--lr_scale', type=float, default=1e4, help='wgan discrinator lr, default=0.0002')
parser.add_argument("--optimizer", default="Adam", help="Adam SGD")
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--gpus", default="0")
parser.add_argument("--batch_size", type=int, default=128)
# wgan的 discriminator
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--w_loss_weight', type=float, default=1e-5, help='wloss上加的权重,苗师兄wgan是1e-5')
parser.add_argument('--real_cross_d_weight', type=float, default=1, help='real样本训练鉴别器时候,乘以一个权重,缩小正常值')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

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
results_root = "../results/train_ae_chamfer_and_crossgan"
os.makedirs(results_root, exist_ok=True)
results_root = results_root + f"/{args}"
os.makedirs(results_root, exist_ok=True)
file = open(results_root + "/args.txt", "w")
file.write(f"{args}")
file.close()
results_pic_root = results_root + "/pic"
results_pth_root = results_root + "/pth"
os.makedirs(results_pic_root, exist_ok=True)
os.makedirs(results_pth_root, exist_ok=True)
writer = SummaryWriter()
writer.add_text("实验描述", f"wgan+chamfer 的结构 但是鉴别器换成crossentropy十分类器,{args}")

# 数据集

num_classes = 10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
trainset = torchvision.datasets.CIFAR10(root="../data/cifar10", train=True, download=False, transform=transform_only_tensor)
testset = torchvision.datasets.CIFAR10(root="../data/cifar10", train=False, download=False, transform=transform_only_tensor)
trainloader = DataLoader(trainset, args.batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(testset, args.batch_size, shuffle=True, num_workers=2)

# 模型


model_g = AutoEncoder_Miao().cuda()
model_g = torch.nn.DataParallel(model_g)

model_g.apply(weights_init)
state_g = torch.load("../betterweights/ae_miao_OnlyToTensor--sigmoid--epoch348--loss0.03.pth")
model_g.load_state_dict(state_g["model"])

model_d = getResNet("resnet" + "18").cuda()  # 后面也没有接softmax
model_d = torch.nn.DataParallel(model_d)
model_d.apply(weights_init)
# 优化器
# criterion_bce = nn.BCEWithLogitsLoss().cuda()
criterion_cross = nn.CrossEntropyLoss().cuda()
chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()

optimizer_d = torch.optim.Adam(model_d.parameters(), lr=args.lr_dis, betas=(args.beta1, 0.999))
optimizer_g = torch.optim.Adam(model_g.parameters(), lr=args.lr_g, betas=(args.beta1, 0.999))

origin_data, origin_label = next(iter(testloader))
origin_data, origin_label = origin_data.cuda(), origin_label.cuda()
save_image(origin_data, results_pic_root + "/origin_data" + ".jpg")


# 训练
def ae(epoch):
    one = torch.FloatTensor([1])
    mone = one * -1
    one, mone = one.cuda(), mone.cuda()
    # 观察量 cross_real , cross_fake         cross_fake_g,chamferloss
    cross_real_all = 0
    cross_fake_all = 0
    cross_fake_g_all = 0
    chamferloss_all = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # 更新鉴别器
        inputs = inputs.cuda()
        targets = targets.cuda()
        optimizer_d.zero_grad()
        real_cpu = inputs.cuda()
        output = model_d(real_cpu)
        d_loss_real = criterion_cross(output, targets)
        (d_loss_real * args.real_cross_d_weight).backward()  # real为0
        cross_real_all += d_loss_real.item()
        virtual_data = model_g.module.generate_virtual(inputs)  # 虚假图像
        virtual_label = F.one_hot(targets, num_classes) / 2
        index_0 = range(0, len(virtual_label), 2)
        index_1 = range(1, len(virtual_label), 2)
        virtual_label = virtual_label[index_0] + virtual_label[index_1]
        virtual_label = virtual_label.detach()  # 虚假图像的标签
        # 我们应该是希望鉴别器对虚假图像输出均匀置信度
        virtual_target_uniform = (torch.ones([len(virtual_data), 10]) * 0.1).cuda().detach()
        output = model_d(virtual_data.detach())  # 注意detach
        d_loss_fake = criterion_cross(output, virtual_target_uniform)
        d_loss_fake.backward()  # fake为1
        cross_fake_all += d_loss_fake.item()
        # Update D
        optimizer_d.step()

        # 更新生成器
        model_g.zero_grad()
        # cross loss  像两个类
        output = model_d(virtual_data)
        cross_fake_g = criterion_cross(output, virtual_label)
        g_loss = args.w_loss_weight * cross_fake_g
        g_loss.backward(retain_graph=True)
        cross_fake_g_all += cross_fake_g.item()

        ## 计算chamfer loss
        inputs_concat = inputs.transpose(1, 3).transpose(1, 2)  # n h w c
        inputs_concat = inputs_concat.reshape(-1, inputs_concat.shape[1] * 2, inputs_concat.shape[2], inputs_concat.shape[3])  # n/2 2h w c
        inputs_concat = inputs_concat.reshape(inputs_concat.shape[0], -1, inputs_concat.shape[3])  # n/2 2h*w c
        inputs_concat = inputs_concat.cuda()
        virtual_data = virtual_data.transpose(1, 3).transpose(1, 2)  # n/2 h w c
        virtual_data = virtual_data.reshape(virtual_data.shape[0], -1, virtual_data.shape[3])
        dist1, dist2, _, _ = chamLoss(inputs_concat, virtual_data)  # 苗师兄是这么写的,(原始头像,生成的图像)
        # dist1, dist2, _, _ = chamLoss(virtual_data, inputs_concat)  # 不看文档,直接翻转一下会怎么样
        loss_chamfer = (torch.mean(dist1)) + (torch.mean(dist2))
        loss_chamfer.backward()  # 带one会报错,莫名其妙
        chamferloss_all += loss_chamfer.item()

        optimizer_g.step()
    # 观察量
    cross_real_all /= len(trainloader)
    cross_fake_all /= len(trainloader)
    cross_fake_g_all /= len(trainloader)
    chamferloss_all /= len(trainloader)
    print(
        f"[{epoch}/{args.epochs}] : cross_real_all={cross_real_all:3f},cross_fake_all={cross_fake_all:.3f},cross_fake_g_all={cross_fake_g_all:.3f},chamferloss_all={chamferloss_all:.3f}")
    writer.add_scalar("chamfer_loss", chamferloss_all, epoch)
    writer.add_scalar("cross_fake_g_all", cross_fake_g_all, epoch)
    # 每个epoch生成并保存一张虚假图片
    virtual_data = model_g.module.generate_virtual(origin_data)
    save_image(virtual_data, results_pic_root + f"/virpic_--epoch{epoch}--chamferloss{chamferloss_all:.3f}--cross{cross_fake_g_all:.3f}.jpg")

    # 保存模型权重
    if True and (epoch + 1) % 100 == 0:
        state = {"model": model_g.state_dict(), "loss": cross_fake_g_all}
        torch.save(state, results_pth_root + f"model_ae_chamfer_and_wcross--epoch{epoch}--cross{cross_fake_g_all:.2f}.pth")


for epoch in range(args.epochs):
    ae(epoch)
