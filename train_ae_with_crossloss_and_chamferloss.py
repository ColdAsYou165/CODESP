''''
chamfer loss 和 crossentropyloss
2022年8月22日在等结果  weight_crossloss为1的话生成的是颜色块
2022年8月31日调整crossloss权重,观察下crossloss有没有降低下来,再看压制训练效果.
'''
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr_g", type=float, default=1e-4, help="model_g的lr")
parser.add_argument("--weight_crossloss", type=float, default=0.1, help="crossloss的权重")
parser.add_argument("--optimizer", default="Adam", help="Adam SGD")
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--gpus", default="0")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

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

# v2 2022年8月31日多实验几个参数
args_str = get_args(args)
results_root = "../results/train_ae_with_crossloss_and_chamferloss_v2" + f"/{args_str}"

os.makedirs(results_root, exist_ok=True)
results_pic_root = results_root + "/pic"  # 存图像
results_pth_root = results_root + "/pth"  # 存权重
os.makedirs(results_pic_root, exist_ok=True)
os.makedirs(results_pth_root, exist_ok=True)
writer = SummaryWriter()
writer.add_text("实验描述", f"chamferloss and wloss,{args}")

# 数据集

num_classes = 10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
trainset = torchvision.datasets.CIFAR10(root="../data/cifar10", train=True, download=False,
                                        transform=transform_only_tensor)
testset = torchvision.datasets.CIFAR10(root="../data/cifar10", train=False, download=False,
                                       transform=transform_only_tensor)
trainloader = DataLoader(trainset, args.batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(testset, args.batch_size, shuffle=True, num_workers=2)

# 模型

model_d = getResNet("resnet" + "18").cuda()
model_d = torch.nn.DataParallel(model_d)
state = torch.load("../betterweights/resnet18--transform_onlyToTensor--epoch199--acc095--loss017.pth")
model_d.load_state_dict(state["model"])

model_g = AutoEncoder_Miao().cuda()
model_g = torch.nn.DataParallel(model_g)
model_g.apply(weights_init)
state_g = torch.load("../betterweights/ae_miao_OnlyToTensor--sigmoid--epoch348--loss0.03.pth")
model_g.load_state_dict(state_g["model"])

# 优化器
criterion_cross = nn.CrossEntropyLoss().cuda()
criterion_chamfer = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
optimizer_g = torch.optim.Adam(model_g.parameters(), lr=args.lr_g, betas=(args.beta1, 0.999))

origin_data, origin_label = next(iter(testloader))
origin_data, origin_label = origin_data.cuda(), origin_label.cuda()
save_image(origin_data, results_pic_root + "/origin_data" + ".jpg")


# 训练
def ae(epoch):
    # 观察量
    loss_cross_all = 0
    loss_chamfer_all = 0
    for batch_idx, (data, label) in enumerate(trainloader):
        data = data.cuda()
        label = label.cuda()
        ## 生成虚假图像 并获得标签
        virtual_data = model_g.module.generate_virtual(data, set_encoded_detach=True)
        virtual_label = F.one_hot(label, num_classes) / 2
        index_0 = range(0, len(virtual_label), 2)
        index_1 = range(1, len(virtual_label), 2)
        virtual_label = (virtual_label[index_0] + virtual_label[index_1]).detach()

        ## 计算交叉熵损失
        pred_virtual = model_d(virtual_data)
        loss_cross = criterion_cross(pred_virtual, virtual_label)
        (loss_cross * args.weight_crossloss).backward(retain_graph=True)
        loss_cross_all += loss_cross.item()

        ## 计算chamfer loss
        data_concat = data.transpose(1, 3).transpose(1, 2)  # n h w c
        data_concat = data_concat.reshape(-1, data_concat.shape[1] * 2, data_concat.shape[2],
                                          data_concat.shape[3])  # n/2 2h w c
        data_concat = data_concat.reshape(data_concat.shape[0], -1, data_concat.shape[3])  # n/2 2h*w c
        data_concat = data_concat.cuda()
        virtual_data = virtual_data.transpose(1, 3).transpose(1, 2)  # n/2 h w c
        virtual_data = virtual_data.reshape(virtual_data.shape[0], -1, virtual_data.shape[3])
        dist1, dist2, _, _ = criterion_chamfer(data_concat, virtual_data)  # 苗师兄是这么写的,(原始头像,生成的图像)
        loss_chamfer = (torch.mean(dist1)) + (torch.mean(dist2))
        loss_chamfer.backward()
        loss_chamfer_all += loss_chamfer.item()
        optimizer_g.step()
    # 观察量
    loss_cross_all /= len(trainloader)
    loss_chamfer_all /= len(trainloader)
    print(
        f"[{epoch}/{args.epochs}]:loss_chamfer_all={loss_chamfer_all},loss_cross_all={loss_cross_all}")
    writer.add_scalar("chamfer_loss", loss_chamfer, epoch)
    writer.add_scalar("crossloss", loss_cross_all, epoch)

    # 每个epoch生成并保存一张虚假图片
    if True:
        virtual_data = model_g.module.generate_virtual(origin_data)
        save_image(virtual_data,
                   results_pic_root + f"/virpic_--epoch{epoch}--chamferloss{loss_chamfer_all:.3f}--crossloss{loss_cross_all:.3f}.jpg")

    # 保存模型权重
    if True and (epoch + 1) % 100 == 0:
        state = {"model": model_g.state_dict(), "loss": loss_chamfer_all}
        torch.save(state, results_pth_root + f"/model_g_chamfer_and_crossloss--epoch{epoch}.pth")


for epoch in range(args.epochs):
    ae(epoch)
