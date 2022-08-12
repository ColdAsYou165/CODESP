'''
gan的思想训练ae
'''
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import math
# import torch.backends.cudnn as cudnn
import sys
from model import *

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.00006, help="train loss 趋于不变，test loss趋于不变，说明学习遇到瓶颈，需要减小学习率或批量数目; ")
parser.add_argument("--optimizer", default="Adam", help="Adam SGD")
parser.add_argument("--begin_epoch", type=int, default=0, help="开始训练的轮数")
parser.add_argument("--epoch", type=int, default=300)
args = parser.parse_args()
os.makedirs("../weights_gan_ae", exist_ok=True)
writer = SummaryWriter()
writer.add_text("实验描述", f"正常训练ae,loss为分类器的分类损失,使用苗师兄的ae,lr={args.lr},optimizer={args.optimizer}")
torch.set_printoptions(profile="full")
# 数据集
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
num_classes = 10
batch_size = 128
epochs = 200
lr = args.lr
# [, 3, 32, 32]

trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=False, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=False, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# 模型
model_d = getResNet("resnet" + "18").cuda()
model_d = torch.nn.DataParallel(model_d)
state = torch.load("../betterweights/resnet18--epoch199_transform_train.pth")  # model acc epoch
model_d.load_state_dict(state["model"])

model_g = AutoEncoder().cuda()
model_g = torch.nn.DataParallel(model_g)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if args.begin_epoch == 0:
    model_g.apply(weights_init)
elif args.begin_epoch != 0:
    model_g.load_state_dict(torch.load(f"../weights_ae/ae_normal___epoch199.pth")["model"])
    # model_g.load_state_dict(torch.load(f"../weights_ae/ae_normalofMiaoshixiong___epoch{args.begin_epoch}.pth")["model"])

criterion = torch.nn.CrossEntropyLoss().cuda()  # 使得生成的图像更像两个类别
criterion_MSE = torch.nn.MSELoss().cuda()

if args.optimizer == "SGD":
    optimizer_g = torch.optim.SGD(model_g.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)  # 5的10的-4次方,0.0005
elif args.optimizer == "Adam":
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=lr)
scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=200)


def train_and_test():
    # 训练与测试
    for epoch in range(args.begin_epoch + 1, args.begin_epoch + epochs, 1):
        # 训练
        model_g.train()
        loss_train = 0
        for batch, (data, label) in enumerate(tqdm(trainloader)):
            optimizer_g.zero_grad()
            data = data.cuda()
            label = label.cuda()
            outputs = model_g(data)
            loss = criterion(pred, label)
            loss.backward()
            optimizer_g.step()
            loss_train += loss.item()
        loss_train = loss_train / len(trainloader)
        print(f"train epoch{epoch} loss={loss_train}")
        # 测试
        model_g.eval()
        loss_test = 0
        for batch, (data, label) in enumerate(tqdm(testloader)):
            with torch.no_grad():
                data = data.cuda()
                label = label.cuda()
                outputs = model_g(data)
                pred = model_d(outputs)
                loss = criterion(pred, label)
                loss_test += loss.item()
        loss_test = loss_test / len(testloader)
        scheduler_g.step()
        # 保存
        print(f"test epoch{epoch} loss={loss_test}")
        writer.add_scalars("loss", {"train": loss_train, "test": loss_test}, epoch)
        state_g = {"model": model_g.state_dict(), "loss": loss_test, "epoch": epoch}
        torch.save(state_g, f"../weights_ae/ae_normalofMiaoshixiong___epoch{epoch}.pth")


def showaeimg():
    model_g.eval()
    model_d.eval()
    for data, label in testloader:
        data = data.cuda()
        label = label.cuda()
        vir = model_g(data)
        # vir=torch.rand([4,3,32,32])
        pred = model_d(vir)
        print(pred.softmax(dim=1))
        grad = torchvision.utils.make_grid(vir)
        print(grad.shape)
        grad = grad / 2 + 0.5
        grad = grad.cpu().detach().numpy()
        grad = np.transpose(grad, [1, 2, 0])
        plt.imshow(grad)
        plt.show()


showaeimg()
