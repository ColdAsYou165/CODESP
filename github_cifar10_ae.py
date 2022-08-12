'''
从自己model出发,训练github那个model
'''
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

os.makedirs("../weights_github_mymodel", exist_ok=True)
batch_size = 8


trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=False, transform=transform_tensor_norm)
testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=False, transform=transform_tensor_norm)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
model = AutoEncoder().cuda()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())


def train_and_test():
    '''
    训练和测试
    :return:
    '''
    for epoch in range(100):
        loss_train = 0
        loss_test = 0
        model.train()
        for step, (data, label) in enumerate(tqdm(trainloader)):
            data = data.cuda()
            label = label.cuda()
            outs = model(data)
            loss = criterion(outs, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        loss_train /= len(trainloader)

        model.eval()
        with torch.no_grad():
            for step, (data, label) in enumerate(tqdm(testloader)):
                data = data.cuda()
                label = label.cuda()
                outs = model(data)
                loss = criterion(outs, data)
                loss_test += loss.item()
        loss_test /= len(testloader)
        print(f"epoch{epoch}:loss_train]{loss_train},loss_test={loss_test}")
        state = {"model": model.state_dict(), "loss": loss_test, "epoch": epoch}
        torch.save(state, f"../weights_github_mymodel/mymodel__epoch{epoch}__transform_tensor_norm.pth")


def show_2classes_ae_mine(model, path, loader, transforms):
    '''
    观察模型重构能力
    :param model:
    :param path:
    :param loader:
    :param transforms:
    :return:
    '''
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=False, transform=transforms)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=False, transform=transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)
    model = model.cuda()
    state = torch.load(path)
    # model_g=torch.nn.DataParallel(model_g)

    model.load_state_dict(state["model"])
    model=torch.nn.DataParallel(model)

    i = 0
    model.eval()
    for data, label in trainloader:
        data, label = data.cuda(), label.cuda()
        outs = model(data)
        print(torch.min(outs),torch.max(outs))
        imgs = torch.concat([data, outs], dim=0)
        # imgs=outs
        imgs = torchvision.utils.make_grid(imgs)
        imgs=transform_convert(imgs,transforms)
        imgs = imgs.cpu().detach().numpy()
        imgs = np.transpose(imgs, [1, 2, 0])
        # plt.imshow((imgs*255).astype(np.uint))
        plt.imshow(imgs)
        plt.show()
        i += 1
        if (i >= 5):

            # state = {"model":model.state_dict(),"epoch":30,"loss":0.56}
            # torch.save(state,"../betterweights/ae_mine_goodreconstruct--OnlyToTensor.pth")
            break


if __name__ == "__main__":
    # train_and_test()
    epoch=3
    model=nn.DataParallel(AutoEncoder().cuda())
    show_2classes_ae_mine(model=model,path=f"../betterweights/ae_mine_goodreconstruct--OnlyToTensor.pth", loader=trainloader,
                          transforms=transform_only_tensor)
