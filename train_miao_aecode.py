'''
训练苗师兄的ae
'''
import argparse
import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
from model import *
from utils import *
from torch.utils.tensorboard import SummaryWriter

results_root = "../results/miao_ae_onlytensor_bootomsigmoid"
os.makedirs(results_root, exist_ok=True)
writer = SummaryWriter()
'''init setting'''
parser = argparse.ArgumentParser(description='Autoencoder_train: Inference Parameters')
parser.add_argument('--epoch',
                    type=int,
                    default=2000,
                    help='training epoch setting')
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.00005,
                    help='learning rate setting')
parser.add_argument('--save_weight_dir',
                    default=results_root,
                    help='Path to folder of saving weight')
parser.add_argument('--load_weight_dir',
                    default=results_root,
                    help='Path to folder of saving weight')
parser.add_argument('--save_loss_figure_dir',
                    default=f'{results_root}/loss_figure.pickle',
                    help='Path to folder of saving loss figure, , if you dont rename it, it will combine current and previous training result')
parser.add_argument('--gpuid',
                    default=0,
                    type=int,
                    help='GPU device ids (CUDA_VISIBLE_DEVICES)')
parser.add_argument('--dataset', type=str, default='cifar10')

'''gobal setting'''
# global args
args = parser.parse_args()
torch.manual_seed(0)

transform_train = transforms.Compose([

    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=False, transform=transform_only_tensor)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_only_tensor)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

'''training setting'''
EPOCH = args.epoch
loss_iter = 0

'''set the training gpu'''
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

'''init model'''
model = AutoEncoder_Miao().cuda()
model = torch.nn.DataParallel(model)

'''opt setting'''
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # optimize all cnn parameters
loss_func = nn.L1Loss().cuda()

'''golder for saving  weight'''
save_path = args.save_weight_dir


def train_miao_ae():
    '''
    训练苗师兄的ae
    :return:
    '''
    # 开始训练
    for epoch in range(EPOCH):
        loss_train = 0
        loss_test = 0
        model.train()
        for step, (data, label) in enumerate(trainloader):
            data = data.cuda()
            label = label.cuda()
            outs = model(data)
            loss = loss_func(outs, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        loss_train /= len(trainloader)
        model.eval()
        for step, (data, label) in enumerate(testloader):
            with torch.no_grad():
                data = data.cuda()
                label = label.cuda()
                outs = model(data)
                loss = loss_func(outs, data)
                loss_test += loss.item()
        loss_test /= len(testloader)
        writer.add_scalars("aemiao_norm048loss", {"train": loss_train, "test": loss_test}, epoch)
        print(f"loss_train={loss_train},loss_test={loss_test}")
        state = {"model": model.state_dict(), "loss": loss_test, "epoch": epoch}
        torch.save(state, results_root + f"/miao_ae_norm049--epoch{epoch}.pth")


def observe_reconstruct_img():
    '''
    观察重构效果
    :return:
    '''
    # 选择模型 mine还是miao
    model = "miao"
    if model == "miao":
        print(model)
        model = torch.nn.DataParallel(AutoEncoder_Miao().cuda())
        state = torch.load(results_root+f"/miao_ae_norm049--epoch348.pth")
        # state = torch.load("../betterweights/ae_miao_norm049--epoch1756.pth")
        # state = torch.load("../betterweights/ae_miao_OnlyToTensor--sigmoid--epoch63--loss005.pth")
        print("loss", state["loss"])
    elif model == "mine":
        print(model)
        model = torch.nn.DataParallel(AutoEncoder().cuda())
        state = torch.load("../betterweights/ae_mine_goodreconstruct--OnlyToTensor.pth")
        print("loss=", state["loss"])
    model.load_state_dict(state["model"])
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_only_tensor)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)
    criterion = torch.nn.MSELoss()
    for batch, (data, label) in enumerate(testloader):
        data = data.cuda()
        label = label.cuda()
        outs = model(data)
        print(outs[0])
        print(outs.max(), outs.min())
        loss = criterion(outs, data)
        imgs = torch.concat([data, outs], dim=0)
        imgs = torchvision.utils.make_grid(imgs)
        # imgs=transform_convert(imgs,transform_tensor_norm)
        imgs = imgs.cpu().detach().numpy()
        imgs = np.transpose(imgs, [1, 2, 0])
        print(f"loss:{loss}")
        plt.figure()
        plt.imshow(imgs)
        plt.show()


if __name__ == "__main__":
    # 训练常规ae
    # train_miao_ae()
    # 查看重构效果
    # observe_reconstruct_img()
    pass