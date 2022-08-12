import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch

torch.set_printoptions(profile="full", precision=4, sci_mode=False)
print(f"可用gpu数量{torch.cuda.device_count()}")
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
# import torch.backends.cudnn as cudnn
import sys
from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.05)
parser.add_argument("--optimizer", default="Adam", help="Adam SGD")
args = parser.parse_args()
os.makedirs("../weights_ae", exist_ok=True)
writer = SummaryWriter()
writer.add_text("实验描述", f"训练ae,lr={args.lr},optimizer={args.optimizer}")
torch.set_printoptions(profile="full")
# 数据集

num_classes = 10
batch_size = 4
epochs = 200
lr = args.lr
# [, 3, 32, 32]

trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=False, transform=transform_only_tensor)
testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=False, transform=transform_only_tensor)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

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
model_g = torch.nn.DataParallel(model_g)

criterion = torch.nn.CrossEntropyLoss().cuda()  # 使得生成的图像更像两个类别

if args.optimizer == "SGD":
    optimizer_g = torch.optim.SGD(model_g.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)  # 5的10的-4次方,0.0005
elif args.optimizer == "Adam":
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=lr)
scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=200)


# custom weights_resnet initialization called on netG and netD

def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    img = img_tensor.cpu().numpy() * 255
    img = np.transpose(img, [1, 2, 0])
    print(img.shape)
    plt.imshow(img)
    plt.show()


# 查看图像
def showimg(img):
    img = img * 0.5 + 0.5
    print(img.shape)
    img = img.cpu().detach().numpy()
    print(img.shape)
    img = np.transpose(img, [1, 2, 0])
    plt.imshow(img)
    plt.show()


'''dataiter=iter(testloader)
datas,labels=next(dataiter)
datas=datas.cuda()
labels=labels.cuda()
'''
'''print(datas.shape)
imgs=torchvision.utils.make_grid(datas)
# transform_convert(imgs,transform_test)
showimg(imgs)'''
'''
virtual=model_g.module.generate_virtual(datas)
pred=model_d(virtual).softmax(dim=1)
print(virtual.shape)
for i in range(batch_size):
    showimg(virtual[i])'''


def guanchachongou():
    '''
    查看这么多权重的重构效果
    :return:
    '''
    data_train, label_train = next(iter(trainloader))
    data_test, label_test = next(iter(testloader))
    data_train, data_test = data_train.cuda(), data_test.cuda()
    label_train, label_test = label_train.cuda(), label_test.cuda()
    root_weights = "../weights_ae"
    file_list = os.listdir(root_weights)
    plt.figure()
    model_g.eval()
    model_d.eval()
    torch.no_grad()
    doc = open("../ae_vir_pred.txt", "a")
    for root in tqdm(file_list):
        state = torch.load("../weights_ae/" + root)
        model_g.load_state_dict(state["model"])
        vir_train, vir_test = model_g.module.generate_virtual(data_train), model_g.module.generate_virtual(data_test)
        img = torch.concat([data_train, data_test, vir_train, vir_test], dim=0)
        img = torchvision.utils.make_grid(img)
        img = img.cpu().detach().numpy()
        img = np.transpose(img, [1, 2, 0])
        plt.figure()
        plt.imshow(img)
        os.makedirs("../ae_img", exist_ok=True)
        loss = state["loss"]
        epoch = state["epoch"]
        plt.savefig(f"../ae_img/img_aevir--loss{loss:.2f}--{epoch}.jpg")
        plt.clf()
        pred_train, pred_test = model_d(vir_train), model_d(vir_test)
        pred_train, pred_test = pred_train.softmax(dim=1), pred_test.softmax(dim=1)
        pred_train, pred_test = torch.topk(pred_train, 2, dim=1), torch.topk(pred_test, 2, dim=1)
        doc.write(f"{pred_train}\r\n{pred_test}")
        print(f"{pred_train}\n{pred_test}")
    doc.close()


# guanchachongou()
def zhaodaozuixiaoloss():
    '''
    找所有权重中loss最小的那个权重
    :return:
    '''
    root_weights = "../weights_ae"
    file_list = os.listdir(root_weights)
    min = 5
    k = None
    for root in tqdm(file_list):
        state = torch.load(root_weights + f"/{root}")
        if state["loss"] < min and "fixid0" not in root and "fixid1" not in root and "fixid3" not in root:
            min = state["loss"]
            k = root_weights + f"/{root}"
    print(f"最小loss为{min},权重路径为{k}")
    # 最小loss为1.5453410011947535,权重路径为../weights_ae/ae__fixid0__epoch192__0.06__Adam__batch64.pth
    # 最小loss为1.8596896973385173,权重路径为../weights_ae/ae__fixid1__epoch197__0.06__Adam__batch64.pth
    # 最小loss为1.8718528254016948,权重路径为../weights_ae/ae__fixid3__epoch170__0.06__Adam__batch64.pth
    # 最小loss为1.905036196587192, 权重路径为../weights_ae/ae__fixid4__epoch160__0.06__Adam__batch64.pth


# zhaodaozuixiaoloss()

def tongjizhixindu():
    '''
    统计topk置信度情况
    那不就是交叉熵
    还有均值,还有最大值最小值
    :return:
    '''
    model_g.eval()
    torch.no_grad()
    root_lsit = ["../weights_ae/ae__fixid0__epoch192__0.06__Adam__batch64.pth",
                 "../weights_ae/ae__fixid1__epoch197__0.06__Adam__batch64.pth",
                 "../weights_ae/ae__fixid3__epoch170__0.06__Adam__batch64.pth",
                 "../weights_ae/ae__fixid4__epoch160__0.06__Adam__batch64.pth"]
    for root in root_lsit:
        state = torch.load(root)
        for data, label in testloader:
            data = data.cuda()
            label = label.cuda()
            virtual_data = model_g.module.generate_virtual(data)
            outputs = model_d(virtual_data).softmax(dim=1)
            virtual_label = F.one_hot(label, num_classes) / 2
            index_0 = range(0, len(virtual_label), 2)
            index_1 = range(1, len(virtual_label), 2)
            virtual_label = virtual_label[index_0] + virtual_label[index_1]
            virtual_label = virtual_label.detach()
            loss = criterion(outputs, virtual_label)
            loss.backward()
            for a, b in zip(outputs, virtual_label):
                print(a)
                print(b)
                print("-" * 10)


def guancha_ae_chonggou():
    '''
    查看ae输出
    :return:
    '''
    state_g = torch.load("../betterweights/ae_miao_OnlyToTensor--sigmoid--epoch348--loss0.03.pth")
    model_g.load_state_dict(state_g["model"])


# tongjizhixindu()
if __name__ == "__main__":
    pass
    guancha_ae_chonggou()

