import argparse
import math
import os

import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn.functional as F
from utils import *
from model import *
import torchvision
import matplotlib.pyplot as plt

'''import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import sys
# from resnet_10 import resnet32
from model import ResNet18, ResNet34'''

'''# 超参数
batch_size = 4
epochs = 200
lr = 0.1
# 数据集 # [, 3, 32, 32]
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_train_cifar = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    # transforms.Normalize(mean, std),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=False, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=False, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False, num_workers=0)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def showimg(img):
    img=(img+1)/2
    # img=img/2+0.5
    img=img.cpu().numpy()
    img=np.transpose(img,[1,2,0])
    plt.imshow(img)
    plt.show()
dataiter=iter(trainloader)
images,labels=next(dataiter)
img=torchvision.utils.make_grid(images)
showimg(img)'''
'''
virtual_label=torch.tensor([0,2,2,3,4,5,1,1])
b=torch.nn.functional.one_hot(virtual_label,6)/2
idx_0=range(0,len(b),2)
idx_1=range(1,len(b),2)
print(b)
b=b[idx_0]+b[idx_1]
print(b.requires_grad)'''

'''import torchvision.transforms as transforms
def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        return img_tensor.mul_(std[:,None,None]).add_(mean[:,None,None]).detach()
        # print(img_tensor)
a=torch.ones([3,6,6])*24
b=torch.tensor([2,3,4])
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.2,1)),
])
c=transform_convert(a,transform_train)
print(c)'''
'''from model import *
from utils import *
import torchvision
import numpy as np
import matplotlib.pyplot as plt
batch_size = 128
transform = transforms.Compose(
    [transforms.ToTensor(), ])
transform_only_norm = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
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

trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=False, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)
i=0
for data, label in trainloader:
    data, label = data.cuda(), label.cuda()
    imgs = data
    imgs = torchvision.utils.make_grid(imgs)
    imgs = transform_convert(imgs, transform_only_norm)
    print(imgs.shape)
    imgs = imgs.cpu().detach().numpy()
    imgs = np.transpose(imgs, [1, 2, 0])
    plt.imshow((imgs * 255).astype(np.uint8))
    plt.show()
    i += 1
    if (i >= 3):
        break'''
'''import torch
from model import *

model = AutoEncoder()
parameters = "conv1.0.weight conv1.0.bias conv1.1.weight conv1.1.bias conv2.0.weight conv2.0.bias conv2.1.weight conv2.1.bias conv3.0.weight conv3.0.bias conv3.1.weight conv3.1.bias "
for k, v in model.named_parameters():
    if k in parameters:
        v.requires_grad=False
for k,v in model.named_parameters():
    print(k,v.requires_grad)
'''
'''data=torch.tensor([i for i in range(10)])
x=F.one_hot(data,10)/2
id1=range(0,len(x),2)
id2=range(1,len(x),2)
x=x[id1]+x[id2]
print(data)
print(x)
criterion=torch.nn.CrossEntropyLoss()
y=torch.tensor([0.5,0.5,0,0,0,0,0,0,0,0],dtype=float).softmax(dim=0)
# y=y.softmax(dim=0)
print(y)
print(x[0])
loss=criterion(y,x[0])
print(loss)'''
# x=F.one_hot(torch.tensor(1),10).float()
'''x=torch.tensor([0.5,0.4,0,0,0,0,0,0,0,0],dtype=float)
print(x)
y=torch.tensor([0.5,0.5,0,0,0,0,0,0,0,0],dtype=float)
print(y)
criterion=torch.nn.CrossEntropyLoss()
loss=criterion(x,y)
print("loss=",loss)'''
'''trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=False, transform=transform_only_tensor)
testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=False, transform=transform_only_tensor)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                         shuffle=True, num_workers=2)
model_g=AutoEncoder().cuda()
model_g=torch.nn.DataParallel(model_g)
model_g.load_state_dict("../betterweights/")'''

'''# cifar10类,只取两类
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=False, transform=transform_only_tensor)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_only_tensor)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
x = testset.data
y = testset.targets
y = np.array(y).squeeze()

index = np.where((y == 5) | (y==9))[0]
trainset.data = x[index]
trainset.targets = y[index]
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
print(len(trainloader),len(trainset))
for x, y in trainloader:
    imgs=torchvision.utils.make_grid(x).numpy()
    imgs=np.transpose(imgs,[1,2,0])
    plt.figure()
    plt.imshow(imgs)
    plt.title([classes[i.item()] for i in y])
    print(y)
    plt.show()
    print(y.shape)'''
'''model = AutoEncoder_Miao()
l = []
for k, v in model.named_parameters():
    l.append(k)
print(l)
for k, v in model.named_parameters():
    if k in l:
        print(k)
l = ['conv1.0.weight', 'conv1.0.bias', 'conv2.0.weight', 'conv2.0.bias', 'conv3.0.weight', 'conv3.0.bias',
     'conv4.0.weight', 'conv4.0.bias', 'ct0.0.weight', 'ct0.0.bias', 'ct1.0.weight', 'ct1.0.bias', 'ct2.0.weight',
     'ct2.0.bias', 'ct3.0.weight', 'ct3.0.bias', 'ct4.0.weight', 'ct4.0.bias', 'ct5.0.weight', 'ct5.0.bias',
     'ct6.0.weight', 'ct6.0.bias', 'ct7.0.weight', 'ct7.0.bias', 'ct8.0.weight', 'ct8.0.bias']
print(" ".join(l))'''
'''
import torch
import math
Y_pred1 = torch.Tensor([[0.1,0.2,0.7],
                        [1.1,0.1,0.2],
                        [0.2,2.1,0.1]])         # 注意这里没有经过激活函数（即softmax函数），而是直接扔进CrossEntropyLoss里
Y_pred2 = torch.Tensor([[0.8,0.2,0.3],
                        [0.2,0.3,0.5],
                        [0.2,0.2,0.5]])
a=math.exp(0.2)/(math.exp(0.1)+math.exp(0.2)+math.exp(0.7))
print(a)
b=-math.log(a)
print(b)
c=torch.tensor([0.1,0.2,0.7])
print(c)
d=c.softmax(dim=0)
print(d)
cel=torch.nn.CrossEntropyLoss()
loss=cel(c,torch.tensor([0,1,0],dtype=float))
print(loss)
'''
'''
# 保存argparse的信息
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument("--optimizer", default="Adam", help="Adam SGD")
parser.add_argument("--epoch", type=int, default=600)
parser.add_argument("--gpus", default="0,1")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--need_fixed", type=bool, default=False)
parser.add_argument("--fixedid", type=int, default=1)
args = parser.parse_args()
print(args)
'''
'''
十分类各方面都比较均匀,最大loss为2.3026
'''
'''
# 交叉熵损失函数的值
import torch
a=torch.tensor([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
a_1=torch.tensor([0,1,0,0,0,0,0,0,0,0])
a_2=torch.tensor([0,0.4,0.4,0,0,0,0,0,0,0])
b=torch.tensor([0,1,0,0,0,0,0,0,0,0])
b_1=torch.tensor([0,0.5,0.5,0,0,0,0,0,0,0])

def getloss(x,y):
    x=-torch.log(x)
    z=x*y
    print(z)
    print(z.sum())

getloss(a_2,b_1)
for i in range(11):
    x=i*0.1
    y=torch.tensor([x])
    y=-torch.log(y)
    print(f"{x:.1f} : {y.item():.2f}")
'''
'''# 两次backward
one=torch.tensor(-1)
x=torch.tensor([[1,2],[3,4]],requires_grad=False)
w1=torch.tensor(2.0,requires_grad=True)
w2=torch.tensor(3.0,requires_grad=True)
l1=x*w1
l2=l1+w2
loss=l2.mean()
loss.backward(retain_graph=True)
print(w1.grad,w2.grad)
loss2=l2.sum()
loss2.backward()
print(w1.grad,w2.grad)'''
#
'''#BCELoss
x=torch.tensor([0.8]).float()
y=torch.tensor([0.8]).float()
criterion=torch.nn.BCELoss()
loss=criterion(x,y)
print(loss)
z=torch.FloatTensor([0.5])
a=torch.log(z)
print(a)
x=0.8
y=0.8
c=y*math.log(x)+(1-y)*math.log(1-x)
print(-c)'''
'''def f(x):
    return (x*w1+w2).mean()
#我知道了,retaingraph保存的是计算图,但是如果我计算图没改变的话是可以多次backward的
one=torch.tensor(-1)
x=torch.tensor([[1,2],[3,4]],requires_grad=False)
x1=torch.tensor([[1,2],[3,4]],requires_grad=False)
w1=torch.tensor(2.0,requires_grad=True)
w2=torch.tensor(3.0,requires_grad=True)
l1=f(x)
l1.backward()
print(w1.grad,w2.grad)
l2=f(x1)*l1
l2.backward()
print(w1.grad,w2.grad)'''
'''model = AutoEncoder_Miao()


def fix_para(model, set_fixed=True, model_name="miao"):
    fixed_pa = get_fixed_parameters(set_fixed, model_name)
    for k, v in model.named_parameters():
        if k.replace("module.", "") in fixed_pa:
            v.requires_grad = False
    print("-" * 10, "固定权重参数")
    for k, v in model.named_parameters():
        print(f"{k}:{v.requires_grad}")
    print("-" * 10)
    return model
model=fix_para(model)
print("hah")
for k,v in model.named_parameters():
    print(f"{k}:{v.requires_grad}")'''
'''a=torch.FloatTensor([-1])
print(a.shape)
b=torch.tensor(1)
print(b,b.shape)'''
'''def cross(x,y):
    return -1*y*math.log(x)
a=cross(0.1,0.5)+cross(0.1,0.5)
print(a)'''
'''os.environ["CUDA_VISIBLE_DEVICES"]="2"
torch.cuda.empty_cache()'''
'''
#cifar100获得类名字
cifar100_train = torchvision.datasets.CIFAR100(root="../data/cifar100", train=True, download=False,
                                               transform=transform_only_tensor)
cifar100_test = torchvision.datasets.CIFAR100(root="../data/cifar100", train=False, download=False,
                                              transform=transform_only_tensor)
import _pickle


def unpickle(file):
    import _pickle
    with open(file, 'rb') as fo:
        dict = _pickle.load(fo, encoding='bytes')
    return dict




# 解压数据
def unpickle(file):
    import _pickle
    with open(file, 'rb') as fo:
        dict = _pickle.load(fo, encoding='bytes')
    return dict


# 给定路径添加数据
def Dealdata(meta, train):
    fineLabelNameDict = {}
    # 精细类别对应的粗糙类别 精细序号：粗糙序号-粗糙名称
    fineLableToCoraseLabelDict = {}
    for fineLabel, coarseLabel in zip(train[b'fine_labels'], train[b'coarse_labels']):
        if fineLabel not in fineLabelNameDict.keys():
            fineLabelNameDict[fineLabel] = meta[b'fine_label_names'][fineLabel].decode('utf-8')
        if fineLabel not in fineLableToCoraseLabelDict.keys():
            fineLableToCoraseLabelDict[fineLabel] = str(coarseLabel) + "-" + meta[b'coarse_label_names'][
                coarseLabel].decode('utf-8')
    return fineLabelNameDict, fineLableToCoraseLabelDict


# 解压后meta的路径
metaPath = '../data/cifar100/cifar-100-python/meta'
# 解压后train的路径
trainPath = '../data/cifar100/cifar-100-python/train'

meta = unpickle(metaPath)
train = unpickle(trainPath)
fineLabelNameDict, fineLableToCoraseLabelDict = Dealdata(meta, train)
# print
cifar100_name_label={v:k for k,v in fineLabelNameDict.items()}
# print(cifar100_name_label)

cifar10_label=["airplane" ,"automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
for i in cifar100_name_label.keys():
    if i in cifar10_label:
        print(i,cifar100_name_label[i])
'''
'''cifar10_train = torchvision.datasets.CIFAR10(root="../data/cifar10", train=True, download=False, transform=transform_only_tensor)
trainloader_cifar10 = torch.utils.data.DataLoader(cifar10_train, batch_size=32, shuffle=True, num_workers=2)
for data,label in trainloader_cifar10:
    data=data.cuda()
    label=label.cuda()
    v=torch.ones([int(len(label)/2),10])*0.1.detach()
    print(v)
    print(label.shape,v.shape)
    print(label.dtype,v.dtype)'''
'''real_labels = 0.7 + 0.5 * torch.rand(10)
fake_labels = 0.3 * torch.rand(10)
print(real_labels,fake_labels)
real_label = real_labels[idx % 10]
fake_label = fake_labels[idx % 10]'''
'''
# 画crossentropyloss图
def cross(x,y):
    return -1*y*math.log(x)
a=cross(0.5,0.5)+cross(0.1,0.5)
print(a)
def draw_cel():

    import matplotlib.pyplot as plt
    for x in np.arange(0.1,1,0.01):
        y=1-x
        print(x,y)
        cel=cross(x,0.5)+cross(y,0.5)
        plt.scatter(x,cel)
    plt.show()
draw_cel()'''
'''
#所有pth文件
import os
results_root = "../results/train_generate_virtual_by_add"
a=os.listdir(results_root)
# [44:52] [57:62]
list=[]
for i in a:
    msel=float(i[44:52])
    cel=float(i[57:62])
    if cel<2.7 and msel<0.01:
        list.append(i)
print(list)
print(len(list))
# ae_miao_trained_by_add_with_mse_and_cel--mse0.034710--cel1.727.pth
'''

def jianchammchanshu():
    '''
    检查mmc函数
    '''
    model_d = getResNet("resnet" + "18").cuda()
    model_d = torch.nn.DataParallel(model_d)
    # 不应该拿训练好的模型来训练,应该是重头训练.
    state_d = torch.load("../betterweights/resnet18--transform_onlyToTensor--epoch199--acc095--loss017.pth")
    model_d.load_state_dict(state_d["model"])

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--gpus", default="0")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--use_scheduler", type=str, default="False", help="使用virtual训练discriminator的时候使用动态学习scheduler")
parser.add_argument("--loss_virtual_weight", type=float, default=1, help="压制训练时候,loss_virtual的权重")
# 压制训练时生成多少倍数的虚假图像
parser.add_argument("--virtual_scale", type=int, default=2, help="要多少个0.5倍正常样本数量的 virtual example")
args = parser.parse_args()
a=str(args).replace("(","").replace(")","").replace(",","--").replace(" ","").replace("=","")
print(a)