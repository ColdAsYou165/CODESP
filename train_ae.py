'''
按照老师思想试图生成虚假样本
固定分类器,训练ae.
效果不好,使用Adam lr0.06余弦退火 loss可以从2降到1.4但是再往下就降不下来了.\
'''
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.00001, help="model_g的lr")
parser.add_argument("--optimizer", default="Adam", help="Adam SGD")
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--gpus", default="0,1")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--need_fixed", type=bool, default=False)
parser.add_argument("--set_only_two_classes", type=bool, default=False, help="只使用两个类别")
parser.add_argument("--fixedid", type=int, default=1)
parser.add_argument("--loss", default="crossentropyloss", help="l1loss crossentropyloss")
parser.add_argument("--use_scheduler", type=bool, default=False, help="使用virtual训练discriminator的时候使用动态学习scheduler")
parser.add_argument("--wgan_optim", default="RMSprop", help="wgan中优化器的选择,推荐使用RMSprop,而不是Adam")

# wgan的 discriminator
parser.add_argument('--lr_d', type=float, default=0.0002, help='wgan discrinator lr, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--w_loss_weight', type=float, default=1e-5, help='wloss上加的权重,苗师兄wgan是1e-5')
parser.add_argument("--set_sigmoid", type=bool, default=True, help="wgan的dis是否需要sigmoid")
# 压制训练时生成多少倍数的虚假图像
parser.add_argument("--virtual_scale", type=int, default=2, help="要多少个0.5倍正常样本数量的 virtual example")
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
from tqdm import tqdm
# import torch.backends.cudnn as cudnn
import sys
from model import *
from utils import *

results_root = "../results/miao_ae_trainedbywgan"
os.makedirs(results_root, exist_ok=True)
writer = SummaryWriter()
torch.set_printoptions(profile="full")
# 数据集

num_classes = 10
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
# [, 3, 32, 32]
# 这个删去之后得有部分函数没有加载数据集,记得在每个函数里面加载数据集
'''trainset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=True,
                                        download=False, transform=transform_only_tensor)
testset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=False,
                                       download=False, transform=transform_only_tensor)'''


def getLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, set_only_two_classes=False, c1=0,
              c2=1):
    '''
    获得dataloader,cifar10类,只取两类
    :param dataset:
    :param batch_size:
    :param shuffle:
    :param num_workers:
    :param set_only_two_classes: True则cifar10类,只取两类
    :param c1:
    :param c2:
    :return:
    '''
    print(f"原始数据集大小:{len(dataset)}", end=",")
    y = np.array(dataset.targets)
    if set_only_two_classes:
        index = np.where((y == c1) | (y == c2))[0]
        x = dataset.data[index]
        y = np.array(dataset.targets)[index]
        dataset.data = x
        dataset.targets = y
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    print(
        f"新数据集大小{len(dataset)},loader数量:{len(dataloader)},data形状:{dataset.data.shape},targects形状{len(dataset.targets)}")
    return dataloader


# trainloader = getLoader(trainset, set_only_two_classes=args.set_only_two_classes, c1=0, c2=1)
# testloader = getLoader(testset, set_only_two_classes=args.set_only_two_classes, c1=0, c2=1)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# 模型

model_d = getResNet("resnet" + "18").cuda()
model_d = torch.nn.DataParallel(model_d)
state = torch.load("../betterweights/resnet18--transform_onlyToTensor--epoch199--acc095--loss017.pth")
model_d.load_state_dict(state["model"])

model_g = AutoEncoder_Miao().cuda()
model_g = torch.nn.DataParallel(model_g)
# 固定model_g的部分权重
if args.need_fixed:
    print("固定模型部分权重...")
    fixed_pa = get_fixed_parameters(args.fixedid, model="miao")
    for k, v in model_g.named_parameters():
        if k.replace("module.", "") in fixed_pa:
            v.requires_grad = False
    print("-" * 10, "固定权重参数")
    for k, v in model_g.named_parameters():
        print(f"{k}:{v.requires_grad}")
    print("-" * 10)


def fix_para(model, set_fixed=args.fixedid, model_name="miao"):
    '''
    固定模型部分权重
    :param model:模型
    :param set_fixed:是否固定权重
    :param model_name: "miao"是AE_miao
    :return: 固定权重的模型
    '''
    fixed_pa = get_fixed_parameters(set_fixed, model_name)
    for k, v in model.named_parameters():
        if k.replace("module.", "") in fixed_pa:
            v.requires_grad = False
    print("-" * 10, "固定权重参数")
    for k, v in model.named_parameters():
        print(f"{k}:{v.requires_grad}")
    print("-" * 10)
    return model


# 待会再加一个原始图像与新图像的loss,使得生成的图像与原来的图像很接近
# criterion = torch.nn.CrossEntropyLoss().cuda()  # 使得生成的图像更像两个类别
if args.loss == "l1loss":
    criterion = torch.nn.L1Loss().cuda()  # 使得生成的图像更像两个类别
elif args.loss == "crossentropyloss":
    criterion = torch.nn.CrossEntropyLoss().cuda()  # 使得生成的图像更像两个类别
if args.optimizer == "SGD":
    optimizer_g = torch.optim.SGD(filter(lambda p: p.requires_grad, model_g.parameters()), lr=lr, momentum=0.9,
                                  weight_decay=5e-4)  # 5的10的-4次方,0.0005
elif args.optimizer == "Adam":
    optimizer_g = torch.optim.Adam(filter(lambda p: p.requires_grad, model_g.parameters()), lr=lr)


# scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=200)
# 这航要改,训练次数要改

def testDiscriminator(model=getResNet("resnet" + "18").cuda(),
                      state_root="../betterweights/resnet18--transform_onlyToTensor--epoch199--acc095--loss017.pth"):
    '''
    测试分类器性能
    :param model:
    :param state_root:
    :return:
    '''
    criterion = torch.nn.CrossEntropyLoss()
    state = torch.load(state_root)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(state["model"])
    print("已加载分类器权重,acc=", state["acc"])
    model.eval()
    loss_test = 0
    acc_test = 0
    with torch.no_grad():
        for data, label in tqdm(testloader):
            data = data.cuda()
            label = label.cuda()
            pred = model(data)
            loss = criterion(pred, label)
            loss_test += loss.item()
            pred_label = torch.argmax(pred, dim=1)
            print(pred[0])
            print(pred.softmax(dim=1)[0])
            acc_test += (pred_label == label).int().sum().item()
    loss_test /= len(testloader)
    acc_test /= len(testset)
    print(f'经测试分类器在测试集上acc={acc_test:.4f},loss={loss_test:.4f}')


# print(len(trainloader))#782,batchsize=64
def train_generate_virtual():
    '''
    由多个类生成虚拟样本
    :return:
    '''
    # 训练与测试
    for epoch in range(0, epochs):
        # 训练
        model_g.train()
        loss_train = 0
        print("training...")
        for batch, (data, label) in enumerate(trainloader):
            data = data.cuda()
            label = label.cuda()
            virtual_data = model_g.module.generate_virtual(data)
            # 后面不需要接softmax,因为训练的时候model_d就没加sotfmax,另外,crossentropyloss方法中已经自带了softmax.
            outputs = model_d(virtual_data)
            virtual_label = F.one_hot(label, num_classes) / 2
            index_0 = range(0, len(virtual_label), 2)
            index_1 = range(1, len(virtual_label), 2)
            virtual_buffer = virtual_label[index_0] + virtual_label[index_1]
            virtual_label = virtual_buffer.detach()
            loss = criterion(outputs, virtual_label)
            optimizer_g.zero_grad()
            loss.backward()
            optimizer_g.step()
            loss_train += loss.item()
            # 打印
            if True:
                if batch % 500 == 0:
                    for a, b in zip(outputs[:2].softmax(dim=1), virtual_label[:2]):
                        print("-" * 10)
                        print(a)
                        print(b)
                        print("-" * 10)
        loss_train = loss_train / len(trainloader)
        print(f"train epoch{epoch} loss={loss_train}")
        # 测试
        model_g.eval()
        loss_test = 0
        for batch, (data, label) in enumerate(testloader):
            with torch.no_grad():
                data = data.cuda()
                label = label.cuda()
                virtual_data = model_g.module.generate_virtual(data)
                outputs = model_d(virtual_data)
                virtual_label = F.one_hot(label, num_classes) / 2
                index_0 = range(0, len(virtual_label), 2)
                index_1 = range(1, len(virtual_label), 2)
                virtual_label = virtual_label[index_0] + virtual_label[index_1]
                virtual_label = virtual_label.detach()
                loss = criterion(outputs, virtual_label)
                loss_test += loss.item()

        loss_test = loss_test / len(testloader)
        # scheduler_g.step()
        # 保存
        print(f"test epoch{epoch} loss={loss_test}")
        writer.add_scalars("loss", {"train": loss_train, "test": loss_test}, epoch)
        state_g = {"model": model_g.state_dict(), "loss": loss_test, "epoch": epoch, "args": f"{args}"}
        torch.save(state_g,
                   f"{results_root}/modelmiao__epoch{epoch}__lr{lr}__batch{args.batch_size}__loss{loss_test:.2f}.pth")


def train_generate_virtual_with_WGAN():
    '''
    引入w距离,和分类器 一起训练ae生成virtual example的能力
    w距离并不是完全的w距离,要加上sigmoid,这样wgan的discrimnitor容易优化.
    :return:
    '''
    results_root = "../results/miao_ae_trainedbywgan"
    os.makedirs(results_root, exist_ok=True)
    writer.add_text(f"trained by wgan,实验描述{args}", f"")
    # 模型
    discriminator_sigmoid = ResNet18_sigmoid(set_sigmoid=args.set_sigmoid).cuda()  # 0正常1虚假
    discriminator_sigmoid = torch.nn.DataParallel(discriminator_sigmoid)

    state_g = torch.load("../betterweights/ae_miao_OnlyToTensor--sigmoid--epoch348--loss0.03.pth")
    model_g = torch.nn.DataParallel(AutoEncoder_Miao().cuda())
    model_g.load_state_dict(state_g["model"])
    model_g = fix_para(model_g, 1, "miao")  # 固定生成器模型部分权重

    model_d = getResNet("resnet" + "18").cuda()
    model_d = torch.nn.DataParallel(model_d)
    state_d = torch.load("../betterweights/resnet18--transform_onlyToTensor--epoch199--acc095--loss017.pth")
    model_d.load_state_dict(state_d["model"])
    # 数据集
    trainset = torchvision.datasets.CIFAR10(root="../data/cifar10", train=True, download=False, transform=transform_only_tensor)
    testset = torchvision.datasets.CIFAR10(root="../data/cifar10", train=False, download=False, transform=transform_only_tensor)
    trainloader = getLoader(trainset, args.batch_size, True, 2, False)
    testloader = getLoader(testset, args.batch_size, True, 2, False)
    # 选择优化器
    if args.wgan_optim == "Adam":
        optimizer_dis = torch.optim.Adam(discriminator_sigmoid.parameters(), lr=args.lr_d, betas=(args.beta1, 0.999))
        optimizer_g = torch.optim.Adam(filter(lambda p: p.requires_grad, model_g.parameters()), lr=args.lr)
    elif args.wgan_optim == "RMSprop":
        optimizer_dis = torch.optim.RMSprop(discriminator_sigmoid.parameters(), lr=args.lr_d)
        optimizer_g = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model_g.parameters()), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    model_d.eval()
    one = torch.tensor(1).float().cuda()
    print("training...")
    for epoch in range(epochs):
        min_text = 6
        loss_all_discriminator_real = 0  # 分类器在正常样本的loss,对正常样本识别正确的概率
        loss_all_discriminator_vir = 0  # 分类器在正常样本的loss,对正常样本识别正确的概率
        loss_all_w = 0  # 整体的wloss,希望我们生成的virtual的wloss越低越好,说明越像真实样本分布
        loss_all_v = 0  # 整体的virtual example的crossentropyloss
        for batch, (data, label) in enumerate(trainloader):
            data = data.cuda()
            label = label.cuda()
            # 训练wgan的discriminator
            optimizer_dis.zero_grad()
            # 正常样本
            d_loss_real = discriminator_sigmoid(data.detach())
            d_loss_real = torch.mean(d_loss_real)
            d_loss_real.backward(one)  # real0 virtual1
            loss_all_discriminator_real += d_loss_real.item()  # 分类器在正常样本的loss,对正常样本识别正确的概率
            # 虚假样本
            virtual_data = model_g.module.generate_virtual(data)
            virtual_label = F.one_hot(label, num_classes) / 2
            index_0 = range(0, len(virtual_label), 2)
            index_1 = range(1, len(virtual_label), 2)
            virtual_label = virtual_label[index_0] + virtual_label[index_1]
            virtual_label = virtual_label.detach()
            d_loss_virtual = discriminator_sigmoid(virtual_data.detach())
            d_loss_virtual = torch.mean(d_loss_virtual)
            d_loss_virtual.backward(-one)
            loss_all_discriminator_vir += d_loss_virtual.item()  # 分类器在vir样本的输出,越接近1越好
            optimizer_dis.step()

            # 训练自编码器生成virtual example
            optimizer_g.zero_grad()
            pred = discriminator_sigmoid(virtual_data)
            pred = torch.mean(pred)
            w_loss = pred * args.w_loss_weight
            w_loss.backward(one, retain_graph=True)  # 我们希望生成的bir样本在dis越接近0(真)越好

            outs = model_d(virtual_data)
            v_loss = criterion(outs, virtual_label)  # 同时我们希望生成的vir在父母label上的值越大越好.
            v_loss.backward()
            optimizer_g.step()
            loss_all_w += pred.item()  # 虚假样本在discriminator上的值,值越大越虚假
            loss_all_v += v_loss.item()  # 虚假样本在model_d上的loss,
        loss_all_w /= len(trainloader)
        loss_all_v /= len(trainloader)
        loss_all_discriminator_real /= len(trainloader)
        loss_all_discriminator_vir /= len(trainloader)
        print(
            f"train epoch[{epoch}/{epochs}]:wloss={loss_all_w:.8f},virtuacrosslloss={loss_all_v:.8f},loss_all_discriminator_real={loss_all_discriminator_real:.8f},loss_all_discriminator_vir={loss_all_discriminator_vir:.8f}")

        # 测试
        model_g.eval()
        discriminator_sigmoid.eval()
        test_crossentropyloss = 0

        for batch, (data, label) in enumerate(testloader):
            with torch.no_grad():
                data = data.cuda()
                label = label.cuda()
                virtual_data = model_g.module.generate_virtual(data)
                outputs = model_d(virtual_data)
                virtual_label = F.one_hot(label, num_classes) / 2
                index_0 = range(0, len(virtual_label), 2)
                index_1 = range(1, len(virtual_label), 2)
                virtual_label = virtual_label[index_0] + virtual_label[index_1]
                virtual_label = virtual_label.detach()
                loss = criterion(outputs, virtual_label)
                test_crossentropyloss += loss.item()
        test_crossentropyloss = test_crossentropyloss / len(testloader)
        print(f"test epoch[{epoch}/{epochs}:crossentropyloss={test_crossentropyloss}]")
        writer.add_scalars("loss", {"wloss": loss_all_w,
                                    "train_entroloss": loss_all_v,
                                    "loss_all_discriminator_real": loss_all_discriminator_real,
                                    "loss_all_discriminator_vir": loss_all_discriminator_vir,
                                    "test_entroloss": test_crossentropyloss},
                           epoch)
        if test_crossentropyloss < min_text:
            min_text = test_crossentropyloss
            state = {"model": model_g.state_dict(), "loss": test_crossentropyloss, "epoch": epoch}
            torch.save(state,
                       results_root + f"/ae_miao__epoch{epoch}__w_weight{args.w_loss_weight}__entroloss{test_crossentropyloss:.2f}.pth")


# 由两个类生成虚假样本
# 先判断这两个类在model_d上的acc,再判断这两个类经过ae重构后在model_d上的acc,证明我们的模型没有问题,最后再是用这两个类生成虚假样本.其实用这两个类,直接像上面那样shuffle就可以了.


def debug_reallabel():
    '''
    我怀疑我标签给错了,所以会有这个函数
    :return:
    '''
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                              shuffle=True, num_workers=2)
    for batch, (data, label) in enumerate(trainloader):
        data = data.cuda()
        label = label.cuda()
        img = torchvision.utils.make_grid(data)
        names = [f"{classes[i.item()]}.{i.item()}" for i in label]

        virtual_label = F.one_hot(label, num_classes) / 2
        index_0 = range(0, len(virtual_label), 2)
        index_1 = range(1, len(virtual_label), 2)
        virtual_label = virtual_label[index_0] + virtual_label[index_1]
        virtual_label = virtual_label.detach()

        plt.figure()
        for i in range(10):
            print(classes[i], i, end=" ")
        print()
        print(label)
        print("-" * 10)
        print(virtual_label, torch.topk(virtual_label, 2, dim=1)[1])
        plt.title(f"{names}")
        img = img.cpu().detach().numpy()
        img = np.transpose(img, [1, 2, 0])
        plt.imshow(img)
        plt.show()


def observe_virtual():
    '''
    加载ae模型,检查ae生成的virtual图像长什么样子
    :return:
    '''
    results_root = "../results/miao_ae_cross10loss"
    # state_g = torch.load(f"{results_root}/__modelmiao__epoch{100}__lr{0.0001}__batch{args.batch_size}__loss{1.43}.pth")
    state_g = torch.load(results_root + "/" + "modelmiao__epoch135__lr0.0001__batch256__loss1.47.pth")
    print("已加载model_g权重,loss为", state_g["loss"])
    model_g.load_state_dict(state_g["model"])
    trainset = torchvision.datasets.CIFAR10(root="../data/cifar10", train=True, download=False, transform=transform_only_tensor)
    testset = torchvision.datasets.CIFAR10(root="../data/cifar10", train=False, download=False, transform=transform_only_tensor)
    trainloader = getLoader(trainset, set_only_two_classes=False, c1=0, c2=1, batch_size=16)
    testloader = getLoader(testset, set_only_two_classes=False, c1=0, c2=1, batch_size=16)
    model_g.eval()
    for data, label in testloader:
        data = data.cuda()
        label = label.cuda()
        reconstruct = model_g(data)
        vir = model_g.module.generate_virtual(data)
        pred = model_d(vir)
        pred_softmax = pred.softmax(dim=1)
        print("pred:", pred_softmax)
        outs = vir
        print(f"virts min={outs.min()} and max={outs.max()}")
        # 计算loss
        virtual_label = F.one_hot(label, num_classes) / 2
        index_0 = range(0, len(virtual_label), 2)
        index_1 = range(1, len(virtual_label), 2)
        virtual_label = virtual_label[index_0] + virtual_label[index_1]
        virtual_label = virtual_label.detach()
        loss = criterion(pred, virtual_label)
        print(f"loss:{loss}")

        imgs = torch.concat([data, reconstruct, outs], dim=0)
        imgs = torchvision.utils.make_grid(imgs)
        imgs = imgs.detach().cpu().numpy()
        imgs = np.transpose(imgs, [1, 2, 0])
        names = [f"{classes[i.item()]}" for i in label]
        plt.figure()
        plt.imshow(imgs)
        plt.title(names)
        plt.show()


def get_mmc(model, loader):
    '''
    用于ood数据集,给定model和loader,求mmc
    '''
    t0 = time.time()
    mmc = 0
    model.eval()
    with torch.no_grad():
        for batch, (data, label) in enumerate(loader):
            data = data.cuda()
            label = label.cuda()
            pred = model(data).softmax(dim=1)
            pred_max = torch.max(pred, dim=1)[0].mean()
            mmc += pred_max.item()
    mmc /= len(loader)
    t1 = time.time()
    # print(f"耗时{t1 - t0:.2f}")
    return mmc


def get_acc(model, loader):
    '''
    用于id数据集,给定模型和loader,求acc
    '''
    num = 0
    acc = 0
    model.eval
    with torch.no_grad():
        for batch, (data, label) in enumerate(loader):
            data = data.cuda()
            label = label.cuda()
            pred = model(data)
            acc += (torch.argmax(pred, dim=1) == label).int().sum().item()
            num += len(data)
    acc /= num
    return acc


def observe_ood_mmc():
    '''
    观察resnet18模型在cifar10训练后,观察在ood数据集(cifar100,SVHN)上的mmc
    :return:
    '''
    results_root = "../result/observe_ood_mmc"
    '''    
    os.makedirs(results_root,exist_ok=True)
    state_g=torch.load("../betterweights/ae_miao_OnlyToTensor--sigmoid--epoch348--loss0.03.pth")
    model_g.load_state_dict(state_g["model"])
    '''
    # 先测试下原始的resnet18在各个数据集上的mmc
    cifar100_train = torchvision.datasets.CIFAR100(root="../data/cifar100", train=True, download=False,
                                                   transform=transform_only_tensor)
    cifar100_test = torchvision.datasets.CIFAR100(root="../data/cifar100", train=False, download=False,
                                                  transform=transform_only_tensor)
    svhn_train = torchvision.datasets.SVHN(root="../data/svhn", split="train", download=False,
                                           transform=transform_only_tensor)
    svhn_test = torchvision.datasets.SVHN(root="../data/svhn", split="test", download=False,
                                          transform=transform_only_tensor)
    cifar10_train = torchvision.datasets.CIFAR10(root="../data/cifar10", train=True, download=False, transform=transform_only_tensor)
    cifar10_test = torchvision.datasets.CIFAR10(root="../data/cifar10", train=False, download=False, transform=transform_only_tensor)
    # 哦是我忘了加载loader
    trainloader_cifar100 = torch.utils.data.DataLoader(cifar100_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader_cifar100 = torch.utils.data.DataLoader(cifar100_test, batch_size=args.batch_size, shuffle=True, num_workers=2)
    trainloader_svhn = torch.utils.data.DataLoader(svhn_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader_svhn = torch.utils.data.DataLoader(svhn_test, batch_size=args.batch_size, shuffle=True, num_workers=2)
    trainloader_cifar10 = DataLoader(cifar10_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader_cifar10 = DataLoader(cifar10_test, batch_size=args.batch_size, shuffle=True, num_workers=2)
    # mmc_cifar100
    # mmc_cifar100 = get_mmc(model_d, testloader_cifar100)
    # print(f"mmc_cifar100={mmc_cifar100}")
    # mmc_svhn
    # mmc_svhn = get_mmc(model_d, testloader_svhn)
    # print(f"mmc_svhn={mmc_svhn}")
    mmc_cifar10 = get_mmc(model_d, testloader_cifar10)
    print(mmc_cifar10)


def get_virtual_example(model_g=model_g, data=None, scale=2):
    '''
    通过gan生成virtual example,virtual label为0.1,0.1,0.1....
    :return:virtual_data,virtual_label
    '''
    list = []
    for i in range(scale):
        data = data[torch.randperm(len(data))].detach()
        data_virtual = model_g.module.generate_virtual(data).detach()
        list.append(data_virtual)
        # 压制训练时候,虚假样本的label应该都是0.1,我设置错了.
    data_virtual = torch.concat(list, dim=0).detach()
    label_virtual = (torch.ones([len(data_virtual), 10]) * 0.1).cuda().detach()
    return data_virtual, label_virtual


def train_discriminator_by_virtual():
    '''
    拿虚拟样本和正常样本训练discriminator
    v2:我们的目标是观察在ood数据集上的mmc有没有降下来
    进一步提升
    :return:
    '''
    print(str(args))
    writer.add_text("args", str(args))
    results_root = "../results/train_discriminator_by_virtual"
    os.makedirs(results_root, exist_ok=True)
    # 模型
    model_d = getResNet("resnet" + "18").cuda()
    model_d = torch.nn.DataParallel(model_d)
    state_d = torch.load("../betterweights/resnet18--transform_onlyToTensor--epoch199--acc095--loss017.pth")
    model_d.load_state_dict(state_d["model"])

    model_g = AutoEncoder_Miao().cuda()
    model_g = torch.nn.DataParallel(model_g)
    state_g = torch.load("../betterweights/modelmiao_generatevirtual0__epoch103__lr0.0001__batch256__loss1.47.pth")  # 没有引入wgan训练的ae
    # state_g = torch.load("../betterweights/ae_miao__epoch234__w_weight0.001__entroloss1.97.pth")  # 引入wgan训练的ae
    model_g.load_state_dict(state_g["model"])
    model_g.eval()

    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=200)
    # 数据集
    cifar10_train = torchvision.datasets.CIFAR10(root="../data/cifar10", train=True, download=False, transform=transform_only_tensor)
    cifar10_test = torchvision.datasets.CIFAR10(root="../data/cifar10", train=False, download=False, transform=transform_only_tensor)
    cifar100_train = torchvision.datasets.CIFAR100(root="../data/cifar100", train=True, download=False, transform=transform_only_tensor)
    cifar100_test = torchvision.datasets.CIFAR100(root="../data/cifar100", train=False, download=False, transform=transform_only_tensor)
    svhn_train = torchvision.datasets.SVHN(root="../data/svhn", split="train", download=False, transform=transform_only_tensor)
    svhn_test = torchvision.datasets.SVHN(root="../data/svhn", split="test", download=False, transform=transform_only_tensor)


    trainloader_cifar10 = torch.utils.data.DataLoader(cifar10_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader_cifar10 = torch.utils.data.DataLoader(cifar10_test, batch_size=args.batch_size, shuffle=True, num_workers=2)
    trainloader_cifar100 = torch.utils.data.DataLoader(cifar100_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader_cifar100 = torch.utils.data.DataLoader(cifar100_test, batch_size=args.batch_size, shuffle=True, num_workers=2)
    trainloader_svhn = torch.utils.data.DataLoader(svhn_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader_svhn = torch.utils.data.DataLoader(svhn_test, batch_size=args.batch_size, shuffle=True, num_workers=2)
    # 训练和测试
    # origin_mmc_cifar100 = get_mmc(model_d, testloader_cifar100)
    # origin_mmc_svhn = get_mmc(model_d, testloader_svhn)
    # print(f"origin_mmc_cifar100={origin_mmc_cifar100},origin_mmc_svhn={origin_mmc_svhn}")
    # 记录起初的mmc
    mmc_cifar100 = get_mmc(model_d, testloader_cifar100)
    mmc_svhn = get_mmc(model_d, testloader_svhn)
    acc = get_acc(model_d, testloader_cifar10)
    writer.add_scalars("loss", {"mmc_cifar100": mmc_cifar100, "mmc_svhn": mmc_svhn}, 0)
    writer.add_scalar("cifar10_acc", acc, 0)
    print(f"压制训练前:acc:{acc},mmc_cifar100={mmc_cifar100},mmc_svhn={mmc_svhn}")
    for epoch in range(args.epochs):
        # 训练
        model_d.train()
        loss_train_containv = 0
        for batch, (data, label) in enumerate(trainloader_cifar10):
            data = data.cuda()
            label = label.cuda()
            data_normal = data.detach()
            label_normal = F.one_hot(label, num_classes).detach()
            # data_virtual = model_g.module.generate_virtual(data).detach()
            # 压制训练时候,虚假样本的label应该都是0.1,我设置错了.
            # label_virtual = (torch.ones([int(len(label) / 2), 10]) * 0.1).cuda().detach()
            data_virtual, label_virtual = get_virtual_example(model_g=model_g, data=data, scale=args.virtual_scale)
            data_all = torch.concat([data_normal, data_virtual], dim=0).detach()
            label_all = torch.concat([label_normal, label_virtual], dim=0).detach()
            pred = model_d(data_all)
            loss = criterion(pred, label_all)
            optimizer_d.zero_grad()
            loss.backward()
            optimizer_d.step()
            if args.use_scheduler:
                scheduler.step()
            loss_train_containv += loss.item()
        loss_train_containv /= len(trainloader_cifar10)
        # 测试
        mmc_cifar100 = get_mmc(model_d, testloader_cifar100)
        mmc_svhn = get_mmc(model_d, testloader_svhn)
        acc = get_acc(model_d, testloader_cifar10)
        '''loss_test = 0
        acc_test = 0
        with torch.no_grad():
            for batch, (data, label) in enumerate(testloader):
                data = data.cuda()
                label = label.cuda()
                pred = model_d(data)
                loss = criterion(pred, label)
                loss_test += loss.item()
                acc_test += (torch.argmax(pred, dim=1) == label).int().sum().item()
        loss_test /= len(testloader)
        acc_test /= len(testset)'''
        writer.add_scalars("loss", {"train_containv": loss_train_containv, "mmc_cifar100": mmc_cifar100, "mmc_svhn": mmc_svhn}, epoch + 1)
        writer.add_scalar("cifar10_acc", acc, epoch + 1)
        print(f"cifar10_test_acc={acc}")
        print(f"epoch[{epoch}/{epochs}]", "loss_train_containv", loss_train_containv, "mmc_cifar100", mmc_cifar100, "mmc_svhn", mmc_svhn)


# ae_miao__epoch234__w_weight0.001__entroloss1.97.pth#通过wgan训练的不成功的ae

if __name__ == "__main__":
    pass
    # testDiscriminator() #测试分类器acc和loss,证明没有问题.
    # train_generate_virtual()  # 数据集中多个类混合生成虚假样本
    # debug_reallabel() #我怀疑我标签给错了,所以会有这个函数
    # observe_virtual()  # 观察检查ae生成的virtual图像长什么样子
    # train_discriminator_by_virtual()  # 最简单的方法,尝试用生成的virtual example来训练discriminator
    # train_generate_virtual_with_WGAN()  # 引入w距离训练ae生成虚假样本
    # observe_ood_mmc()  # 计算分类器模型在ood数据集上的mmc
    train_discriminator_by_virtual()  # 看看引入训练后ood mmc有没有降低
