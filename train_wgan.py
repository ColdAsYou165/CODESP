''''
难受啊.单独拎出来一个文件吧这样还好看点
结合wloss来训练 ae生成虚假图像,其中encoder()之后detach一下,这样只改变decoder的参数
discriminator使用师兄的discr
'''
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr_g", type=float, default=0.0002, help="model_g的lr")
parser.add_argument('--lr_dis', type=float, default=0.0002, help='wgan discrinator lr, default=0.0002')
parser.add_argument('--lr_scale', type=float, default=1e4, help='wgan discrinator lr, default=0.0002')
parser.add_argument("--optimizer", default="Adam", help="Adam SGD")
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--gpus", default="0")
parser.add_argument("--batch_size", type=int, default=128)
# parser.add_argument("--loss", default="crossentropyloss", help="l1loss crossentropyloss")
# crossentropyloss 和 mseloss 联合训练 ae
# wgan的 discriminator
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--w_loss_weight', type=float, default=1e-5, help='wloss上加的权重,苗师兄wgan是1e-5')
parser.add_argument("--set_sigmoid", type=str, default='False', help="wgan的dis是否需要sigmoid,不能要sigmoid!")
parser.add_argument("--g_k", type=int, default=1, help="一个batch内,生成器训练次数,目前有bug")
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
from tqdm import tqdm
# import torch.backends.cudnn as cudnn
import sys
from model import *
from utils import *

# v3是被老师骂了之后,才发现自己是那么的菜翻了那么多的错误.
# 到v5了,老师让我去试一下 wloss的权重从1e4到1e-4
# v6再试一试苗师兄的code吧,把sigmoid去掉
# v7换那个网上的lr0.0002的且没有sigmoid的discriminator
results_root = "../results/miao_ae_trainedbywgan_v6"
os.makedirs(results_root, exist_ok=True)
results_pic_root = results_root + f"/{args}"
os.makedirs(results_pic_root, exist_ok=True)
writer = SummaryWriter()
writer.add_text("实验描述", f"使用苗师兄ae和discriminator,{args}")

# 数据集

num_classes = 10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
trainset = torchvision.datasets.CIFAR10(root="../data/cifar10", train=True, download=False, transform=transform_only_tensor)
testset = torchvision.datasets.CIFAR10(root="../data/cifar10", train=False, download=False, transform=transform_only_tensor)
trainloader = DataLoader(trainset, args.batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(testset, args.batch_size, shuffle=True, num_workers=2)

# 模型
# discriminator = Discriminator_WGAN_miao_cifar10(set_sigmoid=args.set_sigmoid).cuda()
discriminator = simple_discriminator().cuda()
discriminator = torch.nn.DataParallel(discriminator)
model_d = getResNet("resnet" + "18").cuda()
model_d = torch.nn.DataParallel(model_d)
model_g = AutoEncoder_Miao().cuda()
model_g = torch.nn.DataParallel(model_g)

discriminator.apply(weights_init)

state_d = torch.load("../betterweights/resnet18--transform_onlyToTensor--epoch199--acc095--loss017.pth")
model_d.load_state_dict(state_d["model"])

model_g.apply(weights_init)
state_g = torch.load("../betterweights/ae_miao_OnlyToTensor--sigmoid--epoch348--loss0.03.pth")
model_g.load_state_dict(state_g["model"])

# 优化器
criterion_cel = torch.nn.CrossEntropyLoss().cuda()
criterion_bce = nn.BCEWithLogitsLoss().cuda()
#试一下rospop
optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=args.lr_dis, betas=(args.beta1, 0.999))
optimizer_g = torch.optim.Adam(model_g.parameters(), lr=args.lr_g, betas=(args.beta1, 0.999))
origin_data, origin_label = next(iter(testloader))
origin_data, origin_label = origin_data.cuda(), origin_label.cuda()
save_image(origin_data, results_pic_root + "/origin_data" + ".jpg")


def show_ae(model_g, loader):
    '''
    观察重构图像
    :param model_g:
    :param loader:
    :return:
    '''
    for batch, (data, label) in enumerate(loader):
        data = data.cuda()
        label = label.cuda()
        decoded = model_g(data)
        data_all = torch.concat([data, decoded], dim=0)
        img = torchvision.utils.make_grid(data_all)
        img = img.detach().cpu().numpy()
        img = np.transpose(img, [1, 2, 0])
        plt.figure()
        plt.imshow(img)
        plt.show()


# 训练
def train_miao_code():
    '''
    照抄苗师兄代码
    wloss和crossentropyloss训练ae生成fake样本的能力
    :return:
    '''
    min_test = 1.7

    one = torch.FloatTensor([1])
    mone = one * -1
    one, mone = one.cuda(), mone.cuda()
    for epoch in range(args.epochs):
        loss_all_discriminator_real = 0  # 分类器在正常样本的loss,对正常样本识别正确的概率
        loss_all_discriminator_vir = 0  # 分类器在虚假样本的loss,对虚假样本识别正确的概率
        loss_all_w = 0  # 整体的wloss,希望我们生成的virtual的wloss越低越好,说明越像真实样本分布
        loss_all_v = 0  # 整体的virtual example的crossentropyloss
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs = inputs.cuda()
            targets = targets.cuda()

            # 更新鉴别器
            discriminator.zero_grad()
            real_cpu = inputs.cuda()
            output = discriminator(real_cpu.detach()).reshape(-1)  # 鉴别器预测真实图像
            loss_all_discriminator_real += output.sigmoid().mean().item()
            label_real = torch.zeros(output.shape).cuda()

            d_loss_real = criterion_bce(output, label_real)

            inputs_adv = model_g.module.generate_virtual(inputs, set_detach=True)  # encoder之后设置detach
            inputs_adv = inputs_adv.cuda()
            decoded = inputs_adv
            output = discriminator(decoded.detach()).reshape(-1)  # 注意detach
            loss_all_discriminator_vir += output.sigmoid().mean().item()
            label_fake = torch.ones(output.shape).cuda()  # fake为1
            d_loss_fake = criterion_bce(output, label_fake)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_dis.step()

            # 更新生成器
            model_g.zero_grad()

            for i in range(args.g_k):  # 不要弄倍数? shengchegqi学习率弄成2倍
                # wloss
                output = discriminator(decoded).reshape(-1)
                label_fake = torch.zeros(output.shape).cuda()
                # g_loss_w = args.w_loss_weight * output
                g_loss_w = criterion_bce(output, label_fake)
                # (g_loss_w * args.w_loss_weight).backward(retain_graph=True)
                # crossentropyloss
                # model_d.enval()#就不eval了,增加随机性
                pred = model_d(decoded)
                virtual_label = F.one_hot(targets, num_classes) / 2
                index_0 = range(0, len(virtual_label), 2)
                index_1 = range(1, len(virtual_label), 2)
                virtual_label = virtual_label[index_0] + virtual_label[index_1]
                virtual_label = virtual_label.detach()
                g_loss_ce = criterion_cel(pred, virtual_label)
                # g_loss_ce.backward()
                g_loss = g_loss_w * args.w_loss_weight + g_loss_ce
                g_loss.backward()
                optimizer_g.step()
            loss_all_w += output.sigmoid().mean().item()
            loss_all_v += g_loss_ce.item()
        loss_all_w /= len(trainloader)
        loss_all_v /= len(trainloader)
        loss_all_discriminator_real /= len(trainloader)
        loss_all_discriminator_vir /= len(trainloader)
        print(
            f"train epoch[{epoch}/{args.epochs}]:wloss={loss_all_w:.8f},virtuacrosslloss={loss_all_v:.8f},loss_all_discriminator_real={loss_all_discriminator_real:.8f},loss_all_discriminator_vir={loss_all_discriminator_vir:.8f}")

        # 测试
        model_g.eval()
        discriminator.eval()
        test_crossentropyloss = 0
        test_wloss = 0
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
                loss = criterion_cel(outputs, virtual_label)
                test_crossentropyloss += loss.item()
                test_wloss += discriminator(virtual_data).sigmoid().mean().item()  # 置信度
        test_wloss /= len(testloader)
        test_crossentropyloss = test_crossentropyloss / len(testloader)
        print(f"test epoch[{epoch}/{args.epochs}] : crossentropyloss={test_crossentropyloss},test_wloss={test_wloss}")
        writer.add_scalars("loss", {"train_wloss": loss_all_w,
                                    "train_entroloss": loss_all_v,
                                    "train_discriminator_real": loss_all_discriminator_real,
                                    "train_discriminator_vir": loss_all_discriminator_vir,
                                    "test_entroloss": test_crossentropyloss,
                                    "test_wloss": test_wloss},
                           epoch)
        # 结尾
        if test_crossentropyloss < min_test:
            min_test = test_crossentropyloss
            state = {"model": model_g.state_dict(), "loss": test_crossentropyloss, "epoch": epoch}
            torch.save(state,
                       results_root + f"/ae_miao__entroloss{test_crossentropyloss:.2f}__wloss{test_wloss}__lr_g{args.lr_g}__lr_dis{args.lr_dis}__w_weight{args.w_loss_weight}__epoch{epoch}.pth")
        save_image(virtual_data, results_pic_root + f"/fakeimg__epoch{epoch}__wloss{loss_all_w:.2f}.jpg")


def train_only_wgan():
    '''
    '''
    results_root = "../results/train_only_wgan"
    os.makedirs(results_root, exist_ok=True)
    one = torch.FloatTensor([1])
    mone = one * -1
    one, mone = one.cuda(), mone.cuda()
    args.epochs = 200
    for epoch in range(args.epochs):
        loss_all_discriminator_real = 0  # 分类器在正常样本的loss,对正常样本识别正确的概率
        loss_all_discriminator_vir = 0  # 分类器在虚假样本的loss,对虚假样本识别正确的概率
        loss_all_w = 0  # 整体的wloss,希望我们生成的virtual的wloss越低越好,说明越像真实样本分布
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            discriminator.zero_grad()
            real_cpu = inputs.cuda()
            output = discriminator(real_cpu)  # 鉴别器预测真实图像
            d_loss_real = output
            d_loss_real.backward(one)  # real为0,所以d_loss_real越小越好

            inputs_adv = model_g.module.generate_virtual(inputs, set_detach=True)  # encoder之后设置detach
            inputs_adv = inputs_adv.cuda()
            decoded = inputs_adv
            output = discriminator(decoded.detach())  # 注意detach
            d_loss_fake = output
            d_loss_fake.backward(mone)  # fake为1,所以backward的参数为-1 代表希望预测输出越大越好
            optimizer_dis.step()

            model_g.zero_grad()

            loss_all_discriminator_real += d_loss_real.item()
            loss_all_discriminator_vir += d_loss_fake.item()

            for i in range(args.g_k):  # 不要弄倍数? shengchegqi学习率弄成2倍
                # wloss
                output = discriminator(decoded)
                g_loss_w = output
                g_loss_w.backward(one)

                optimizer_g.step()
            loss_all_w += output.item()
        loss_all_w /= len(trainloader)
        loss_all_discriminator_real /= len(trainloader)
        loss_all_discriminator_vir /= len(trainloader)
        print(
            f"train epoch[{epoch}/{args.epochs}]:wloss={loss_all_w:.8f}mloss_all_discriminator_real={loss_all_discriminator_real:.8f},loss_all_discriminator_vir={loss_all_discriminator_vir:.8f}")

        # 测试
        model_g.eval()
        discriminator.eval()
        test_wloss = 0
        for batch, (data, label) in enumerate(testloader):
            with torch.no_grad():
                data = data.cuda()
                label = label.cuda()
                virtual_data = model_g.module.generate_virtual(data)
                outputs = model_d(virtual_data)
                test_wloss += discriminator(virtual_data).item()
        test_wloss /= len(testloader)
        print(f"test epoch[{epoch}/{args.epochs}] : test_wloss={test_wloss}")

        # 结尾
        state = {"model": model_g.state_dict(), "epoch": epoch}
        torch.save(state,
                   results_root + f"/ae_miao____epoch{epoch}__lr_g{args.lr_g}__lr_dis{args.lr_dis}__w_weight{args.w_loss_weight}.pth")


if __name__ == "__main__":
    train_miao_code()
    # train_only_wgan()
