'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
import torch
import torchvision.transforms as transforms

# 数据预处理

# 尽量只用这两个
mean_cifar = [x / 255 for x in [125.3, 123.0, 113.9]]
std_cifar = [x / 255 for x in [63.0, 62.1, 66.7]]
transform_train_cifar_miao = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    # 苗师兄源代码归一化注释掉了
    transforms.Normalize(mean_cifar, std_cifar),
])

transform_test_cifar_miao = transforms.Compose([
    transforms.ToTensor(),
    # 苗师兄源代码归一化注释掉了
    transforms.Normalize(mean_cifar, std_cifar),
])

transform_only_tensor = transforms.Compose(
    [transforms.ToTensor()])
transform_only_tensor_train = transforms.Compose(
    [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
transform_tensor_norm = transforms.Compose(
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


def get_fixed_parameters(id, model="miao"):
    '''
    model=="mine"|"miao"
    1 只有encoder,即解放t2t3t4
    2 除了两个z合并的那一层tconv4,其他都返回,只解放t4
    3 解放tconv4 tconv2
    4.解放 tconv4 tonv3
    model=="miao
    1:只学习两个z合到一起的ct0
    :return: str返回需要固定参数的名称
    '''
    parameters = ""
    if model == "mine":
        if id == 1:
            parameters = "conv1.0.weight conv1.0.bias conv1.1.weight conv1.1.bias conv2.0.weight conv2.0.bias conv2.1.weight conv2.1.bias conv3.0.weight conv3.0.bias conv3.1.weight conv3.1.bias "
        elif id == 2:
            parameters = "conv1.0.weight conv1.0.bias conv1.1.weight conv1.1.bias conv2.0.weight conv2.0.bias conv2.1.weight conv2.1.bias conv3.0.weight conv3.0.bias conv3.1.weight conv3.1.bias tconv1.0.weight tconv1.0.bias tconv1.1.weight tconv1.1.bias tconv2.0.weight tconv2.0.bias tconv2.1.weight tconv2.1.bias tconv3.0.weight tconv3.0.bias tconv3.1.weight tconv3.1.bias"
        elif id == 3:
            parameters = "conv1.0.weight conv1.0.bias conv1.1.weight conv1.1.bias conv2.0.weight conv2.0.bias conv2.1.weight conv2.1.bias conv3.0.weight conv3.0.bias conv3.1.weight conv3.1.bias tconv1.0.weight tconv1.0.bias tconv1.1.weight tconv1.1.bias tconv3.0.weight tconv3.0.bias tconv3.1.weight tconv3.1.bias"
        elif id == 4:
            parameters = "conv1.0.weight conv1.0.bias conv1.1.weight conv1.1.bias conv2.0.weight conv2.0.bias conv2.1.weight conv2.1.bias conv3.0.weight conv3.0.bias conv3.1.weight conv3.1.bias tconv1.0.weight tconv1.0.bias tconv1.1.weight tconv1.1.bias tconv2.0.weight tconv2.0.bias tconv2.1.weight tconv2.1.bias"
    elif model == "miao":
        if id == 1:
            l = ['conv1.0.weight', 'conv1.0.bias', 'conv2.0.weight', 'conv2.0.bias', 'conv3.0.weight', 'conv3.0.bias',
                 'conv4.0.weight', 'conv4.0.bias', 'ct1.0.weight', 'ct1.0.bias', 'ct2.0.weight',
                 'ct2.0.bias', 'ct3.0.weight', 'ct3.0.bias', 'ct4.0.weight', 'ct4.0.bias', 'ct5.0.weight', 'ct5.0.bias',
                 'ct6.0.weight', 'ct6.0.bias', 'ct7.0.weight', 'ct7.0.bias', 'ct8.0.weight', 'ct8.0.bias']
            parameters = " ".join(l)
    else:
        print("get_fixed_parameters获得模型参数失败")
        exit(0)
    return parameters


def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        print("morm了")
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        return img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None]).detach()
    else:
        return img_tensor


# 调整lr
def adjust_learning_rate(optimizer, lr):
    '''
    调整学习率
    :param optimizer:
    :param lr:
    :return:
    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def weights_init(m):
    '''
    权重初始化
    :param m:
    :return:
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


'''_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time'''


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


# 最初处理
def get_args(args):
    '''
    :param args:
    :return: 字符类型的args,且替换掉特殊字符
    '''
    return str(args).replace("(", "").replace(")", "").replace(",", "--").replace(" ", "").replace("=", "")


# 压制训练用
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
