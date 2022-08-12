import torch
import torchvision
from utils import transform_only_tensor
from torch.utils.data import DataLoader, Dataset


def downloadDatasets():
    cifar100_train = torchvision.datasets.CIFAR100(root="../data/cifar100", train=True, download=True,
                                                   transform=transform_only_tensor)
    cifar100_test = torchvision.datasets.CIFAR100(root="../data/cifar100", train=False, download=True,
                                                  transform=transform_only_tensor)
    svhn_train = torchvision.datasets.SVHN(root="../data/svhn", split="train", download=True,
                                           transform=transform_only_tensor)
    svhn_test = torchvision.datasets.SVHN(root="../data/svhn", split="test", download=True,
                                          transform=transform_only_tensor)


def chakantupian():
    cifar100_train = torchvision.datasets.CIFAR100(root="../data/cifar100", train=True, download=False,
                                                   transform=transform_only_tensor)
    cifar100_test = torchvision.datasets.CIFAR100(root="../data/cifar100", train=False, download=False,
                                                  transform=transform_only_tensor)
    svhn_train = torchvision.datasets.SVHN(root="../data/svhn", split="train", download=False,
                                           transform=transform_only_tensor)
    svhn_test = torchvision.datasets.SVHN(root="../data/svhn", split="test", download=False,
                                          transform=transform_only_tensor)
    print(svhn_test.data.shape)


def downloade_LSUN():
    lsun_train = torchvision.datasets.LSUN(root="../data/lsun", classes="train", transform=transform_only_tensor)
    lsun_test = torchvision.datasets.LSUN(root="../data/lsun", classes="test", transform=transform_only_tensor)
    trainloader_lsun = DataLoader(lsun_train, batch_size=128, shuffle=True, num_workers=4)
    for data, label in trainloader_lsun:
        print(data.shape, label.shape)


if __name__ == "__main__":
    pass
    # downloadDatasets()#下载数据集
    # chakantupian()
    downloade_LSUN()
