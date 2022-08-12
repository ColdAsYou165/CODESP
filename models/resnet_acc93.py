'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    '''
    训练的效果不错,resnet18 epoch200时候acc=0.956,loss=0.1652
    '''

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def getResNet(name):
    '''
    :param name: 想要的resnet版本183450 101 152
    :return: 返回相应的resnet

    '''
    if "18" in name:
        return ResNet18()
    elif "34" in name:
        return ResNet34()
    elif "50" in name:
        return ResNet50()
    elif "101" in name:
        return ResNet101()
    elif "152" in name:
        return ResNet152()
    else:
        print(f"error,没有{name}模型")
        exit(1)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


class AutoEncoder_origin(nn.Module):
    '''
    原始
    cifar10 ae
    https://github.com/chenjie/PyTorch-CIFAR-10-autoencoder/blob/master/main.py
    效果如何不清楚
    '''

    def __init__(self):
        # 需要一个正常的重构损失,表示能将原始图像还原
        # 需要disc target两个标签的值要平均,一个loss
        # 两个loss相加
        # 还可以加一个chamferloss
        super(AutoEncoder_origin, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            # 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
            #             nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            #             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            #             nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class AutoEncoder(nn.Module):
    '''
    自己写的,可以将两张图像何在一起
    目前是相邻两张图像合在一起,即随机何在一起,并没有判断两张图像target是否相同
    cifar10 ae
    https://github.com/chenjie/PyTorch-CIFAR-10-autoencoder/blob/master/main.py
    效果如何不清楚
    '''

    def __init__(self):
        # 需要一个正常的重构损失,表示能将原始图像还原
        # 需要disc target两个标签的值要平均,一个loss
        # 两个loss相加
        # 还可以加一个chamferloss
        super(AutoEncoder, self).__init__()
        # encoder Input size: [batch, 3, 32, 32]
        self.conv1 = nn.Sequential(nn.Conv2d(3, 12, 4, stride=2, padding=1), nn.ReLU())  # [batch, 12, 16, 16]
        self.conv2 = nn.Sequential(nn.Conv2d(12, 24, 4, stride=2, padding=1), nn.ReLU())  # [batch, 24, 8, 8]
        self.conv3 = nn.Sequential(nn.Conv2d(24, 48, 4, stride=2, padding=1), nn.ReLU())  # [batch, 48, 4, 4]
        '''nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]'''
        # decoder [batch, 48, 4, 4]->[batch, 3, 32, 32]
        self.tconv1 = nn.Sequential(nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1), nn.ReLU())  # [batch, 24, 8, 8]
        self.tconv2 = nn.Sequential(nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
                                    nn.ReLU())  # [batch, 12, 16, 16]
        self.tconv3 = nn.Sequential(nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
                                    nn.Sigmoid())  # [batch, 3, 32, 32]

        # 两张vector合在一起的[batch, 96, 4, 4] -> [batch, 24, 8, 8],代替tconv1
        self.tconv4 = nn.Sequential(nn.ConvTranspose2d(96, 24, 4, stride=2, padding=1), nn.ReLU())  # [batch, 24, 8, 8]

    def encoder(self, x):
        # x[batch, 3, 32, 32]
        x = self.conv1(x)# [batch, 12, 16, 16]
        x = self.conv2(x)# [batch, 24, 8, 8]
        x = self.conv3(x)# [batch, 48, 4, 4]
        return x

    def decoder(self, z):
        # z[batch, 48, 4, 4]
        z = self.tconv1(z)# [batch, 24, 8, 8]
        z = self.tconv2(z)# [batch, 12, 16, 16]
        z = self.tconv3(z)# [batch, 3, 32, 32]
        return z

    def decoder_virtual(self, double_z):
        # double_z [batch/2, 48*2, 4, 4]
        z = self.tconv4(double_z)# [batch, 24, 8, 8]
        z = self.tconv2(z)# [batch, 12, 16, 16]
        z = self.tconv3(z)# [batch, 3, 32, 32]
        return z

    def generate_virtual(self, x):
        '''
        :param x:一个batch的图像样本,batch必须为整数
        :return: batch/2 个生成的虚假图像
        # 不需要用torch.random.shuffle()因为,我们这样相邻两个合在一起就已经相当于随机拼接了,何况dataloader也有shuffle
        # 不需要顾及z的batch不是偶数的情况,dataset是偶数,batchsize是偶数,所以batch一定是偶数
        '''
        z = self.encoder(x)  # [batch, 48, 4, 4]
        z = z.reshape(-1, z.shape[1] * 2, z.shape[2], z.shape[3])
        virtual = self.decoder_virtual(z)
        return virtual

    # 先不判断标签一致,写一个一个batch里随机混合的吧
    #
    def forward(self, x):
        '''

        :param x:
        :return: encoded, decoded
        '''
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


if __name__ == "__main__":
    # test()
    # getResNet("1")
    model = AutoEncoder()
    x = torch.randn([16, 3, 32, 32])
    '''z = model.encoder(x)
    print(z.shape)
    y = model.decoder(z)
    print(y.shape)'''
    v=model.generate_virtual(x)
    print(v.shape)
