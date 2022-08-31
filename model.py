'''
针对cifar10数据集
resnet
autoencoder
'''
'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
'''
        # 原版添加genenrate的苗师兄的ae
        super(AutoEncoder_Miao, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 16 32 32
        self.conv1 = nn.Sequential(nn.Conv2d(
            in_channels=3,  # input height
            out_channels=16,
            kernel_size=3,  # filter size
            stride=1,  # filter movement/step
            padding=1), nn.LeakyReLU())
        # 32 16 16
        self.conv2 = nn.Sequential(nn.Conv2d(
            in_channels=16,  # input height
            out_channels=32,  # n_filters
            kernel_size=3,  # filter size
            stride=1,  # filter movement/step
            padding=1,
        ),
            nn.LeakyReLU(),  # activation
            nn.MaxPool2d(kernel_size=2))
        # 32 16 16
        self.conv3 = nn.Sequential(nn.Conv2d(
            in_channels=32,  # input height
            out_channels=32,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU())  # activation)
        # 64 8 8
        self.conv4 = nn.Sequential(nn.Conv2d(
            in_channels=32,  # input height
            out_channels=64,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU(),  # activation
            nn.MaxPool2d(kernel_size=2))
        # decoder
        # 128 8 8 -> 32 16 16
        self.ct0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=2, stride=2, padding=0), nn.LeakyReLU())
        # 64 8 8 -> 32 16 16
        self.ct1 = nn.Sequential(nn.ConvTranspose2d(
            in_channels=64,  # input height
            out_channels=32,  # n_filters
            kernel_size=2,  # filter size
            stride=2,  # filter movement/step
            padding=0,
        ),
            nn.LeakyReLU())
        #
        self.ct2 = nn.Sequential(nn.Conv2d(
            in_channels=32,  # input height
            out_channels=32,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU())
        # -> 16 16 16
        self.ct3 = nn.Sequential(nn.ConvTranspose2d(
            in_channels=32,  # input height
            out_channels=16,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ))
        self.ct4 = nn.Sequential(nn.Conv2d(
            in_channels=16,  # input height
            out_channels=16,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU())
        # ->16 32 32
        self.ct5 = nn.Sequential(nn.ConvTranspose2d(
            in_channels=16,  # input height
            out_channels=16,  # n_filters
            kernel_size=2,  # filter size
            stride=2,  # filter movement/step
            padding=0,
        ),
            nn.LeakyReLU())
        self.ct6 = nn.Sequential(nn.Conv2d(
            in_channels=16,  # input height
            out_channels=16,  # n_filters
            kernel_size=3,  # filter size
            stride=1,  # filter movement/step
            padding=1,
        ),
            nn.LeakyReLU())
        # -> 3 32 32
        self.ct7 = nn.Sequential(nn.ConvTranspose2d(
            in_channels=16,  # input height
            out_channels=3,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU())
        self.ct8 = nn.Sequential(nn.Conv2d(
            in_channels=3,  # input height
            out_channels=3,  # n_filters
            kernel_size=3,  # filter size
            stride=1,  # filter movement/step
            padding=1,
        ))
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
    crossentropyloss自带softmax所以模型内部不用接softmax
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


def ResNet18_sigmoid(set_sigmoid=True):
    '''
    得到ResNet18_sigmoid
    :return:
    '''
    return ResNet_sigmoid(BasicBlock, [2, 2, 2, 2], set_sigmoid=set_sigmoid)


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
    起初实验都是用的这个,现在老师说这个结构不好,我试试苗师兄的
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
        self.conv1 = nn.Sequential(nn.Conv2d(3, 12, 4, stride=2, padding=1), nn.BatchNorm2d(12),
                                   nn.ReLU())  # [batch, 12, 16, 16]
        self.conv2 = nn.Sequential(nn.Conv2d(12, 24, 4, stride=2, padding=1), nn.BatchNorm2d(24),
                                   nn.ReLU())  # [batch, 24, 8, 8]
        self.conv3 = nn.Sequential(nn.Conv2d(24, 48, 4, stride=2, padding=1), nn.BatchNorm2d(48),
                                   nn.ReLU())  # [batch, 48, 4, 4]
        '''nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]'''
        # decoder [batch, 48, 4, 4]->[batch, 3, 32, 32]
        self.tconv1 = nn.Sequential(nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1), nn.BatchNorm2d(24),
                                    nn.ReLU())  # [batch, 24, 8, 8]
        self.tconv2 = nn.Sequential(nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1), nn.BatchNorm2d(12),
                                    nn.ReLU())  # [batch, 12, 16, 16]
        self.tconv3 = nn.Sequential(nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1), nn.BatchNorm2d(3),
                                    nn.Sigmoid())  # [batch, 3, 32, 32]

        # 两张vector合在一起的[batch, 96, 4, 4] -> [batch, 24, 8, 8],代替tconv1
        self.tconv4 = nn.Sequential(nn.ConvTranspose2d(96, 24, 4, stride=2, padding=1),
                                    nn.ReLU())  # [batch, 24, 8, 8]

    def encoder(self, x):
        # x[batch, 3, 32, 32]
        x = self.conv1(x)  # [batch, 12, 16, 16]
        x = self.conv2(x)  # [batch, 24, 8, 8]
        x = self.conv3(x)  # [batch, 48, 4, 4]
        return x

    def decoder(self, z):
        # z[batch, 48, 4, 4]
        z = self.tconv1(z)  # [batch, 24, 8, 8]
        z = self.tconv2(z)  # [batch, 12, 16, 16]
        z = self.tconv3(z)  # [batch, 3, 32, 32]
        return z

    def decoder_virtual(self, double_z):
        # double_z [batch/2, 48*2, 4, 4]
        z = self.tconv4(double_z)  # [batch, 24, 8, 8]
        z = self.tconv2(z)  # [batch, 12, 16, 16]
        z = self.tconv3(z)  # [batch, 3, 32, 32]
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
        :return: encodeddecoded
        '''
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoEncoder_origin_Miaoshixiong(nn.Module):
    '''
    原版苗师兄CODEs使用的ae
    改类名记得改supper
    '''

    def __init__(self):
        super(AutoEncoder_origin_Miaoshixiong, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3,  # input height
                out_channels=16,
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),
            nn.LeakyReLU(),  # activation
            nn.Conv2d(
                in_channels=16,  # input height
                out_channels=32,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),
            nn.LeakyReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=32,  # input height
                out_channels=32,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.LeakyReLU(),  # activation
            nn.Conv2d(
                in_channels=32,  # input height
                out_channels=64,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.LeakyReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,  # input height
                out_channels=32,  # n_filters
                kernel_size=2,  # filter size
                stride=2,  # filter movement/step
                padding=0,
            ),
            nn.LeakyReLU(),  # activation
            nn.Conv2d(
                in_channels=32,  # input height
                out_channels=32,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.LeakyReLU(),  # activation
            nn.ConvTranspose2d(
                in_channels=32,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.Conv2d(
                in_channels=16,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.LeakyReLU(),  # activation
            nn.ConvTranspose2d(
                in_channels=16,  # input height
                out_channels=16,  # n_filters
                kernel_size=2,  # filter size
                stride=2,  # filter movement/step
                padding=0,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=16,  # input height
                out_channels=16,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),
            nn.LeakyReLU(),  # activation
            nn.ConvTranspose2d(
                in_channels=16,  # input height
                out_channels=3,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=3,  # input height
                out_channels=3,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),
            # nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoEncoder_Miao(nn.Module):
    '''
    改版自苗师兄的ae,不出意外,以后用这个

    注意:他这个没有加batchnorm,我不知道我要不要加上,但是不加的话,输出范围Wie[-0.1,1.1]超过[0,1]了
    '''

    def __init__(self):
        super(AutoEncoder_Miao, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 16 32 32
        self.conv1 = nn.Sequential(nn.Conv2d(
            in_channels=3,  # input height
            out_channels=16,
            kernel_size=3,  # filter size
            stride=1,  # filter movement/step
            padding=1), nn.LeakyReLU())
        # 32 16 16
        self.conv2 = nn.Sequential(nn.Conv2d(
            in_channels=16,  # input height
            out_channels=32,  # n_filters
            kernel_size=3,  # filter size
            stride=1,  # filter movement/step
            padding=1,
        ),
            nn.LeakyReLU(),  # activation
            nn.MaxPool2d(kernel_size=2))
        # 32 16 16
        self.conv3 = nn.Sequential(nn.Conv2d(
            in_channels=32,  # input height
            out_channels=32,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU())  # activation)
        # 64 8 8
        self.conv4 = nn.Sequential(nn.Conv2d(
            in_channels=32,  # input height
            out_channels=64,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU(),  # activation
            nn.MaxPool2d(kernel_size=2))
        # decoder
        # 128 8 8 -> 32 16 16
        self.ct0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=2, stride=2, padding=0), nn.LeakyReLU())
        # 64 8 8 -> 32 16 16
        self.ct1 = nn.Sequential(nn.ConvTranspose2d(
            in_channels=64,  # input height
            out_channels=32,  # n_filters
            kernel_size=2,  # filter size
            stride=2,  # filter movement/step
            padding=0,
        ),
            nn.LeakyReLU())
        #
        self.ct2 = nn.Sequential(nn.Conv2d(
            in_channels=32,  # input height
            out_channels=32,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU())
        # -> 16 16 16
        self.ct3 = nn.Sequential(nn.ConvTranspose2d(
            in_channels=32,  # input height
            out_channels=16,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ))
        self.ct4 = nn.Sequential(nn.Conv2d(
            in_channels=16,  # input height
            out_channels=16,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU())
        # ->16 32 32
        self.ct5 = nn.Sequential(nn.ConvTranspose2d(
            in_channels=16,  # input height
            out_channels=16,  # n_filters
            kernel_size=2,  # filter size
            stride=2,  # filter movement/step
            padding=0,
        ),
            nn.LeakyReLU())
        self.ct6 = nn.Sequential(nn.Conv2d(
            in_channels=16,  # input height
            out_channels=16,  # n_filters
            kernel_size=3,  # filter size
            stride=1,  # filter movement/step
            padding=1,
        ),
            nn.LeakyReLU())
        # -> 3 32 32
        self.ct7 = nn.Sequential(nn.ConvTranspose2d(
            in_channels=16,  # input height
            out_channels=3,  # n_filters
            kernel_size=5,  # filter size
            stride=1,  # filter movement/step
            padding=2,
        ),
            nn.LeakyReLU())
        self.ct8 = nn.Sequential(nn.Conv2d(
            in_channels=3,  # input height
            out_channels=3,  # n_filters
            kernel_size=3,  # filter size
            stride=1,  # filter movement/step
            padding=1,
        ), nn.Sigmoid())

    def encoder(self, x):
        x = self.conv1(x)  # 16,32,32
        x = self.conv2(x)  # 32,16,16
        x = self.conv3(x)  # 32,16,16
        x = self.conv4(x)  # 64,8,8
        return x

    def decoder(self, x):
        # 64,8,8
        x = self.ct1(x)  # 32,16,16
        x = self.ct2(x)
        x = self.ct3(x)  # 16,16,16
        x = self.ct4(x)
        x = self.ct5(x)  # 16,32,32
        x = self.ct6(x)
        x = self.ct7(x)  # 3,32,32
        x = self.ct8(x)
        return x

    def decoder_virtual(self, double_z):
        # double_z [batch/2, 64*2, 8, 8]
        x = self.ct0(double_z)  # 32 16 16
        x = self.ct2(x)
        x = self.ct3(x)  # 16,16,16
        x = self.ct4(x)
        x = self.ct5(x)  # 16,32,32
        x = self.ct6(x)
        x = self.ct7(x)  # 3,32,32
        x = self.ct8(x)
        return x

    def generate_virtual(self, x, set_encoded_detach=False):
        '''
        :param x:一个batch的图像样本,batch必须为整数
        :return: batch/2 个生成的虚假图像
        # 不需要用torch.random.shuffle()因为,我们这样相邻两个合在一起就已经相当于随机拼接了,何况dataloader也有shuffle
        # 不需要顾及z的batch不是偶数的情况,dataset是偶数,batchsize是偶数,所以batch一定是偶数
        '''
        z = self.encoder(x)  # [batch, 64, 8, 8]
        z = z.reshape(-1, z.shape[1] * 2, z.shape[2], z.shape[3])
        if set_encoded_detach:
            z = z.detach()
        virtual = self.decoder_virtual(z)
        return virtual

    def generate_virtual_by_add(self, x):
        pass

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ResNet_sigmoid(nn.Module):
    '''
    ResNet_sigmoid
    不是,训练的效果不错,resnet18 epoch200时候acc=0.956,loss=0.1652
    '''

    def __init__(self, block, num_blocks, num_classes=10, set_sigmoid=True):
        '''
        :param block:
        :param num_blocks:
        :param num_classes:
        :param set_sigmoid: 设置是否需要sigmoid
        '''
        super(ResNet_sigmoid, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, 1)
        self.set_sigmoid = set_sigmoid
        self.sigmoid = torch.nn.Sigmoid()

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
        if self.set_sigmoid:
            out = self.sigmoid(out)
        return out


class Discriminator_WGAN_miao_cifar10(nn.Module):
    '''
    苗师兄wgan的discriminator,用于数据集 cifar10,cifar100,svhn
    这个没有接sigmoid,很是奇怪,mnist接sigmoid了 --- 实验发现,接了sigmoid要好一些.
    --实验结果: 效果不好,猜测是discriminator太强了.
    '''

    def __init__(self, set_sigmoid=False, ngpu=1):
        '''
        :param set_sigmoid: 最后的输出是否接sigmoid
        '''
        super(Discriminator_WGAN_miao_cifar10, self).__init__()
        nc = 3
        ndf = 32
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64  32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32  16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8 4
            # nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),

        )
        self.set_sigmoid = set_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        output = self.main(input)
        if self.set_sigmoid:
            output = self.sigmoid(output)
        # print('-----------------------------')
        # print(output.shape)
        output = output.mean(0)
        # return output.view(-1, 1).squeeze(1)
        return output.view(1)


class simple_ae(nn.Module):
    '''
    wgan调通了的生成器
    '''

    def __init__(self):
        latent_size = 64
        n_channel = 3
        n_g_feature = 64
        super(simple_ae, self).__init__()
        # encoder
        self.c1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.c2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.c3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.c4 = nn.Sequential(nn.Conv2d(256, 64, 4, 1, 0), nn.BatchNorm2d(64), nn.ReLU())

        # dncoder
        self.ct1 = nn.Sequential(nn.ConvTranspose2d(64, 4 * 64, kernel_size=4, bias=False),
                                 nn.BatchNorm2d(4 * 64),
                                 nn.ReLU())
        self.ct2 = nn.Sequential(nn.ConvTranspose2d(4 * 64, 2 * 64, kernel_size=4, stride=2, padding=1, bias=False),
                                 nn.BatchNorm2d(2 * 64),
                                 nn.ReLU())
        self.ct3 = nn.Sequential(nn.ConvTranspose2d(2 * 64, 64, kernel_size=4, stride=2, padding=1, bias=False),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU())
        self.ct4 = nn.Sequential(nn.ConvTranspose2d(64, n_channel, kernel_size=4, stride=2, padding=1),
                                 nn.Sigmoid())

    def encoder(self, x):
        x = self.c1(x)  # 64, 16, 16
        x = self.c2(x)  # 128, 8, 8
        x = self.c3(x)  # 256, 4, 4
        x = self.c4(x)  # 64, 1, 1
        return x

    def decoder(self, z):
        # z [b,latensize=64,1,1]
        z = self.ct1(z)  # 256 4 4
        z = self.ct2(z)  # 128, 8, 8
        z = self.ct3(z)  # 64, 16, 16
        z = self.ct4(z)  # 3, 32, 32
        return z

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class simple_discriminator(nn.Module):
    '''
    简单discriminator,源码没有接sigmoid,我尝试加sigmoid之后直接完蛋.
    '''
    def __init__(self):
        super(simple_discriminator, self).__init__()
        n_d_feature = 64
        n_channel = 3
        self.dis = nn.Sequential(
            nn.Conv2d(n_channel, n_d_feature, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_d_feature, 2 * n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * n_d_feature),
            nn.LeakyReLU(0.2),

            nn.Conv2d(2 * n_d_feature, 4 * n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * n_d_feature),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4 * n_d_feature, 1, kernel_size=4)
        )

    def forward(self, x):
        x = self.dis(x)
        return x


if __name__ == "__main__":
    # test()
    # getResNet("1")
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # model = simple_ae().cuda()
    # for batch in range(1):
    #     x = torch.randn([16, 3, 32, 32]).cuda()
    #     pred = model.encoder(x)
    #     print(pred.shape)
    #     decoded = model.decoder(pred)
    #     print(decoded.shape)
    model=simple_discriminator()
    x=torch.randn([32,3,32,32])
    y=model(x).reshape(-1)
    print(y.shape)