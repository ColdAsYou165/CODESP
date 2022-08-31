import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch.nn as nn
import torch.nn.init as init
import torch
import torch.optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.tensorboard import SummaryWriter

writter = SummaryWriter()
results_root = "../results/wgan"
os.makedirs(results_root, exist_ok=True)
os.makedirs(results_root + f'/GAN_saved02', exist_ok=True)
latent_size = 64
n_channel = 3
n_g_feature = 64
gnet = nn.Sequential(
    nn.ConvTranspose2d(latent_size, 4 * n_g_feature, kernel_size=4, bias=False),
    nn.BatchNorm2d(4 * n_g_feature),
    nn.ReLU(),

    nn.ConvTranspose2d(4 * n_g_feature, 2 * n_g_feature, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(2 * n_g_feature),
    nn.ReLU(),

    nn.ConvTranspose2d(2 * n_g_feature, n_g_feature, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(n_g_feature),
    nn.ReLU(),

    nn.ConvTranspose2d(n_g_feature, n_channel, kernel_size=4, stride=2, padding=1),
    nn.Sigmoid()
)

n_d_feature = 64
dnet = nn.Sequential(
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
dnet = dnet.cuda()
gnet = gnet.cuda()


def weights_init(m):
    if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
        init.xavier_normal_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0)


gnet.apply(weights_init)
dnet.apply(weights_init)

dataset = CIFAR10(root="../data/cifar10", train=True, download=False, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

criterion = nn.BCEWithLogitsLoss()
# goptimizer = torch.optim.Adam(gnet.parameters(), lr=0.0002, betas=(0.5, 0.999))
# doptimizer = torch.optim.Adam(dnet.parameters(), lr=0.0002, betas=(0.5, 0.999))
goptimizer = torch.optim.Adam(gnet.parameters(), lr=0.0004, betas=(0.5, 0.999))
doptimizer = torch.optim.Adam(dnet.parameters(), lr=0.0002, betas=(0.5, 0.999))
batch_size = 64
fixed_noises = torch.randn(batch_size, latent_size, 1, 1).cuda()

epoch_num = 50
for epoch in range(epoch_num):
    dloss_epoch = 0
    gloss_epoch = 0
    dmean_real_epoch = 0
    dmean_fake_epoch = 0
    for batch_idx, data in enumerate(dataloader):
        real_images, _ = data
        real_images = real_images.cuda()
        batch_size = real_images.size(0)

        labels = torch.ones(batch_size).cuda()
        preds = dnet(real_images)
        outputs = preds.reshape(-1)
        dloss_real = criterion(outputs, labels)
        dmean_real = outputs.sigmoid().mean()
        # dmean_real = outputs.mean()

        noises = torch.randn(batch_size, latent_size, 1, 1).cuda()
        fake_images = gnet(noises)
        labels = torch.zeros(batch_size).cuda()
        fake = fake_images.detach()

        preds = dnet(fake)
        outputs = preds.view(-1)
        dloss_fake = criterion(outputs, labels)
        dmean_fake = outputs.sigmoid().mean()
        # dmean_fake = outputs.mean()

        dloss = dloss_real + dloss_fake
        dnet.zero_grad()
        dloss.backward()
        doptimizer.step()

        labels = torch.ones(batch_size).cuda()
        preds = dnet(fake_images)
        outputs = preds.view(-1)
        gloss = criterion(outputs, labels)
        gmean_fake = outputs.sigmoid().mean()
        gnet.zero_grad()
        gloss.backward()
        goptimizer.step()

        dloss_epoch += dloss.item()
        gloss_epoch += gloss.item()
        dmean_real_epoch += dmean_real.item()
        dmean_fake_epoch += dmean_fake.item()
    dloss_epoch /= len(dataloader)
    gloss_epoch /= len(dataloader)
    dmean_real_epoch /= len(dataloader)
    dmean_fake_epoch /= len(dataloader)

    fake = gnet(fixed_noises)
    save_image(fake, results_root + f'/GAN_saved02/images_epoch{epoch:02d}__gloss{gloss_epoch:.2f}.png')
    print(f"[{epoch}]/[{epoch_num}]:dloss={dloss_epoch},gloss={gloss_epoch},dmean_real={dmean_real_epoch},dmean_fake={dmean_fake_epoch}")
gnet_save_path = results_root + "/wgan_g.pth"
torch.save(gnet, gnet_save_path)
# gnet = torch.load(gnet_save_path)
# gnet.eval()

dnet_save_path = results_root + "/wgan_d.pth"
torch.save(dnet, dnet_save_path)
# dnet = torch.load(dnet_save_path)
# dnet.eval()


# print(gnet, dnet)
