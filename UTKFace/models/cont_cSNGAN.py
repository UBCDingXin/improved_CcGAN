'''
https://github.com/christiancosgrove/pytorch-spectral-normalization-gan

chainer: https://github.com/pfnet-research/sngan_projection
'''

# ResNet generator and discriminator
import torch
from torch import nn
import torch.nn.functional as F

# from spectral_normalization import SpectralNorm
import numpy as np
from torch.nn.utils import spectral_norm



channels = 3
bias = False

class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=bias)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )

        self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0) #h=h
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)
        self.bypass = nn.Sequential(
            nn.Upsample(scale_factor=2),
            self.bypass_conv,
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=bias)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                spectral_norm(self.conv1),
                nn.ReLU(),
                spectral_norm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                spectral_norm(self.conv1),
                nn.ReLU(),
                spectral_norm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )

        self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0, bias=bias)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)
        if stride != 1:
            self.bypass = nn.Sequential(
                spectral_norm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
        else:
            self.bypass = nn.Sequential(
                spectral_norm(self.bypass_conv),
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=bias)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0, bias=bias)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            spectral_norm(self.conv1),
            nn.ReLU(),
            spectral_norm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            spectral_norm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

GEN_SIZE=64
DISC_SIZE=64

class cont_cSNGAN_Generator(nn.Module):
    def __init__(self, z_dim):
        super(cont_cSNGAN_Generator, self).__init__()
        self.z_dim = z_dim

        self.dense = nn.Linear(self.z_dim, 4 * 4 * (GEN_SIZE*8), bias=bias)
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1, bias=bias)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        self.genblock0 = nn.Sequential(
            # state size: 4 x 4
            nn.ConvTranspose2d(GEN_SIZE * 8, GEN_SIZE * 4, kernel_size=4, stride=2, padding=1, bias=bias), #h=2h #4--->8
            nn.BatchNorm2d(GEN_SIZE * 4),
            nn.ReLU(True),
            # state size. 8 x 8
        )

        self.genblock1 = ResBlockGenerator((GEN_SIZE*4), (GEN_SIZE*2)) #8--->16

        self.genblock2 = ResBlockGenerator((GEN_SIZE*2), GEN_SIZE) #16--->32

        self.genblock3 = nn.Sequential(
            ResBlockGenerator(GEN_SIZE, GEN_SIZE), #32--->64
            nn.BatchNorm2d(GEN_SIZE),
            nn.ReLU(),
            self.final,
            nn.Tanh()
        )

    def forward(self, z, y):
        z = z.view(z.size(0), z.size(1))
        # y_rep = y.view(-1, 1).repeat(1,self.z_dim)
        # z = z+y_rep

        y_rep = y.view(-1, 1).repeat(1,4 * 4 * (GEN_SIZE*8))
        out = self.dense(z) + y_rep
        out = out.view(-1, (GEN_SIZE*8), 4, 4)

        y_rep = y.view(-1, 1).repeat(1,GEN_SIZE*4*8*8).view(-1, GEN_SIZE*4, 8, 8)
        out = self.genblock0(out) + y_rep

        #y_rep = y.view(-1, 1).repeat(1,GEN_SIZE*2*16*16).view(-1, GEN_SIZE*2, 16, 16)
        out = self.genblock1(out) #+ y_rep

        #y_rep = y.view(-1, 1).repeat(1,GEN_SIZE*32*32).view(-1, GEN_SIZE, 32, 32)
        out = self.genblock2(out) #+ y_rep

        out = self.genblock3(out)

        return out


class cont_cSNGAN_Discriminator(nn.Module):
    def __init__(self):
        super(cont_cSNGAN_Discriminator, self).__init__()

        # self.model = nn.Sequential(
        #         FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2), #64--->32
        #         ResBlockDiscriminator(DISC_SIZE  , DISC_SIZE, stride=2), #32--->16
        #         ResBlockDiscriminator(DISC_SIZE  , DISC_SIZE*2, stride=2), #16--->8
        #         ResBlockDiscriminator(DISC_SIZE*2, DISC_SIZE*4, stride=2), #8--->4
        #         ResBlockDiscriminator(DISC_SIZE*4, DISC_SIZE*8, stride=1), #4--->4;
        #         nn.ReLU(),
        #         #nn.AvgPool2d(2), #6--->3
        #     )


        self.discblock1 = nn.Sequential(
            FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2), #64--->32
            ResBlockDiscriminator(DISC_SIZE  , DISC_SIZE, stride=2), #32--->16
            ResBlockDiscriminator(DISC_SIZE  , DISC_SIZE*2, stride=2), #16--->8
        )
        self.discblock2 = ResBlockDiscriminator(DISC_SIZE*2, DISC_SIZE*4, stride=2) #8--->4
        self.discblock3 = nn.Sequential(
            ResBlockDiscriminator(DISC_SIZE*4, DISC_SIZE*8, stride=1), #4--->4;
            nn.ReLU(),
        )


        self.fc = nn.Linear(DISC_SIZE*8, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = spectral_norm(self.fc)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):

        # y_rep = y.view(-1, 1).repeat(1,DISC_SIZE*8*4*4).view(-1, DISC_SIZE*8, 4, 4)
        # out = self.model(x) + y_rep

        #y_rep = y.view(-1, 1).repeat(1,DISC_SIZE*4*8*8).view(-1, DISC_SIZE*4, 8, 8)
        out = self.discblock1(x) #+ y_rep

        #y_rep = y.view(-1, 1).repeat(1,DISC_SIZE*8*4*4).view(-1, DISC_SIZE*8, 4, 4)
        out = self.discblock2(out) #+ y_rep

        y_rep = y.view(-1, 1).repeat(1,DISC_SIZE*8*4*4).view(-1, DISC_SIZE*8, 4, 4)
        out = self.discblock3(out) + y_rep

        out = torch.sum(out, dim=(2, 3)) # Global pooling
        out = self.fc(out)
        out = self.sigmoid(out)
        return out






if __name__=="__main__":
    #test
    ngpu=1
    device="cuda"

    netG = cont_cSNGAN_Generator(z_dim=128).to(device)
    netD = cont_cSNGAN_Discriminator().to(device)

    z = torch.randn(32, 128).cuda()
    y = np.random.randint(100, 300, 32)
    y = torch.from_numpy(y).type(torch.float).view(-1,1).cuda()
    x = netG(z,y)
    o = netD(x,y)
    print(y.size())
    print(x.size())
    print(o.size())
