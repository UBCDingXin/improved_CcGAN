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
GEN_SIZE=64
DISC_SIZE=64

class ResBlockGenerator(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=True)
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

        self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0, bias=True) #h=h
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)
        self.bypass = nn.Sequential(
            nn.Upsample(scale_factor=2),
            self.bypass_conv,
        )

    def forward(self, x):
        out = self.model(x) + self.bypass(x)
        return out



class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, bias=True):
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

    def __init__(self, in_channels, out_channels, stride=1, bias=True):
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



class SNGAN_Generator(nn.Module):
    def __init__(self, dim_z=128, dim_c=1):
        super(SNGAN_Generator, self).__init__()
        self.dim_z = dim_z
        self.dim_c = dim_c

        self.dense = nn.Linear(self.dim_z+self.dim_c, 4 * 4 * GEN_SIZE*16, bias=True)
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        self.genblock0 = ResBlockGenerator(GEN_SIZE*16, GEN_SIZE*8) #4--->8
        self.genblock1 = ResBlockGenerator(GEN_SIZE*8, GEN_SIZE*4) #8--->16
        self.genblock2 = ResBlockGenerator(GEN_SIZE*4, GEN_SIZE*2) #16--->32
        self.genblock3 = ResBlockGenerator(GEN_SIZE*2, GEN_SIZE) #32--->64

        self.final = nn.Sequential(
            nn.BatchNorm2d(GEN_SIZE),
            nn.ReLU(),
            self.final,
            nn.Tanh()
        )

    def forward(self, z, y): #y is embedded in the feature space
        z = z.view(z.size(0), z.size(1))
        y = y.view(y.size(0), 1)
        out = self.dense(torch.cat((z,y),dim=1))
        out = out.view(-1, GEN_SIZE*16, 4, 4)

        out = self.genblock0(out)
        out = self.genblock1(out)
        out = self.genblock2(out)
        out = self.genblock3(out)
        out = self.final(out)

        return out


class SNGAN_Discriminator(nn.Module):
    def __init__(self, dim_c=1):
        super(SNGAN_Discriminator, self).__init__()
        self.dim_c=dim_c

        self.discblock = nn.Sequential(
            FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2, bias=True), #64--->32
            ResBlockDiscriminator(DISC_SIZE , DISC_SIZE*2, stride=2, bias=True), #32--->16
            ResBlockDiscriminator(DISC_SIZE*2  , DISC_SIZE*4, stride=2, bias=True), #16--->8
            ResBlockDiscriminator(DISC_SIZE*4, DISC_SIZE*8, stride=2, bias=True), #8--->4
            ResBlockDiscriminator(DISC_SIZE*8, DISC_SIZE*16, stride=1, bias=True), #4--->4
            nn.ReLU()
        )

        self.linear = nn.Linear(DISC_SIZE*16*4*4+dim_c, 1)
        nn.init.xavier_uniform_(self.linear.weight.data, 1.)
        self.linear = spectral_norm(self.linear)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        y = y.view(-1,1)
        output = self.discblock(x)

        # output = torch.sum(output, dim=(2, 3))
        output = output.view(-1,DISC_SIZE*16*4*4)
        output = self.linear(torch.cat((output,y),dim=1))
        output = self.sigmoid(output)
        return output.view(-1, 1)



if __name__ == "__main__":
    netG = SNGAN_Generator(dim_z=128, dim_c=1).cuda()
    netD = SNGAN_Discriminator(dim_c=1).cuda()
    z = torch.randn(16, 128).cuda()
    c = torch.randn(16, 1).cuda()
    out_G = netG(z,c)
    out_D = netD(out_G,c)
    print(out_G.size())
    print(out_D.size())