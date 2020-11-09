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



channels = 1
bias = True
GEN_SIZE=64
DISC_SIZE=64

DIM_EMBED=128



class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, dim_embed):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)

        self.embed_gamma = nn.Linear(dim_embed, num_features, bias=False)
        self.embed_beta = nn.Linear(dim_embed, num_features, bias=False)

    def forward(self, x, y):
        out = self.bn(x)

        gamma = self.embed_gamma(y).view(-1, self.num_features, 1, 1)
        beta = self.embed_beta(y).view(-1, self.num_features, 1, 1)
        out = out + out*gamma + beta

        return out



#########################################################
# genearator
class cont_cond_cnn_generator(nn.Module):
    def __init__(self, nz=128, dim_embed=DIM_EMBED, ngf = GEN_SIZE):
        super(cont_cond_cnn_generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.dim_embed = dim_embed

        self.linear = nn.Linear(nz, 4 * 4 * ngf * 8) #4*4*512

        self.deconv1 = nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=bias) #h=2h 8
        self.deconv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=bias) #h=2h 16
        self.deconv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=bias) #h=2h 32
        self.deconv4 = nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=bias) #h=2h 64
        self.condbn1 = ConditionalBatchNorm2d(ngf * 8, dim_embed)
        self.condbn2 = ConditionalBatchNorm2d(ngf * 4, dim_embed)
        self.condbn3 = ConditionalBatchNorm2d(ngf * 2, dim_embed)
        self.condbn4 = ConditionalBatchNorm2d(ngf, dim_embed)
        self.relu = nn.ReLU()

        self.final_conv = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            nn.Conv2d(ngf, 1, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            nn.Tanh()
        )


    def forward(self, z, y):
        z = z.view(-1, self.nz)

        out = self.linear(z)
        out = out.view(-1, 8*self.ngf, 4, 4)

        out = self.deconv1(out)
        out = self.condbn1(out, y)
        out = self.relu(out)

        out = self.deconv2(out)
        out = self.condbn2(out, y)
        out = self.relu(out)

        out = self.deconv3(out)
        out = self.condbn3(out, y)
        out = self.relu(out)

        out = self.deconv4(out)
        out = self.condbn4(out, y)
        out = self.relu(out)

        out = self.final_conv(out)

        return out

#########################################################
# discriminator
class cont_cond_cnn_discriminator(nn.Module):
    def __init__(self, dim_embed=DIM_EMBED, ndf = DISC_SIZE):
        super(cont_cond_cnn_discriminator, self).__init__()
        self.ndf = ndf
        self.dim_embed = dim_embed

        self.conv = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(1, self.ndf, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # input is ndf x 32 x 32
            nn.Conv2d(self.ndf, self.ndf*2, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf*2, self.ndf*4, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf*4, self.ndf*8, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.linear1 = nn.Linear(self.ndf*8*4*4, 1, bias=True)
        nn.init.xavier_uniform_(self.linear1.weight.data, 1.)
        self.linear1 = spectral_norm(self.linear1)
        self.linear2 = nn.Linear(self.dim_embed, self.ndf*8*4*4, bias=False)
        nn.init.xavier_uniform_(self.linear2.weight.data, 1.)
        self.linear2 = spectral_norm(self.linear2)


    def forward(self, x, y):

        out = self.conv(x)

        # out = torch.sum(out, dim=(2,3))
        out = out.view(-1, self.ndf*8*4*4)
        out_y = torch.sum(out*self.linear2(y), 1, keepdim=True)
        out = self.linear1(out) + out_y

        return out.view(-1, 1)
