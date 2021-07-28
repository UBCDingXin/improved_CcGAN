import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)


def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))


def snlinear(in_features, out_features):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))


def sn_embedding(num_embeddings, embedding_dim):
    return spectral_norm(nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim))


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma*attn_g
        return out


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GenBlock, self).__init__()
        self.cond_bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.cond_bn2 = nn.BatchNorm2d(out_channels)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x0 = x

        x = self.cond_bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # upsample
        x = self.snconv2d1(x)
        x = self.cond_bn2(x)
        x = self.relu(x)
        x = self.snconv2d2(x)

        x0 = F.interpolate(x0, scale_factor=2, mode='nearest') # upsample
        x0 = self.snconv2d0(x0)

        out = x + x0
        return out


class cGAN_concat_SAGAN_Generator(nn.Module):
    """Generator."""

    def __init__(self, z_dim, dim_c=1, g_conv_dim=64):
        super(cGAN_concat_SAGAN_Generator, self).__init__()

        self.z_dim = z_dim
        self.dim_c = dim_c
        self.g_conv_dim = g_conv_dim
        self.snlinear0 = snlinear(in_features=z_dim+dim_c, out_features=g_conv_dim*16*4*4)
        self.block1 = GenBlock(g_conv_dim*16, g_conv_dim*16)
        self.block2 = GenBlock(g_conv_dim*16, g_conv_dim*8)
        self.block3 = GenBlock(g_conv_dim*8, g_conv_dim*4)
        self.block4 = GenBlock(g_conv_dim*4, g_conv_dim*2)
        self.self_attn = Self_Attn(g_conv_dim*2)
        self.block5 = GenBlock(g_conv_dim*2, g_conv_dim*2)
        self.block6 = GenBlock(g_conv_dim*2, g_conv_dim)
        self.bn = nn.BatchNorm2d(g_conv_dim, eps=1e-5, momentum=0.0001, affine=True)
        self.relu = nn.ReLU()
        self.snconv2d1 = snconv2d(in_channels=g_conv_dim, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        # Weight init
        self.apply(init_weights)

    def forward(self, z, labels):
        # n x z_dim
        act0 = self.snlinear0(torch.cat((z, labels.view(-1,1)),dim=1))            # n x g_conv_dim*16*4*4
        act0 = act0.view(-1, self.g_conv_dim*16, 4, 4) # n x g_conv_dim*16 x 4 x 4
        act1 = self.block1(act0)    # n x g_conv_dim*16 x 8 x 8
        act2 = self.block2(act1)    # n x g_conv_dim*8 x 16 x 16
        act3 = self.block3(act2)    # n x g_conv_dim*4 x 32 x 32
        act4 = self.block4(act3)    # n x g_conv_dim*2 x 64 x 64
        act4 = self.self_attn(act4)         # n x g_conv_dim*2 x 64 x 64
        act5 = self.block5(act4)    # n x g_conv_dim  x 128 x 128
        act6 = self.block6(act5)    # n x g_conv_dim  x 256 x 256
        act6 = self.bn(act6)                # n x g_conv_dim  x 256 x 256
        act6 = self.relu(act6)              # n x g_conv_dim  x 256 x 256
        act7 = self.snconv2d1(act6)         # n x 3 x 256 x 256
        act7 = self.tanh(act7)              # n x 3 x 256 x 256
        return act7


class DiscOptBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscOptBlock, self).__init__()
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x0 = x

        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        x = self.downsample(x)

        x0 = self.downsample(x0)
        x0 = self.snconv2d0(x0)

        out = x + x0
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, downsample=True):
        x0 = x

        x = self.relu(x)
        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        if downsample:
            x = self.downsample(x)

        if downsample or self.ch_mismatch:
            x0 = self.snconv2d0(x0)
            if downsample:
                x0 = self.downsample(x0)

        out = x + x0
        return out


class cGAN_concat_SAGAN_Discriminator(nn.Module):
    """Discriminator."""

    def __init__(self, dim_c=1, d_conv_dim=64):
        super(cGAN_concat_SAGAN_Discriminator, self).__init__()
        self.d_conv_dim = d_conv_dim
        self.dim_c = dim_c
        self.opt_block1 = DiscOptBlock(3, d_conv_dim)
        self.block1 = DiscBlock(d_conv_dim, d_conv_dim*2)
        self.self_attn = Self_Attn(d_conv_dim*2)
        self.block2 = DiscBlock(d_conv_dim*2, d_conv_dim*4)
        self.block3 = DiscBlock(d_conv_dim*4, d_conv_dim*6)
        self.block4 = DiscBlock(d_conv_dim*6, d_conv_dim*12)
        self.block5 = DiscBlock(d_conv_dim*12, d_conv_dim*12)
        self.block6 = DiscBlock(d_conv_dim*12, d_conv_dim*16)
        self.relu = nn.ReLU()
        self.snlinear1 = snlinear(in_features=d_conv_dim*16*4*4+dim_c, out_features=1)

    def forward(self, x, labels):
        labels = labels.view(-1,1)
        # n x 3 x 256 x 256
        h0 = self.opt_block1(x) # n x d_conv_dim   x 128 x 128
        h1 = self.block1(h0)    # n x d_conv_dim*2 x 64 x 64
        h1 = self.self_attn(h1) # n x d_conv_dim*2 x 64 x 64
        h2 = self.block2(h1)    # n x d_conv_dim*4 x 32 x 32
        h3 = self.block3(h2)    # n x d_conv_dim*8 x 16 x 16
        h4 = self.block4(h3)    # n x d_conv_dim*16 x 8 x  8
        h5 = self.block5(h4)    # n x d_conv_dim*16 x 4 x 4
        h6 = self.block6(h5, downsample=False)  # n x d_conv_dim*16 x 4 x 4
        out = self.relu(h6)              # n x d_conv_dim*16 x 4 x 4
        out = out.view(-1,self.d_conv_dim*16*4*4)
        out = torch.cat((out, labels),dim=1)
        out = self.snlinear1(out)
        return out



if __name__ == "__main__":
    
    netG = cGAN_concat_SAGAN_Generator(z_dim=128).cuda()
    netD = cGAN_concat_SAGAN_Discriminator().cuda()

    n = 4
    y = torch.randn(n, 1).cuda()
    z = torch.randn(n, 128).cuda()
    x = netG(z,y)
    print(x.size())
    o = netD(x,y)
    print(o.size())