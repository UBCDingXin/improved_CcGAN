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

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        self.embed = spectral_norm(self.embed)

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, num_classes, bias=True):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=bias)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        self.condbn1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.condbn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2)

        self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0, bias=bias) #h=h
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)
        self.bypass = nn.Sequential(
            nn.Upsample(scale_factor=2),
            self.bypass_conv,
        )

    def forward(self, x, y):
        out = self.condbn1(x, y)
        out = self.relu(out)
        out = self.upsample(out)
        out = self.conv1(out)
        out = self.condbn2(out, y)
        out = self.relu(out)
        out = self.conv2(out)

        return out + self.bypass(x)


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



class cond_cnn_generator(nn.Module):
    def __init__(self, nz=128, num_classes=100):
        super(cond_cnn_generator, self).__init__()
        self.z_dim = nz

        self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE*16, bias=True)
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        self.genblock0 = ResBlockGenerator(GEN_SIZE*16, GEN_SIZE*8, num_classes=num_classes) #4--->8
        self.genblock1 = ResBlockGenerator(GEN_SIZE*8, GEN_SIZE*4, num_classes=num_classes) #8--->16
        self.genblock2 = ResBlockGenerator(GEN_SIZE*4, GEN_SIZE*2, num_classes=num_classes) #16--->32
        self.genblock3 = ResBlockGenerator(GEN_SIZE*2, GEN_SIZE, num_classes=num_classes) #32--->64

        self.final = nn.Sequential(
            nn.BatchNorm2d(GEN_SIZE),
            nn.ReLU(),
            self.final,
            nn.Tanh()
        )

    def forward(self, z, y):
        z = z.view(z.size(0), z.size(1))
        out = self.dense(z)
        out = out.view(-1, GEN_SIZE*16, 4, 4)

        out = self.genblock0(out, y)
        out = self.genblock1(out, y)
        out = self.genblock2(out, y)
        out = self.genblock3(out, y)
        out = self.final(out)

        return out


class cond_cnn_discriminator(nn.Module):
    def __init__(self, num_classes=100):
        super(cond_cnn_discriminator, self).__init__()

        self.discblock = nn.Sequential(
            FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2, bias=True), #64--->32
            ResBlockDiscriminator(DISC_SIZE , DISC_SIZE*2, stride=2, bias=True), #32--->16
            ResBlockDiscriminator(DISC_SIZE*2  , DISC_SIZE*4, stride=2, bias=True), #16--->8
            ResBlockDiscriminator(DISC_SIZE*4, DISC_SIZE*8, stride=2, bias=True), #8--->4
            ResBlockDiscriminator(DISC_SIZE*8, DISC_SIZE*16, stride=1, bias=True), #4--->4
            nn.ReLU()
        )

        self.linear = nn.Linear(DISC_SIZE*16, 1)
        nn.init.xavier_uniform_(self.linear.weight.data, 1.)
        self.linear = spectral_norm(self.linear)
        self.label_emb = nn.Embedding(num_classes, DISC_SIZE*16)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):

        output = self.discblock(x)

        output = torch.sum(output, dim=(2, 3))
        output_y = torch.sum(output*self.label_emb(y), 1, keepdim=True)
        # output = self.linear(output) + output_y
        output = self.sigmoid(self.linear(output) + output_y)

        return output.view(-1, 1)
