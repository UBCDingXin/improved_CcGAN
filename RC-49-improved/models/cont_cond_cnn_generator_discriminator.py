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
bias = True
GEN_SIZE=64
DISC_SIZE=64

DIM_EMBED=128



class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, dim_embed):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)

        # self.embed = nn.Linear(dim_embed, num_features * 2, bias=False)
        # self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        # self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        # # self.embed = spectral_norm(self.embed) #seems not work

        self.embed_gamma = nn.Linear(dim_embed, num_features, bias=False)
        self.embed_beta = nn.Linear(dim_embed, num_features, bias=False)

    def forward(self, x, y):
        out = self.bn(x)

        # gamma, beta = self.embed(y).chunk(2, 1)
        # out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)

        gamma = self.embed_gamma(y).view(-1, self.num_features, 1, 1)
        beta = self.embed_beta(y).view(-1, self.num_features, 1, 1)
        out = out + out*gamma + beta

        return out


class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, dim_embed, bias=True):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=bias)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        self.condbn1 = ConditionalBatchNorm2d(in_channels, dim_embed)
        self.condbn2 = ConditionalBatchNorm2d(out_channels, dim_embed)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2)

        # unconditional case
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )


        self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0, bias=bias) #h=h
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)
        self.bypass = nn.Sequential(
            nn.Upsample(scale_factor=2),
            self.bypass_conv,
        )

    def forward(self, x, y):
        if y is not None:
            out = self.condbn1(x, y)
            out = self.relu(out)
            out = self.upsample(out)
            out = self.conv1(out)
            out = self.condbn2(out, y)
            out = self.relu(out)
            out = self.conv2(out)
            out = out + self.bypass(x)
        else:
            out = self.model(x) + self.bypass(x)

        return out

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



class cont_cond_cnn_generator(nn.Module):
    def __init__(self, nz=128, img_size=64, dim_embed=DIM_EMBED):
        super(cont_cond_cnn_generator, self).__init__()
        self.z_dim = nz
        self.dim_embed = dim_embed

        self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE*16, bias=True)
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1, bias=bias)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        self.genblock0 = ResBlockGenerator(GEN_SIZE*16, GEN_SIZE*8, dim_embed=dim_embed) #4--->8
        self.genblock1 = ResBlockGenerator(GEN_SIZE*8, GEN_SIZE*4, dim_embed=dim_embed) #8--->16
        self.genblock2 = ResBlockGenerator(GEN_SIZE*4, GEN_SIZE*2, dim_embed=dim_embed) #16--->32
        self.genblock3 = ResBlockGenerator(GEN_SIZE*2, GEN_SIZE, dim_embed=dim_embed) #32--->64

        self.final = nn.Sequential(
            nn.BatchNorm2d(GEN_SIZE),
            nn.ReLU(),
            self.final,
            nn.Tanh()
        )

    def forward(self, z, y): #y is embedded in the feature space
        z = z.view(z.size(0), z.size(1))
        out = self.dense(z)
        out = out.view(-1, GEN_SIZE*16, 4, 4)

        out = self.genblock0(out, y)
        out = self.genblock1(out, y)
        out = self.genblock2(out, y)
        out = self.genblock3(out, y)
        out = self.final(out)

        return out


class cont_cond_cnn_discriminator(nn.Module):
    def __init__(self, img_size=64, dim_embed=DIM_EMBED):
        super(cont_cond_cnn_discriminator, self).__init__()
        self.dim_embed = dim_embed

        self.discblock1 = nn.Sequential(
            FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2), #64--->32
            ResBlockDiscriminator(DISC_SIZE, DISC_SIZE*2, stride=2), #32--->16
            ResBlockDiscriminator(DISC_SIZE*2, DISC_SIZE*4, stride=2), #16--->8
        )
        self.discblock2 = ResBlockDiscriminator(DISC_SIZE*4, DISC_SIZE*8, stride=2) #8--->4
        self.discblock3 = nn.Sequential(
            ResBlockDiscriminator(DISC_SIZE*8, DISC_SIZE*16, stride=1), #4--->4;
            nn.ReLU(),
        )


        # self.linear1 = nn.Linear(DISC_SIZE*16, 1, bias=True)
        # nn.init.xavier_uniform_(self.linear1.weight.data, 1.)
        # self.linear1 = spectral_norm(self.linear1)
        # self.linear2 = nn.Linear(self.dim_embed, DISC_SIZE*16, bias=False)
        # nn.init.xavier_uniform_(self.linear2.weight.data, 1.)
        # self.linear2 = spectral_norm(self.linear2)

        self.linear1 = nn.Linear(DISC_SIZE*16*4*4, 1, bias=True)
        nn.init.xavier_uniform_(self.linear1.weight.data, 1.)
        self.linear1 = spectral_norm(self.linear1)
        self.linear2 = nn.Linear(self.dim_embed, DISC_SIZE*16*4*4, bias=False)
        nn.init.xavier_uniform_(self.linear2.weight.data, 1.)
        self.linear2 = spectral_norm(self.linear2)

    def forward(self, x, y):
        output = self.discblock1(x)
        output = self.discblock2(output)
        output = self.discblock3(output)

        # output = torch.sum(output, dim=(2,3))
        # output_y = torch.sum(output*self.linear2(y), 1, keepdim=True)
        # output = self.linear1(output) + output_y

        output = output.view(-1, DISC_SIZE*16*4*4)
        output_y = torch.sum(output*self.linear2(y), 1, keepdim=True)
        output = self.linear1(output) + output_y

        return output.view(-1, 1)
