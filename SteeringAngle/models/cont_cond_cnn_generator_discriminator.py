'''

based on the CNN structure in "Spectral normalization for generator adversarial networks"

'''
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

NC=3

label_width_g = 32
label_width_d = 32


default_bias = False

def self_act_conv(x, y):
    '''
    x is a tensor: NxCxWxH
    y is Nx1
    '''
    # y = torch.exp(y)
    C = x.shape[1]; W = x.shape[2]; H = x.shape[3]
    y = y.view(-1,1).repeat(1, C*W*H).view(-1, C, W, H)
    output = F.relu(x) + 0.3*y*F.relu(-x)
    # output = y**(-1) * torch.log(1+torch.exp(y*x))
    return output

def self_act_fc(x, y):
    '''
    x is a tensor: NxL
    y is Nx1
    '''
    # y = torch.exp(y)
    L = x.shape[1]
    y = y.view(-1,1).repeat(1, L)
    output = F.relu(x) + 0.3*y*F.relu(-x)
    # output = y**(-1) * torch.log(1+torch.exp(y*x))
    return output


#########################################################
# genearator
class cont_cond_cnn_generator(nn.Module):
    def __init__(self, nz=128, ngf=64, nc=NC, bias = default_bias):
        super(cont_cond_cnn_generator, self).__init__()
        self.nz = nz
        self.ngf =ngf

        # self.linear1 = nn.Linear(nz, 4 * 4 * ngf * 8) #4*4*512
        # self.linear2 = nn.Linear(label_width_g, 4 * 4 * ngf * 8)

        self.linear = nn.Linear(nz, 4 * 4 * ngf * 8)

        self.genblock1 = nn.Sequential(
            # state size: 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=bias), #h=2h
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. 8 x 8

            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        )

        self.genblock2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=bias), #h=2h
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. 16 x 16

            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
        )

        self.genblock3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=bias), #h=2h
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. 32 x 32

            nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
        )

        self.genblock4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=bias), #h=2h
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. 64 x 64

            nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            nn.Tanh()
        )


    def forward(self, z, y):
        y = y.view(-1,1)

        z = z.view(-1, self.nz)

        # output1 = self.linear1(z)
        # output2 = self.linear2(y.repeat(1, label_width_g))
        # output_img = (output1+output2).view(-1, 8*self.ngf, 4, 4)

        output_img = self.linear(z).view(-1, 8*self.ngf, 4, 4) + y.repeat(1,self.ngf*8*4*4).view(-1, self.ngf*8, 4, 4)

        output_img = self.genblock1(output_img) #+ y.repeat(1,self.ngf*8*8*8).view(-1, self.ngf*8, 8, 8)
        # output_img = self_act_conv(output_img, y)

        output_img = self.genblock2(output_img) #+ y.repeat(1,self.ngf*4*16*16).view(-1, self.ngf*4, 16, 16)
        # output_img = self_act_conv(output_img, y)

        output_img = self.genblock3(output_img) #+ y.repeat(1,self.ngf*2*32*32).view(-1, self.ngf*2, 32, 32)
        # output_img = self_act_conv(output_img, y)

        output_img = self.genblock4(output_img)

        return output_img

#########################################################
# discriminator

class cont_cond_cnn_discriminator(nn.Module):
    def __init__(self, use_sigmoid = True, nc=NC, ndf=64, bias = default_bias):
        super(cont_cond_cnn_discriminator, self).__init__()
        self.ndf = ndf

        self.inputblock = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # input is ndf x 32 x 32

            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.discblock1 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (ndf*2) x 16 x 16

            nn.Conv2d(ndf*2, ndf*2, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.discblock2 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (ndf*4) x 8 x 8

            nn.Conv2d(ndf*4, ndf*4, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.discblock3 = nn.Sequential(
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8, ndf*8, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            nn.LeakyReLU(0.2, inplace=True),
        )

        # self.linear1 = nn.Linear(ndf*8*4*4, 1, bias=bias)
        # self.linear2 = nn.Linear(label_width_d, 1, bias=bias)
        # self.sigmoid = nn.Sigmoid()

        # self.linear1 = nn.Linear(ndf*8*4*4, 1, bias=bias)
        # self.linear2 = nn.Linear(label_width_d, ndf*8*4*4, bias=bias)
        # self.sigmoid = nn.Sigmoid()

        self.linear = nn.Sequential(
            nn.Linear(ndf*8*4*4, 1, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        y = y.view(-1,1)

        output = self.inputblock(x)
        output = self.discblock1(output) #+ y.repeat(1,self.ndf*2*16*16).view(-1, self.ndf*2, 16, 16)
        output = self.discblock2(output) #+ y.repeat(1,self.ndf*4*8*8).view(-1, self.ndf*4, 8, 8)
        output = self.discblock3(output) #+ y.repeat(1,self.ndf*8*4*4).view(-1, self.ndf*8, 4, 4)

        # output = output.view(-1, self.ndf*8*4*4)
        # output1 = self.linear1(output)
        # output2 = self.linear2(y.repeat(1, label_width_d))
        # output = self.sigmoid(output1+output2)

        # output = output.view(-1, self.ndf*8*4*4)
        # output_y = torch.sum(output*self.linear2(y.repeat(1, label_width_d)), 1)
        # output = self.sigmoid(self.linear1(output) + output_y)

        output = output.view(-1, self.ndf*8*4*4)
        output = output + y.repeat(1,self.ndf*8*4*4)
        output = self.linear(output)

        return output.view(-1, 1)
