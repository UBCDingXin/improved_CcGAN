'''

based on the CNN structure in "Spectral normalization for generator adversarial networks"

'''
import torch
import torch.nn as nn
import numpy as np

default_bias = False

NC = 1

#########################################################
# genearator
class cond_cnn_generator(nn.Module):
    def __init__(self, nz=128, ngf=128, nc=NC, dim_c=1, bias = default_bias):
        super(cond_cnn_generator, self).__init__()
        self.nz = nz
        self.dim_c = dim_c
        self.ngf =ngf

        self.linear = nn.Linear(nz+dim_c, 4 * 4 * ngf * 8) #4*4*512

        self.main = nn.Sequential(
            # state size: 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=bias), #h=2h
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=bias), #h=2h
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=bias), #h=2h
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=bias), #h=2h
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. 64 x 64
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input, labels):
        input = input.view(-1, self.nz)
        input = torch.cat((input,labels.view(-1,1)), 1)
        output = self.linear(input)
        output = output.view(-1, 8*self.ngf, 4, 4)
        output = self.main(output)
        return output


class cond_cnn_discriminator(nn.Module):
    def __init__(self, nc=NC, ndf=128, dim_c=1, bias = default_bias):
        super(cond_cnn_discriminator, self).__init__()
        self.ndf = ndf
        self.dim_c = dim_c
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # input is ndf x 32 x 32
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            nn.LeakyReLU(0.2, inplace=True),
            # input is (ndf*8) x 4 x 4
        )

        self.linear = nn.Sequential(
            nn.Linear(ndf*8*4*4+dim_c, 1),
            nn.Sigmoid()
        )

    def forward(self, input, labels):
        output = self.main(input)
        output = output.view(-1, self.ndf*8*4*4)
        output = torch.cat((output,labels.view(-1,1)), 1)
        output = self.linear(output)

        return output.view(-1, 1)


if __name__ == "__main__":
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    netG = cond_cnn_generator(nz=128, dim_c=1).cuda()
    netD = cond_cnn_discriminator(dim_c=1).cuda()
    z = torch.randn(16, 128).cuda()
    c = torch.randn(16, 1).cuda()
    out_G = netG(z,c)
    out_D = netD(out_G,c)
    print(out_G.size())
    print(out_D.size())

    print('G:', get_parameter_number(netG))
    print('D:', get_parameter_number(netD))