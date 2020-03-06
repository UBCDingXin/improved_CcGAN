'''

based on the CNN structure in "Spectral normalization for generator adversarial networks"

'''
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import spectral_norm

NC=1
IMG_SIZE=64

label_width = 1
label_factor = 1

default_bias = False
#########################################################
# genearator
class cont_cond_cnn_generator(nn.Module):
    def __init__(self, nz=128, ngf=64, nc=NC, bias = default_bias):
        super(cont_cond_cnn_generator, self).__init__()
        self.nz = nz
        self.ngf =ngf

        ### add label to the first conv
        self.linear = nn.Linear(nz, 4 * 4 * ngf * 8) #4*4*512
        self.conv1 = nn.Sequential(
            # state size: 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=bias), #h=2h
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. 8 x 8
        )
        self.conv2 = nn.Sequential(
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
            nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, z, y):
        z = z.view(-1, self.nz)
        ### add label to the output of the first conv
        y = y.view(-1, 1).repeat(1,self.ngf*8*8*8).view(-1, self.ngf*8, 8, 8)*label_factor
        output = self.linear(z)
        output_img = output.view(-1, 8*self.ngf, 4, 4)
        output_img = self.conv1(output_img)
        output_img = output_img + y
        output_img = self.conv2(output_img)

        return output_img

#########################################################
# discriminator
class cont_cond_cnn_discriminator(nn.Module):
    def __init__(self, use_sigmoid = True, nc=NC, ndf=64, bias = default_bias):
        super(cont_cond_cnn_discriminator, self).__init__()
        self.ndf = ndf

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            nn.BatchNorm2d(ndf),
            # spectral_norm(nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=bias)),
            nn.LeakyReLU(0.2, inplace=True),
            # input is ndf x 32 x 32
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            nn.BatchNorm2d(ndf*2),
            # spectral_norm(nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=bias)),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            nn.BatchNorm2d(ndf*4),
            # spectral_norm(nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=bias)),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            nn.LeakyReLU(0.2, inplace=True),
            # input is (ndf*8) x 4 x 4

        )

        linear = [nn.Linear(ndf*8*4*4, 1)]
        if use_sigmoid:
            linear += [nn.Sigmoid()]
        self.linear = nn.Sequential(*linear)

    def forward(self, x, y):
        y = y.view(-1, 1).repeat(1,self.ndf*8*4*4).view(-1, self.ndf*8, 4, 4)*label_factor

        output = self.main(x) + y
        output = output.view(-1, self.ndf*8*4*4)
        output = self.linear(output)

        return output.view(-1, 1)






if __name__=="__main__":
    #test

    netG = cont_cond_cnn_generator(nz=128, ngf=64, nc=NC).cuda()
    netD = cont_cond_cnn_discriminator(use_sigmoid = True, nc=NC, ndf=64).cuda()

    z = torch.randn(32, 128).cuda()
    y = np.random.randint(100, 300, 32)
    y = torch.from_numpy(y).type(torch.float).view(-1,1).cuda()
    x = netG(z,y)
    o = netD(x,y)
    print(y.size())
    print(x.size())
    print(o.size())
