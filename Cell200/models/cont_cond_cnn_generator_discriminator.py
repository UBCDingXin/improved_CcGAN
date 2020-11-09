'''

based on the CNN structure in "Spectral normalization for generator adversarial networks"

'''
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import spectral_norm

NC=1
IMG_SIZE=64



default_bias = True
#########################################################
# genearator
class cont_cond_cnn_generator(nn.Module):
    def __init__(self, nz=128, ngf=64, nc=NC, bias = default_bias):
        super(cont_cond_cnn_generator, self).__init__()
        self.nz = nz
        self.ngf =ngf


        self.linear = nn.Linear(nz, 4 * 4 * ngf * 8)

        self.initial_conv = nn.Sequential(
            # state size: 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=bias), #h=2h
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        )

        self.main = nn.Sequential(
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

            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, z, y):
        y = y.view(-1, 1)

        z = z.view(-1, self.nz)
        output = self.linear(z) + y.repeat(1,self.ngf*8*4*4)
        output = output.view(-1, 8*self.ngf, 4, 4)

        output = self.initial_conv(output) #+ y.repeat(1,self.ngf*8*8*8).view(-1, 8*self.ngf, 8, 8)

        output = self.main(output)

        return output

#########################################################
# discriminator
class cont_cond_cnn_discriminator(nn.Module):
    def __init__(self, nc=NC, ndf=64, bias = default_bias):
        super(cont_cond_cnn_discriminator, self).__init__()
        self.ndf = ndf

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
        )


        self.linear1 = nn.Linear(ndf*8*4*4, 1, bias=True)
        nn.init.xavier_uniform_(self.linear1.weight.data, 1.)
        self.linear1 = spectral_norm(self.linear1)
        self.linear2 = nn.Linear(1, ndf*8*4*4, bias=False)
        nn.init.xavier_uniform_(self.linear2.weight.data, 1.)
        self.linear2 = spectral_norm(self.linear2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        y = y.view(-1, 1)
        output = self.main(x)

        output = output.view(-1, self.ndf*8*4*4)
        output_y = torch.sum(output*self.linear2(y+1), 1, keepdim=True)
        output = self.sigmoid(self.linear1(output) + output_y)

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
