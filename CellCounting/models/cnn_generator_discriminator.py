'''

based on the CNN structure in "Spectral normalization for generator adversarial networks"

'''
import torch
import torch.nn as nn

NC=1
IMG_SIZE=256


default_bias = True
#########################################################
# genearator
class cnn_generator(nn.Module):
    def __init__(self, ngpu=1, nz=128, ngf=64, nc=NC, bias = default_bias):
        super(cnn_generator, self).__init__()
        self.nz = nz
        self.ngpu = ngpu
        self.ngf =ngf
        self.linear = nn.Linear(nz, 4 * 4 * ngf * 8) #4*4*512
        self.main = nn.Sequential(
            # state size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=4, padding=0, bias=bias), #h=4h
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 16 x 16
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=4, padding=0, bias=bias), #h=4h
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 64 x 64
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=bias), #h=2h
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 128 x 128
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=bias), #h=2h
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 256 x 256
            nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            nn.Tanh()
            # state size. (nc) x 256 x 256
        )

    def forward(self, input):
        # input = input.view(-1, self.nz)
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.linear, input, range(self.ngpu))
            output_img = output.view(-1, 8*self.ngf, 4, 4)
            output_img = nn.parallel.data_parallel(self.main, output_img, range(self.ngpu))
        else:
            output = self.linear(input)
            output_img = output.view(-1, 8*self.ngf, 4, 4)
            output_img = self.main(output_img)
        return output_img

#########################################################
# discriminator
class cnn_discriminator(nn.Module):
    def __init__(self, use_sigmoid = True, ngpu=1, nc=NC, ndf=64, bias = default_bias):
        super(cnn_discriminator, self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, kernel_size=4, stride=4, padding=0, bias=bias), #h=h/4
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # input is ndf x 64 x 64
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (ndf*2) x 32 x 32
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (ndf*4) x 16 x 16
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (ndf*8) x 8 x 8
            nn.Conv2d(ndf*8, ndf*8, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            nn.LeakyReLU(0.2, inplace=True),
            # input is (ndf*8) x 4 x 4
        )

        linear = [nn.Linear(ndf*8*4*4, 1)]
        if use_sigmoid:
            linear += [nn.Sigmoid()]
        self.linear = nn.Sequential(*linear)

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            output = output.view(-1, self.ndf*8*4*4)
            output = nn.parallel.data_parallel(self.linear, output, range(self.ngpu))
        else:
            output = self.main(input)
            output = output.view(-1, self.ndf*8*4*4)
            output = self.linear(output)

        return output.view(-1, 1)






if __name__=="__main__":
    #test
    ngpu=1

    netG = cnn_generator(ngpu=ngpu, nz=128, ngf=64, nc=NC).cuda()
    netD = cnn_discriminator(use_sigmoid = True, ngpu=ngpu, nc=NC, ndf=64).cuda()

    #netG.apply(weights_init_DCGAN_ResNet)
    #netD.apply(weights_init_DCGAN_ResNet)

    # z = torch.randn(32, 128,1,1).cuda()
    z = torch.randn(32, 128).cuda()
    x = netG(z)
    o = netD(x)
    print(x.size())
    print(o.size())
