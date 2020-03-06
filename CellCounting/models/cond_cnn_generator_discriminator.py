'''

based on the CNN structure in "Spectral normalization for generator adversarial networks"

'''
import torch
import torch.nn as nn
import numpy as np

default_bias = True
<<<<<<< HEAD

NC = 1

=======

<<<<<<< HEAD
NC = 1


#########################################################
# genearator
class cond_cnn_generator(nn.Module):
    def __init__(self, nz=128, ngf=128, nc=NC, num_classes=10, bias = default_bias):
        super(cond_cnn_generator, self).__init__()
        self.nz = nz
        self.num_classes = num_classes
        self.ngf =ngf

        self.linear = nn.Linear(nz+num_classes, 4 * 4 * ngf * 8) #4*4*512

        self.main = nn.Sequential(
=======
label_width = 1
label_factor = 1
>>>>>>> 740531af0a182b1d40d461b5b996b744d21e1182

#########################################################
# genearator
class cond_cnn_generator(nn.Module):
    def __init__(self, nz=128, ngf=128, nc=NC, num_classes=10, bias = default_bias):
        super(cond_cnn_generator, self).__init__()
        self.nz = nz
        self.num_classes = num_classes
        self.ngf =ngf

<<<<<<< HEAD
        self.linear = nn.Linear(nz+num_classes, 4 * 4 * ngf * 8) #4*4*512

        self.main = nn.Sequential(
=======
        ### add label to the first conv
        self.linear = nn.Linear(nz, 4 * 4 * ngf * 8) #4*4*512
        self.conv1 = nn.Sequential(
>>>>>>> 7ac9937253c320ac79df75051df2f3f4848e2534
>>>>>>> 740531af0a182b1d40d461b5b996b744d21e1182
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
            nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.label_emb = nn.Embedding(num_classes, num_classes)

<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 740531af0a182b1d40d461b5b996b744d21e1182
    def forward(self, input, labels):
        input = input.view(-1, self.nz)
        input = torch.cat((self.label_emb(labels), input), 1)
        output = self.linear(input)
        output = output.view(-1, 8*self.ngf, 4, 4)
        output = self.main(output)
        return output
<<<<<<< HEAD
=======


class cond_cnn_discriminator(nn.Module):
    def __init__(self, use_sigmoid = True, nc=NC, ndf=128, num_classes=10, bias = default_bias):
        super(cond_cnn_discriminator, self).__init__()
        self.ndf = ndf
        self.num_classes = num_classes
=======
    def forward(self, z, y):
        z = z.view(-1, self.nz)
        ### add label to the output of the first conv
        y = y.view(-1, 1).repeat(1,self.ngf*8*8*8).view(-1, self.ngf*8, 8, 8)*label_factor
        output = self.linear(z)
        output_img = output.view(-1, 8*self.ngf, 4, 4)
        output_img = self.conv1(output_img)
        output_img = output_img + y
        output_img = self.conv2(output_img)
>>>>>>> 740531af0a182b1d40d461b5b996b744d21e1182


class cond_cnn_discriminator(nn.Module):
    def __init__(self, use_sigmoid = True, nc=NC, ndf=128, num_classes=10, bias = default_bias):
        super(cond_cnn_discriminator, self).__init__()
        self.ndf = ndf
<<<<<<< HEAD
        self.num_classes = num_classes
=======

>>>>>>> 7ac9937253c320ac79df75051df2f3f4848e2534
>>>>>>> 740531af0a182b1d40d461b5b996b744d21e1182
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
<<<<<<< HEAD
        linear = [nn.Linear(ndf*8*4*4+num_classes, 1)]
=======
<<<<<<< HEAD
        linear = [nn.Linear(ndf*8*4*4+num_classes, 1)]
=======

        linear = [nn.Linear(ndf*8*4*4, 1)]
>>>>>>> 7ac9937253c320ac79df75051df2f3f4848e2534
>>>>>>> 740531af0a182b1d40d461b5b996b744d21e1182
        if use_sigmoid:
            linear += [nn.Sigmoid()]
        self.linear = nn.Sequential(*linear)

<<<<<<< HEAD
        self.label_emb = nn.Embedding(num_classes, num_classes)
=======
<<<<<<< HEAD
        self.label_emb = nn.Embedding(num_classes, num_classes)

    def forward(self, input, labels):
        output = self.main(input)
        output = output.view(-1, self.ndf*8*4*4)
        output = torch.cat((self.label_emb(labels), output), 1)
        output = self.linear(output)

        return output.view(-1, 1)
=======
    def forward(self, x, y):
        y = y.view(-1, 1).repeat(1,self.ndf*8*4*4).view(-1, self.ndf*8, 4, 4)*label_factor
>>>>>>> 740531af0a182b1d40d461b5b996b744d21e1182

    def forward(self, input, labels):
        output = self.main(input)
        output = output.view(-1, self.ndf*8*4*4)
        output = torch.cat((self.label_emb(labels), output), 1)
        output = self.linear(output)

        return output.view(-1, 1)
<<<<<<< HEAD
=======






if __name__=="__main__":
    #test

    netG = cond_cnn_generator(nz=128, ngf=64, nc=NC).cuda()
    netD = cond_cnn_discriminator(use_sigmoid = True, nc=NC, ndf=64).cuda()

    z = torch.randn(32, 128).cuda()
    y = np.random.randint(100, 300, 32)
    y = torch.from_numpy(y).type(torch.float).view(-1,1).cuda()
    x = netG(z,y)
    o = netD(x,y)
    print(y.size())
    print(x.size())
    print(o.size())
>>>>>>> 7ac9937253c320ac79df75051df2f3f4848e2534
>>>>>>> 740531af0a182b1d40d461b5b996b744d21e1182
