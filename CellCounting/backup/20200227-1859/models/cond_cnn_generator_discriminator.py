'''

based on the CNN structure in "Spectral normalization for generator adversarial networks"

'''
import torch
import torch.nn as nn
import numpy as np

NC=1
IMG_SIZE=64

def float_to_binary(x, m, n):
    """Convert the float value `x` to a binary string of length `m + n`
    where the first `m` binary digits are the integer part and the last
    'n' binary digits are the fractional part of `x`.

    https://stackoverflow.com/questions/51843297/convert-real-numbers-to-binary-and-vice-versa-in-python
    """

    if x>0:
        is_positive = True
    else:
        x = abs(x)
        is_positive = False

    # x_scaled = round(x * 2 ** n)
    x_scaled = int(x * 2 ** n)

    if is_positive:
        return '{:0{}b}'.format(x_scaled, m + n)
    else:
        return '-' + '{:0{}b}'.format(x_scaled, m + n)


bin_m = 8
bin_n = 24
def convert_labels(labels):
    if labels.is_cuda:
        labels_np = labels.data.cpu().numpy().reshape(-1)
    else:
        labels_np = labels.data.numpy().reshape(-1)
    labels_binary = np.zeros((labels_np.shape[0],bin_m+bin_n))
    for i in range(labels_np.shape[0]):
        label_bin = float_to_binary(labels_np[i], bin_m, bin_n)
        label_bin = list(label_bin)
        if label_bin[0] != "-":
            is_positive = True
        else:
            is_positive = False
        label_bin = np.array([int(j) for j in label_bin if j != "-"])
        if is_positive:
            labels_binary[i,:] = label_bin
        else:
            labels_binary[i,:] = label_bin * -1
    labels_binary = torch.from_numpy(labels_binary)
    labels_binary = labels_binary.float()
    if labels.is_cuda:
        labels_binary = labels_binary.cuda()
    return labels_binary


label_width = bin_m+bin_n
default_bias = True
#########################################################
# genearator
class cond_cnn_generator(nn.Module):
    def __init__(self, ngpu=1, nz=128, ngf=64, nc=NC, bias = default_bias):
        super(cond_cnn_generator, self).__init__()
        self.nz = nz
        self.ngpu = ngpu
        self.ngf =ngf
        self.linear = nn.Linear(nz+label_width, 4 * 4 * ngf * 8) #4*4*512
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
            nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, z, y):
        z = z.view(-1, self.nz)
        y = y.view(-1, 1)
        y = convert_labels(y)

        z = torch.cat((z, y), dim=1)
        if z.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.linear, z, range(self.ngpu))
            output_img = output.view(-1, 8*self.ngf, 4, 4)
            output_img = nn.parallel.data_parallel(self.main, output_img, range(self.ngpu))
        else:
            output = self.linear(z)
            output_img = output.view(-1, 8*self.ngf, 4, 4)
            output_img = self.main(output_img)
        return output_img

#########################################################
# discriminator
class cond_cnn_discriminator(nn.Module):
    def __init__(self, use_sigmoid = True, ngpu=1, nc=NC, ndf=64, bias = default_bias):
        super(cond_cnn_discriminator, self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.main = nn.Sequential(
            # # input is (nc) x 64 x 64
            # nn.Conv2d(nc, ndf, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf) x 32 x 32
            # nn.Conv2d(ndf, ndf*2, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(ndf*2, ndf*2, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf) x 16 x 16
            # nn.Conv2d(ndf*2, ndf*4, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(ndf*4, ndf*4, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*2) x 8 x 8
            # nn.Conv2d(ndf*4, ndf*8, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(ndf*8, ndf*8, kernel_size=4, stride=2, padding=1, bias=bias), #h=h/2
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*4) x 4 x 4
            # nn.Conv2d(ndf*8, ndf*8, kernel_size=3, stride=1, padding=1, bias=bias), #h=h
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*8) x 4 x 4

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

        linear = [nn.Linear(ndf*8*4*4+label_width, 512),
                  nn.BatchNorm1d(512),
                  nn.ReLU(),
                  nn.Linear(512,1)]
        if use_sigmoid:
            linear += [nn.Sigmoid()]
        self.linear = nn.Sequential(*linear)

    def forward(self, x, y):
        y = y.view(-1, 1)
        y = convert_labels(y)

        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
            output = output.view(-1, self.ndf*8*4*4)
            output = torch.cat((y, output), dim=1)
            output = nn.parallel.data_parallel(self.linear, output, range(self.ngpu))
        else:
            output = self.main(x)
            output = output.view(-1, self.ndf*8*4*4)
            output = torch.cat((y, output), dim=1)
            output = self.linear(output)

        return output.view(-1, 1)






if __name__=="__main__":
    #test
    ngpu=1

    netG = cond_cnn_generator(ngpu=ngpu, nz=128, ngf=64, nc=NC).cuda()
    netD = cond_cnn_discriminator(use_sigmoid = True, ngpu=ngpu, nc=NC, ndf=64).cuda()

    z = torch.randn(32, 128).cuda()
    y = np.random.randint(100, 300, 32)
    y = torch.from_numpy(y).type(torch.float).view(-1,1).cuda()
    x = netG(z,y)
    o = netD(x,y)
    print(y.size())
    print(x.size())
    print(o.size())
