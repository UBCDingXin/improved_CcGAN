# DCGAN architecture
# Refer to Table 7 of InfoGAN-CR


# ResNet generator and discriminator
import torch
from torch import nn
from torch.nn.utils import spectral_norm


class DCGAN_Generator(nn.Module):
    def __init__(self, dim_z=128, dim_c=1):
        super(DCGAN_Generator, self).__init__()
        self.dim_z = dim_z
        self.dim_c = dim_c

        self.fc = nn.Sequential(
            nn.Linear(self.dim_z+self.dim_c, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 4*4*64),
            nn.BatchNorm1d(4*4*64),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True), #h=2h; 8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True), #h=2h; 16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, bias=True), #h=h*2; 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=True), #h=h*2; 64
            nn.Tanh()
        )
    
    def forward(self, z, c):
        z = z.view(-1, self.dim_z)
        c = c.view(-1, self.dim_c)
        x = torch.cat((z, c), 1)
        output = self.fc(x)
        output = output.view(-1, 64, 4, 4)
        output = self.deconv(output)
        return output


class DCGAN_Discriminator(nn.Module):
    def __init__(self, dim_c=1):
        super(DCGAN_Discriminator, self).__init__()
        self.dim_c = dim_c
        
        self.conv = nn.Sequential(
            
            spectral_norm(nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=True)), #h=h/2; 32
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=True)), #h=h/2; 16
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True)), #h=h/2; 8
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True)), #h=h/2; 4
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(64*4*4+dim_c, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        y = y.view(-1,1)
        output = self.conv(x)
        output = output.view(-1, 64*4*4)
        output = self.fc(torch.cat((output,y), dim=1))
        return output





if __name__ == "__main__":
    netG = DCGAN_Generator(dim_z=128, dim_c=1).cuda()
    netD = DCGAN_Discriminator(dim_c=1).cuda()
    z = torch.randn(16, 128).cuda()
    c = torch.randn(16, 1).cuda()
    out_G = netG(z,c)
    out_D = netD(out_G,c)
    print(out_G.size())
    print(out_D.size())
    