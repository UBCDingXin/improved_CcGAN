import torch
from torch import nn



class encoder(nn.Module):
    def __init__(self, dim_bottleneck=512, ch=64):
        super(encoder, self).__init__()
        self.ch = ch
        self.dim_bottleneck = dim_bottleneck

        self.conv = nn.Sequential(
            nn.Conv2d(3, ch, kernel_size=4, stride=2, padding=1), #h=h/2; 64
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.MaxPool2d(2,2), #h=h/2; 32
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1), #h=h
            nn.BatchNorm2d(ch),
            nn.ReLU(),

            nn.Conv2d(ch, ch*2, kernel_size=4, stride=2, padding=1), #h=h/2; 16
            nn.BatchNorm2d(ch*2),
            nn.ReLU(),
            nn.Conv2d(ch*2, ch*2, kernel_size=3, stride=1, padding=1), #h=h
            nn.BatchNorm2d(ch*2),
            nn.ReLU(),

            nn.Conv2d(ch*2, ch*4, kernel_size=4, stride=2, padding=1), #h=h/2; 8
            nn.BatchNorm2d(ch*4),
            nn.ReLU(),
            nn.Conv2d(ch*4, ch*4, kernel_size=3, stride=1, padding=1), #h=h
            nn.BatchNorm2d(ch*4),
            nn.ReLU(),

            nn.Conv2d(ch*4, ch*8, kernel_size=4, stride=2, padding=1), #h=h/2; 4
            nn.BatchNorm2d(ch*8),
            nn.ReLU(),
            nn.Conv2d(ch*8, ch*8, kernel_size=3, stride=1, padding=1), #h=h
            nn.BatchNorm2d(ch*8),
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
            nn.Linear(ch*8*4*4, dim_bottleneck),
            # nn.ReLU()
        )

    def forward(self, x):
        feature = self.conv(x)
        feature = feature.view(-1, self.ch*8*4*4)
        feature = self.linear(feature)
        return feature



class decoder(nn.Module):
    def __init__(self, dim_bottleneck=512, ch=64):
        super(decoder, self).__init__()
        self.ch = ch
        self.dim_bottleneck = dim_bottleneck

        self.linear = nn.Sequential(
            nn.Linear(dim_bottleneck, ch*16*4*4)
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(ch*16, ch*8, kernel_size=4, stride=2, padding=1), #h=2h; 8
            nn.BatchNorm2d(ch*8),
            nn.ReLU(True),
            nn.Conv2d(ch*8, ch*8, kernel_size=3, stride=1, padding=1), #h=h
            nn.BatchNorm2d(ch*8),
            nn.ReLU(),

            nn.ConvTranspose2d(ch*8, ch*4, kernel_size=4, stride=2, padding=1), #h=2h; 16
            nn.BatchNorm2d(ch*4),
            nn.ReLU(True),
            nn.Conv2d(ch*4, ch*4, kernel_size=3, stride=1, padding=1), #h=h
            nn.BatchNorm2d(ch*4),
            nn.ReLU(),

            nn.ConvTranspose2d(ch*4, ch*2, kernel_size=4, stride=2, padding=1), #h=2h; 32
            nn.BatchNorm2d(ch*2),
            nn.ReLU(True),
            nn.Conv2d(ch*2, ch*2, kernel_size=3, stride=1, padding=1), #h=h
            nn.BatchNorm2d(ch*2),
            nn.ReLU(),

            nn.ConvTranspose2d(ch*2, ch, kernel_size=4, stride=2, padding=1), #h=2h; 64
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1), #h=h
            nn.BatchNorm2d(ch),
            nn.ReLU(),

            nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1), #h=2h; 128
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1), #h=h
            nn.Tanh()
        )

    def forward(self, feature):
        out = self.linear(feature)
        out = out.view(-1, self.ch*16, 4, 4)
        out = self.deconv(out)
        return out


if __name__=="__main__":
    #test

    net_encoder = encoder(dim_bottleneck=512, ch=64).cuda()
    net_decoder = decoder(dim_bottleneck=512, ch=64).cuda()

    x = torch.randn(10, 3, 128, 128).cuda()
    f = net_encoder(x)
    xh = net_decoder(f)
    print(f.size())
    print(xh.size())
