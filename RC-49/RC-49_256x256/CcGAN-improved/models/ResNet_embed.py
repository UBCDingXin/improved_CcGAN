'''
ResNet-based model to map an image from pixel space to a features space.
Need to be pretrained on the dataset.

if isometric_map = True, there is an extra step (elf.classifier_1 = nn.Linear(512, 32*32*3)) to increase the dimension of the feature map from 512 to 32*32*3. This selection is for desity-ratio estimation in feature space.

codes are based on
@article{
zhang2018mixup,
title={mixup: Beyond Empirical Risk Minimization},
author={Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz},
journal={International Conference on Learning Representations},
year={2018},
url={https://openreview.net/forum?id=r1Ddp1-Rb},
}
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

NC = 3
IMG_SIZE = 256
DIM_EMBED = 128


#------------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_embed(nn.Module):
    def __init__(self, block, num_blocks, nc=NC, dim_embed=DIM_EMBED):
        super(ResNet_embed, self).__init__()
        self.in_planes = 64

        self.main = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1, bias=False),  # h=h 256
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2), #h=h/2 128
            # self._make_layer(block, 64, num_blocks[0], stride=1),  # h=h
            self._make_layer(block, 64, num_blocks[0], stride=2),  # h=h/2 64
            nn.MaxPool2d(2,2), # 32
            self._make_layer(block, 128, num_blocks[1], stride=2), # h=h/2 16
            self._make_layer(block, 256, num_blocks[2], stride=2), # h=h/2 8
            self._make_layer(block, 512, num_blocks[3], stride=2), # h=h/2 4
            # nn.AvgPool2d(kernel_size=4)
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.x2h_res = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, dim_embed),
            nn.BatchNorm1d(dim_embed),
            nn.ReLU(),
        )

        self.h2y = nn.Sequential(
            nn.Linear(dim_embed, 1),
            nn.ReLU()
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        features = self.main(x)
        features = features.view(features.size(0), -1)
        features = self.x2h_res(features)
        out = self.h2y(features)

        return out, features


def ResNet18_embed(dim_embed=DIM_EMBED):
    return ResNet_embed(BasicBlock, [2,2,2,2], dim_embed=dim_embed)

def ResNet34_embed(dim_embed=DIM_EMBED):
    return ResNet_embed(BasicBlock, [3,4,6,3], dim_embed=dim_embed)

def ResNet50_embed(dim_embed=DIM_EMBED):
    return ResNet_embed(Bottleneck, [3,4,6,3], dim_embed=dim_embed)

#------------------------------------------------------------------------------
# map labels to the embedding space
class model_y2h(nn.Module):
    def __init__(self, dim_embed=DIM_EMBED):
        super(model_y2h, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(1, dim_embed),
            # nn.BatchNorm1d(dim_embed),
            nn.GroupNorm(8, dim_embed),
            nn.ReLU(),

            nn.Linear(dim_embed, dim_embed),
            # nn.BatchNorm1d(dim_embed),
            nn.GroupNorm(8, dim_embed),
            nn.ReLU(),

            nn.Linear(dim_embed, dim_embed),
            # nn.BatchNorm1d(dim_embed),
            nn.GroupNorm(8, dim_embed),
            nn.ReLU(),

            nn.Linear(dim_embed, dim_embed),
            # nn.BatchNorm1d(dim_embed),
            nn.GroupNorm(8, dim_embed),
            nn.ReLU(),

            nn.Linear(dim_embed, dim_embed),
            nn.ReLU()
        )

    def forward(self, y):
        y = y.view(-1, 1) +1e-8
        # y = torch.exp(y.view(-1, 1))
        return self.main(y)



if __name__ == "__main__":
    net = ResNet34_embed(dim_embed=128).cuda()
    x = torch.randn(16,NC,IMG_SIZE,IMG_SIZE).cuda()
    out, features = net(x)
    print(out.size())
    print(features.size())

    net_y2h = model_y2h()
