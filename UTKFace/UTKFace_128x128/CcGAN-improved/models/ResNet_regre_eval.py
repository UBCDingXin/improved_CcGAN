'''
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
IMG_SIZE = 128


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


class ResNet_regre_eval(nn.Module):
    def __init__(self, block, num_blocks, nc=NC, ngpu = 1, feature_layer='f3'):
        super(ResNet_regre_eval, self).__init__()
        self.in_planes = 64
        self.ngpu = ngpu
        self.feature_layer=feature_layer

        self.block1 = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1, bias=False),  # h=h
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2), #h=h/2 64
            self._make_layer(block, 64, num_blocks[0], stride=2),  # h=h/2 32
        )
        self.block2 = self._make_layer(block, 128, num_blocks[1], stride=2) # h=h/2 16
        self.block3 = self._make_layer(block, 256, num_blocks[2], stride=2) # h=h/2 8
        self.block4 = self._make_layer(block, 512, num_blocks[3], stride=2) # h=h/2 4

        self.pool1 = nn.AvgPool2d(kernel_size=4)
        if self.feature_layer == 'f2':
            self.pool2 = nn.AdaptiveAvgPool2d((2,2))
        elif self.feature_layer == 'f3':
            self.pool2 = nn.AdaptiveAvgPool2d((2,2))
        else:
            self.pool2 = nn.AdaptiveAvgPool2d((1,1))

        linear_layers = [
                nn.Linear(512*block.expansion, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 1),
                # nn.Sigmoid()
                nn.ReLU(),
            ]
        self.linear = nn.Sequential(*linear_layers)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        if x.is_cuda and self.ngpu > 1:
            ft1 = nn.parallel.data_parallel(self.block1, x, range(self.ngpu))
            ft2 = nn.parallel.data_parallel(self.block2, ft1, range(self.ngpu))
            ft3 = nn.parallel.data_parallel(self.block3, ft2, range(self.ngpu))
            ft4 = nn.parallel.data_parallel(self.block4, ft3, range(self.ngpu))
            out = nn.parallel.data_parallel(self.pool1, ft4, range(self.ngpu))
            out = out.view(out.size(0), -1)
            out = nn.parallel.data_parallel(self.linear, out, range(self.ngpu))
        else:
            ft1 = self.block1(x)
            ft2 = self.block2(ft1)
            ft3 = self.block3(ft2)
            ft4 = self.block4(ft3)
            out = self.pool1(ft4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)

        if self.feature_layer == 'f2':
            ext_features = self.pool2(ft2)
        elif self.feature_layer == 'f3':
            ext_features = self.pool2(ft3)
        else:
            ext_features = self.pool2(ft4)

        ext_features = ext_features.view(ext_features.size(0), -1)

        return out, ext_features


def ResNet18_regre_eval(ngpu = 1):
    return ResNet_regre_eval(BasicBlock, [2,2,2,2], ngpu = ngpu)

def ResNet34_regre_eval(ngpu = 1):
    return ResNet_regre_eval(BasicBlock, [3,4,6,3], ngpu = ngpu)

def ResNet50_regre_eval(ngpu = 1):
    return ResNet_regre_eval(Bottleneck, [3,4,6,3], ngpu = ngpu)

def ResNet101_regre_eval(ngpu = 1):
    return ResNet_regre_eval(Bottleneck, [3,4,23,3], ngpu = ngpu)

def ResNet152_regre_eval(ngpu = 1):
    return ResNet_regre_eval(Bottleneck, [3,8,36,3], ngpu = ngpu)


if __name__ == "__main__":
    net = ResNet34_regre_eval(ngpu = 1).cuda()
    x = torch.randn(4,NC,IMG_SIZE,IMG_SIZE).cuda()
    out, features = net(x)
    print(out.size())
    print(features.size())
