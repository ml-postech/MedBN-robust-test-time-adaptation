"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from norm import select_norm
except:
    from .norm import select_norm


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, normlayer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = normlayer(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = normlayer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                normlayer(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, normlayer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = normlayer(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = normlayer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = normlayer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_NORM(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=10,
        norm_type="BN",
        norm_power=0.2,
    ):
        super(ResNet_NORM, self).__init__()

        self.normlayer = select_norm(norm_type, norm_power=norm_power)

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.normlayer(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride, normlayer=self.normlayer)
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, feature_maps=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out = self.layer4(out3)
        out = F.avg_pool2d(out, 4)
        out4 = out.view(out.size(0), -1)
        out = self.linear(out4)

        if feature_maps:
            return out, [out1, out2, out3, out4]
        else:
            return out


class Normalized_ResNet_NORM(ResNet_NORM):
    def __init__(
        self,
        device="cuda",
        depth=18,
        norm_type="BN",
        num_classes=10,
        norm_power=0.2,
    ):
        if depth == 18:
            super(Normalized_ResNet_NORM, self).__init__(
                BasicBlock,
                [2, 2, 2, 2],
                num_classes,
                norm_type,
                norm_power,
            )
        elif depth == 26:
            super(Normalized_ResNet_NORM, self).__init__(
                BasicBlock,
                [3, 3, 3, 3],
                num_classes,
                norm_type,
                norm_power,
            )
        else:
            pass

        self.mu = (
            torch.Tensor([0.4914, 0.4822, 0.4465]).float().view(3, 1, 1).to(device)
        )
        self.sigma = (
            torch.Tensor([0.2023, 0.1994, 0.2010]).float().view(3, 1, 1).to(device)
        )

        if num_classes == 100:
            self.mu = (
                torch.Tensor(
                    [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
                )
                .float()
                .view(3, 1, 1)
                .to(device)
            )
            self.sigma = (
                torch.Tensor(
                    [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
                )
                .float()
                .view(3, 1, 1)
                .to(device)
            )

    def forward(self, x, feature_maps=False):
        x = (x - self.mu) / self.sigma
        return super(Normalized_ResNet_NORM, self).forward(x, feature_maps)
