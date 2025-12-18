import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   stride, padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.pointwise(x)
        x = self.bn2(x)
        return x


class XceptionResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid_channels = out_channels // 4

        self.conv1 = DepthwiseSeparableConv(
            in_channels, mid_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = DepthwiseSeparableConv(
            mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = DepthwiseSeparableConv(
            mid_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = F.relu(self.conv1(x), inplace=True)
        out = F.relu(self.conv2(out), inplace=True)
        out = self.conv3(out)

        out = F.relu(out + residual, inplace=True)
        return out


class XceptionResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.in_channels = 64

        self.layer1 = self._make_layer(out_channels=256, blocks=3, stride=1)
        self.layer2 = self._make_layer(out_channels=512, blocks=4, stride=2)
        self.layer3 = self._make_layer(out_channels=1024, blocks=6, stride=2)
        self.layer4 = self._make_layer(out_channels=2048, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []

        layers.append(XceptionResNetBlock(
            in_channels=self.in_channels,
            out_channels=out_channels,
            stride=stride
        ))
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(XceptionResNetBlock(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=1
            ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
