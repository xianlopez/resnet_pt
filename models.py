import torch
from torch import nn


# 'Same' padding, conv2d, batch normalization, and ReLU.
class ResnetConvSet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResnetBlockSimple(nn.Module):
    def __init__(self, in_channels, downsample):
        super().__init__()
        out_channels = 2 * in_channels if downsample else in_channels
        stride = 2 if downsample else 1

        self.conv1 = ResnetConvSet(in_channels, out_channels, 3, stride)
        self.conv2 = ResnetConvSet(out_channels, out_channels, 3, 1)

        if downsample:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False)
        else:
            self.shortcut = nn.Identity()

        self.final_relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        x = self.final_relu(x)
        return x


class ResnetLayerSimple(nn.Module):
    def __init__(self, in_channels, num_blocks, downsample):
        super().__init__()
        out_channels = 2 * in_channels if downsample else in_channels
        self.blocks = nn.ModuleList([ResnetBlockSimple(in_channels, downsample)])
        for i in range(1, num_blocks):
            self.blocks.append(ResnetBlockSimple(out_channels, False))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, nclasses):
        super().__init__()
        self.conv1 = ResnetConvSet(3, 64, 7, 2)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.layer2 = ResnetLayerSimple(64, 2, False)
        self.layer3 = ResnetLayerSimple(64, 2, True)
        self.layer4 = ResnetLayerSimple(128, 2, True)
        self.layer5 = ResnetLayerSimple(256, 2, True)
        self.avg_pool = nn.AvgPool2d(7)
        self.dense = nn.Linear(512, nclasses)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avg_pool(x)
        x = torch.squeeze(x)  # TODO: Be careful if batch size is 1!
        x = self.dense(x)
        return x
