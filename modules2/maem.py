import torch
from torch import nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels)
        self.BN = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.BN(x)
        x = self.relu(x)
        x = self.pointwise(x)
        return x

class maem(nn.Module):
    def __init__(self,in_channel,output):
        super(maem, self).__init__()
        self.atrous_block1 = nn.Conv2d(in_channel, in_channel, 1, 1)
        self.atrous_block6 = SeparableConv2d(in_channel, in_channel, 3, 1, padding=5, dilation=5)
        self.atrous_block12 = SeparableConv2d(in_channel, in_channel, 3, 1, padding=6, dilation=6)
        self.atrous_block18 = SeparableConv2d(in_channel, in_channel, 3, 1, padding=7, dilation=7)
        self.conv_1x1_output = SeparableConv2d(in_channel * 4, output, 3, 1, 1, 1)
        self.batchnorm = nn.BatchNorm2d(output)
        self.action = nn.ReLU(inplace=True)

    def forward(self, x):
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        cat = torch.cat([atrous_block1, atrous_block6, atrous_block12, atrous_block18], dim=1)
        net = self.conv_1x1_output(cat)
        net = self.batchnorm(net)
        net = self.action(net)

        return net