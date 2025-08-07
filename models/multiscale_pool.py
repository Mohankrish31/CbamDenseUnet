import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiScalePool(nn.Module):
    def __init__(self, in_channels):
        super(MultiScalePool, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(4)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.out_conv = nn.Conv2d(in_channels + 3 * (in_channels // 4), in_channels, kernel_size=1)
    def forward(self, x):
        size = x.size()[2:]
        p1 = F.interpolate(self.conv1(self.pool1(x)), size=size, mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.conv2(self.pool2(x)), size=size, mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.conv3(self.pool3(x)), size=size, mode='bilinear', align_corners=False)
        out = torch.cat([x, p1, p2, p3], dim=1)
        out = self.out_conv(out)
        return out
