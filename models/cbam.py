import torch
import torch.nn as nn
import torch.nn.functional as F
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)               # (B,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)             # (B,1,H,W)
        x_cat = torch.cat([avg_out, max_out], dim=1)               # (B,2,H,W)
        out = self.conv(x_cat)
        return self.sigmoid(out)
class cbam(nn.Module):
    def __init__(self, channels, ratio=8, sa_kernel=7, use_residual=True):
        super(cbam, self).__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size=sa_kernel)
        self.use_residual = use_residual
    def forward(self, x):
        ca_out = self.ca(x) * x
        sa_out = self.sa(ca_out) * ca_out
        return x + sa_out if self.use_residual else sa_out
