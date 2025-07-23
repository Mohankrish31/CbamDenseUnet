import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super(DenseBlock, self).__init__()
        self.blocks = nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers):
            self.blocks.append(self._make_layer(channels, growth_rate))
            channels += growth_rate

    def _make_layer(self, in_channels, growth_rate):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )
        return layer

    def forward(self, x):
        features = [x]
        for layer in self.blocks:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.dense = DenseBlock(in_channels, growth_rate=32)

    def forward(self, x):
        x = self.pool(x)
        return self.dense(x)


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.dense = DenseBlock(out_channels + skip_channels, growth_rate=32)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.dense(x)


class DenseUNet_NoCBAM(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(DenseUNet_NoCBAM, self).__init__()
        self.inc = DenseBlock(in_channels, growth_rate=32)

        self.down1 = Down(3 * 32 + in_channels, 64)
        self.down2 = Down(3 * 32 + 32, 128)
        self.down3 = Down(3 * 32 + 32, 256)

        self.bottleneck = DenseBlock(3 * 32 + 32, growth_rate=32)

        self.up3 = Up(3 * 32 + 32, 3 * 32 + 32, 128)
        self.up2 = Up(3 * 32 + 32, 3 * 32 + 32, 64)
        self.up1 = Up(3 * 32 + 32, 3 * 32 + in_channels, 32)

        self.outc = nn.Conv2d(3 * 32 + 32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x_bottleneck = self.bottleneck(x4)

        x = self.up3(x_bottleneck, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        return self.outc(x)
