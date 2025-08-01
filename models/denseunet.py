# models/denseunet.py
from torch import nn
from collections import OrderedDict

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate=32):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.layer(x)
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers=4, growth_rate=32):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# Replace UNetBlock in UNet with DenseBlock in encoder and bottleneck
# Decoder remains same as baseline for simplicity

