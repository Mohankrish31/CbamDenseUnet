import torch
import torch.nn as nn
from models.cbam import cbam
from models.dense import denseblock
class cbam_denseunet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            denseblock(base_channels, growth_rate=12, num_layers=3),
            cbam(base_channels + 3 * 12)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels + 3 * 12, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        enc = self.encoder(x)
        out = self.decoder(enc)
        return out + x
