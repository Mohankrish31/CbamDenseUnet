import torch
import torch.nn as nn
class EnhancedDecoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(EnhancedDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(x)
