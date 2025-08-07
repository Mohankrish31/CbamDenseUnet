import torch
import torch.nn as nn
class FeatureCompressor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureCompressor, self).__init__()
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.compress(x)
