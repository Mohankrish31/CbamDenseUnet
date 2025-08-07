import torch
import torch.nn as nn
from models.cbam import cbam
from models.dense import denseblock
from models.feature_compressor import FeatureCompressor
from models.multiscale_pool import MultiScalePool
from models.enhanced_decoder import EnhancedDecoder
class cbam_denseunet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super(cbam_denseunet, self).__init__()
        dense_out_channels = base_channels + 3 * 12  # denseblock + growth
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            denseblock(base_channels, growth_rate=12, num_layers=3),
            cbam(dense_out_channels),
            MultiScalePool(dense_out_channels)
        )
        self.feature_compression = FeatureCompressor(dense_out_channels, base_channels)
        self.decoder = EnhancedDecoder(
            in_channels=base_channels,
            mid_channels=base_channels // 2,
            out_channels=in_channels
        )
    def forward(self, x):
        enc = self.encoder(x)
        compressed = self.feature_compression(enc)
        dec = self.decoder(compressed)
        return dec + x
