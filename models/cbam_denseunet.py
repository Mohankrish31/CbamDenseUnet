import torch
import torch.nn as nn
from .cbam import cbam
from .dense import denseblock
from .rdb import ResidualDenseBlock  # ✅ Import your RDB module
from .feature_compressor import FeatureCompressor
from .multiscale_pool import MultiScalePool
from .enhanced_decoder import EnhancedDecoder
# === Brightness & Contrast Adjustment Layer ===
class BrightnessContrastAdjust(nn.Module):
    def __init__(self):
        super().__init__()
        self.brightness = nn.Parameter(torch.zeros(1, 3, 1, 1))  # learnable shift
        self.contrast = nn.Parameter(torch.ones(1, 3, 1, 1))     # learnable scale
    def forward(self, x):
        # Adjust contrast around mean 0.5, then shift brightness
        x = self.contrast * (x - 0.5) + 0.5 + self.brightness
        return torch.clamp(x, 0, 1)
# === Main Model ===
class cbam_denseunet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super(cbam_denseunet, self).__init__()
        # Dense block output channels
        dense_out_channels = base_channels + 3 * 12  # base + growth from denseblock
        # === Encoder ===
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            denseblock(base_channels, growth_rate=12, num_layers=3),  # DenseBlock
            cbam(dense_out_channels),                                 # CBAM
            ResidualDenseBlock(dense_out_channels, growth_channels=16, num_layers=3),  # ✅ RDB
            MultiScalePool(dense_out_channels)                        # Multi-scale pooling
        )
        # === Feature Compression ===
        self.feature_compression = FeatureCompressor(dense_out_channels, base_channels)
        # === Decoder (remove Sigmoid at the end) ===
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, in_channels, kernel_size=3, padding=1),
            nn.Identity()  # No activation here; BC adjust will handle final range
        )
        # === Learnable Skip Scaling ===
        self.skip_scale = nn.Parameter(torch.ones(1))
        # === Illumination Adjustment Layer ===
        self.illum_adjust = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        # === Brightness & Contrast Adjustment Layer ===
        self.bc_adjust = BrightnessContrastAdjust()
    def forward(self, x):
        # Encoder
        enc = self.encoder(x)
        # Feature Compression
        compressed = self.feature_compression(enc)
        # Decoder output
        dec = self.decoder(compressed)
        # Apply learnable skip scaling
        out = dec * self.skip_scale + x * (1 - self.skip_scale)
        # Apply illumination adjustment
        out = out * self.illum_adjust(out)
        # Apply brightness & contrast adjustment
        out = self.bc_adjust(out)
        return out
# === Test Run ===
if __name__ == "__main__":
    model = cbam_denseunet(in_channels=3, base_channels=32)
    inp = torch.randn(1, 3, 224, 224)  # Dummy input
    out = model(inp)
    print("Input shape:", inp.shape)
    print("Output shape:", out.shape)
