import torch
import torch.nn as nn
from .cbam import cbam
from .dense import denseblock
from .rdb import ResidualDenseBlock  # ✅ Import your RDB module
from .feature_compressor import FeatureCompressor
from .multiscale_pool import MultiScalePool
from .enhanced_decoder import EnhancedDecoder
# === Brightness & Contrast Adjustment Layer (Automatic) ===
class BrightnessContrastAdjust(nn.Module):
    def __init__(self, target_brightness=0.47, target_contrast=0.31):
        """
        target_brightness: desired mean pixel intensity in range [0, 1]
        target_contrast: desired standard deviation of pixel intensity in range [0, 1]
        """
        super().__init__()
        self.target_brightness = target_brightness
        self.target_contrast = target_contrast
    def forward(self, x):
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True)
        # Normalize to target brightness and contrast
        x = (x - mean) / (std + 1e-5) * self.target_contrast + self.target_brightness
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
        # === Automatic Brightness & Contrast Adjustment Layer ===
        self.bc_adjust = BrightnessContrastAdjust(
            target_brightness=0.47,  # ~120 in 0–255 range
            target_contrast=0.31     # ~80 in 0–255 range
        )
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
    inp = torch.randn(8, 3, 500, 574)  
    out = model(inp)
    print("Input shape:", inp.shape)
    print("Output shape:", out.shape)
    print("Output brightness:", out.mean().item())
    print("Output contrast:", out.std().item())
