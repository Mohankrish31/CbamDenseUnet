import torch
import torch.nn as nn
from .cbam import cbam
from .dense import denseblock
from .rdb import ResidualDenseBlock
from .feature_compressor import FeatureCompressor
from .multiscale_pool import MultiScalePool
from .enhanced_decoder import EnhancedDecoder
# === Illumination Corrector ===
class IlluminationCorrector(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, out_channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)
# === Main Model: CBAM DenseUNet with Retinex ===
class cbam_denseunet_retinex(nn.Module):
    def __init__(self, in_channels=3, base_channels=48):
        super().__init__()
        # Dense block output channels
        dense_out_channels = base_channels + 3 * 12  # growth_rate * num_layers
        # Feature extractor works on illumination map (1 channel)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            denseblock(base_channels, growth_rate=12, num_layers=3),
            cbam(dense_out_channels),
            ResidualDenseBlock(dense_out_channels, growth_channels=16, num_layers=3),
            MultiScalePool(dense_out_channels),
        )
        # Illumination corrector
        self.illumination_corrector = IlluminationCorrector(dense_out_channels, 1)
    def forward(self, x):
        # x: RGB input
        illumination = x.max(dim=1, keepdim=True)[0] + 1e-5    # Illumination map (B,1,H,W)
        reflectance = x / illumination                          # Reflectance map (B,3,H,W)
        log_illumination = torch.log(illumination)
        illumination_features = self.feature_extractor(log_illumination)
        corrected_illumination = self.illumination_corrector(illumination_features)
        final_output = corrected_illumination * reflectance
        return torch.clamp(final_output, 0, 1)
# === Test Block ===
if __name__ == "__main__":
    print("Running test of cbam_denseunet_retinex...")
    model = cbam_denseunet_retinex(in_channels=3, base_channels=48)
    print("Model initialized.")
    # Simulate batch of overexposed images
    inp = torch.ones(8, 3, 224, 224) * 0.95
    print(f"Input tensor shape: {inp.shape}")
    out = model(inp)
    print(f"Output tensor shape: {out.shape}")
    print(f"Input mean brightness: {inp.mean().item():.4f}")
    print(f"Input std (contrast): {inp.std().item():.4f}")
    print(f"Output mean brightness: {out.mean().item():.4f}")
    print(f"Output std (contrast): {out.std().item():.4f}")
