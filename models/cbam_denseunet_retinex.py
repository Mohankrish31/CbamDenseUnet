import torch
import torch.nn as nn
import torch.nn.functional as F
from .cbam import cbam
from .dense import denseblock
from .rdb import ResidualDenseBlock
from .multiscale_pool import MultiScalePool

# -----------------------------
# Illumination Corrector (optional learnable)
# -----------------------------
class IlluminationCorrector(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Main Model
# -----------------------------
class cbam_denseunet_retinex(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, use_learnable_illumination=True):
        super(cbam_denseunet_retinex_modified, self).__init__()
        self.use_learnable_illumination = use_learnable_illumination

        # Deeper feature extractor
        dense_out_channels = base_channels + 4 * 16  # increased layers/growth
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            denseblock(base_channels, growth_rate=16, num_layers=4),  # deeper DenseBlock
            cbam(dense_out_channels),
            ResidualDenseBlock(dense_out_channels, growth_channels=16, num_layers=4),  # deeper RDB
            MultiScalePool(dense_out_channels)
        )

        # Illumination corrector
        self.illumination_corrector = IlluminationCorrector(dense_out_channels, 3)

        # Decoder CBAM for attention after correction
        self.cbam_decoder = cbam(3)

        # Learnable skip scaling
        self.alpha = nn.Parameter(torch.tensor(0.5))  # scales input for skip connection

    def forward(self, x):
        # Compute reflectance & illumination
        if self.use_learnable_illumination:
            # Pass original image to feature extractor to learn illumination
            illum_feat = self.feature_extractor(x)
            illumination = self.illumination_corrector(illum_feat)
        else:
            # Traditional max-channel illumination
            illumination = x.max(dim=1, keepdim=True)[0] + 1e-5
            illumination = illumination.repeat(1, 3, 1, 1)

        # Reflectance
        reflectance = x / (illumination + 1e-5)

        # Corrected illumination
        corrected_illumination = illumination

        # Apply CBAM decoder attention
        corrected_illumination = self.cbam_decoder(corrected_illumination)

        # Final output with skip connection
        final_output = corrected_illumination * reflectance + self.alpha * x

        return torch.clamp(final_output, 0, 1)


# -----------------------------
# Test Run
# -----------------------------
if __name__ == "__main__":
    model = cbam_denseunet_retinex()
    inp = torch.rand(2, 3, 224, 224)
    out = model(inp)
    print("Input shape:", inp.shape)
    print("Output shape:", out.shape)
    print("Input brightness (mean):", inp.mean().item())
    print("Output brightness (mean):", out.mean().item())
    print("Input contrast (std):", inp.std().item())
    print("Output contrast (std):", out.std().item())
