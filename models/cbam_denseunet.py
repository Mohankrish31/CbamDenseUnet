import torch
import torch.nn as nn
from .cbam import cbam
from .dense import denseblock
from .rdb import ResidualDenseBlock
from .feature_compressor import FeatureCompressor
from .multiscale_pool import MultiScalePool
from .enhanced_decoder import EnhancedDecoder

# === Illumination Corrector (New) ===
# This decoder processes the features to generate a corrected illumination map.
class IlluminationCorrector(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x

# === Main Model (Modified for Retinex-inspired Enhancement) ===
class cbam_denseunet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super(cbam_denseunet, self).__init__()
        # Dense block output channels
        dense_out_channels = base_channels + 3 * 12

        # === Feature Extractor (Re-purposed Encoder) ===
        # This will now extract features from the log-transformed illumination map
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            denseblock(base_channels, growth_rate=12, num_layers=3),
            cbam(dense_out_channels),
            ResidualDenseBlock(dense_out_channels, growth_channels=16, num_layers=3),
            MultiScalePool(dense_out_channels),
        )

        # === Illumination Corrector ===
        # A separate decoder to correct the illumination map
        self.illumination_corrector = IlluminationCorrector(dense_out_channels, in_channels)

    def forward(self, x):
        # 1. Decompose the image into illumination and reflectance based on Retinex theory.
        # Illumination is estimated as the max value across the color channels.
        illumination = x.max(dim=1, keepdim=True)[0] + 1e-5
        reflectance = x / illumination

        # 2. Process the illumination map to correct brightness issues.
        # Log-transform is used to stabilize the dynamic range for processing.
        log_illumination = torch.log(illumination)

        # Use the repurposed encoder to process the log illumination map.
        illumination_features = self.feature_extractor(log_illumination)

        # Decode the features back to a corrected illumination map.
        corrected_illumination = self.illumination_corrector(illumination_features)
        
        # 3. Reconstruct the final image.
        # Multiply the corrected illumination with the original reflectance.
        final_output = corrected_illumination * reflectance

        # Clamp the output to ensure values are in a valid range [0, 1].
        return torch.clamp(final_output, 0, 1)
# === Test Run ===
if __name__ == "__main__":
    # Simulate a batch of overexposed images.
    overexposed_image = torch.ones(8, 3, 256, 256) * 0.95
    print("Input image mean:", overexposed_image.mean().item())

    # Initialize the modified model.
    model = cbam_denseunet(in_channels=3, base_channels=32)

    # Pass the overexposed image through the model.
    output_image = model(overexposed_image)
    
    # Print the output details to verify correction. The output mean should be lower.
    print("Output image shape:", output_image.shape)
    print("Output image mean (should be lower):", output_image.mean().item())
