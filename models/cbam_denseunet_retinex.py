import torch
import torch.nn as nn
from .cbam import cbam
from .dense import denseblock
from .rdb import ResidualDenseBlock
from .feature_compressor import FeatureCompressor
from .multiscale_pool import MultiScalePool
from .enhanced_decoder import EnhancedDecoder
# === Illumination Corrector (Modified and Enhanced) ===
# This version has more layers to learn a more complex mapping for illumination correction.
class IlluminationCorrector(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # *** Change 1: The final layer must output 3 channels for a 3-channel illumination map. ***
            nn.Conv2d(in_channels // 4, out_channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# === Main Model: cbam_denseunet with Retinex ===
class cbam_denseunet_retinex(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super(cbam_denseunet_retinex, self).__init__()
        
        dense_out_channels = base_channels + 3 * 12

        # This feature extractor will now be used for the 3-channel illumination map.
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            denseblock(base_channels, growth_rate=12, num_layers=3),
            cbam(dense_out_channels),
            ResidualDenseBlock(dense_out_channels, growth_channels=16, num_layers=3),
            MultiScalePool(dense_out_channels),
        )

        # *** Change 2: The IlluminationCorrector is now initialized to output 3 channels. ***
        # This allows it to correct each color channel independently.
        self.illumination_corrector = IlluminationCorrector(dense_out_channels, 3)

    def forward(self, x):
        # The illumination and reflectance are still derived from the original 3-channel image
        illumination = x.max(dim=1, keepdim=True)[0] + 1e-5
        reflectance = x / illumination

        log_illumination = torch.log(illumination)

        # Replicate the single-channel log_illumination map to have 3 channels
        illumination_3channel = log_illumination.repeat(1, 3, 1, 1)

        # Pass the 3-channel illumination to the feature extractor
        # It will now work as it's configured for 3-channel input
        illumination_features = self.feature_extractor(illumination_3channel)

        # The corrected_illumination is now a 3-channel tensor
        corrected_illumination = self.illumination_corrector(illumination_features)
        
        final_output = corrected_illumination * reflectance

        return torch.clamp(final_output, 0, 1)

# === Main Test Block ===
if __name__ == "__main__":
    # Test Run
    print("Running a test of the cbam_denseunet_retinex model...")

    # Set up the model for a test run
    model = cbam_denseunet_retinex(in_channels=3, base_channels=32)
    print("Model initialized.")

    # Simulate a batch of overexposed images with your desired dimensions
    # Batch size: 8, Channels: 3, Height: 224, Width: 224
    try:
        inp = torch.ones(8, 3, 224, 224) * 0.95
        print(f"Dummy input tensor created with shape: {inp.shape}")
        
        # Pass the simulated overexposed image through the model
        out = model(inp)
        
        # Print the output details to verify correction
        print(f"Output shape: {out.shape}")
        print(f"Input brightness (mean): {inp.mean().item():.4f}")
        print(f"Input contrast (std): {inp.std().item():.4f}")
        print(f"Output brightness (mean): {out.mean().item():.4f}")
        print(f"Output contrast (std): {out.std().item():.4f}")
        
    except RuntimeError as e:
        print(f"\nAn error occurred during the forward pass: {e}")
        print("Please double-check the model's architecture for any channel or dimension mismatches.")
