import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedDecoder(nn.Module):
    """
    Enhanced decoder module for UNet-like architectures.
    Upsamples features, optionally fuses skip connections,
    and refines with convolution blocks.
    """
    def __init__(self, in_channels, skip_channels=0, out_channels=None, use_bn=True):
        """
        Args:
            in_channels (int): Number of input channels from previous layer.
            skip_channels (int): Number of channels from skip connection (default 0 if no skip).
            out_channels (int): Output channels after decoder block.
            use_bn (bool): Whether to use BatchNorm.
        """
        super(EnhancedDecoder, self).__init__()
        if out_channels is None:
            out_channels = in_channels // 2

        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, skip=None):
        """
        x: input feature map
        skip: skip connection feature map (optional)
        """
        if skip is not None:
            x = torch.cat([x, skip], dim=1)  # Concatenate along channel dimension
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.up(x)
        return x


# ===== Example Usage =====
if __name__ == "__main__":
    x = torch.randn(2, 128, 32, 32)   # Input features
    skip = torch.randn(2, 64, 32, 32) # Skip connection features
    decoder = EnhancedDecoder(in_channels=128, skip_channels=64, out_channels=64)
    out = decoder(x, skip)
    print(f"Output shape: {out.shape}")  # Expected: [2, 64, 64, 64] after upsample
