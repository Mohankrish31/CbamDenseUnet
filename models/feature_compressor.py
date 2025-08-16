import torch
import torch.nn as nn
import torch.nn.functional as F
class FeatureCompressor(nn.Module):
    """
    Compresses feature maps by reducing the number of channels using 1x1 conv.
    Optionally, you can add batch normalization and activation.
    """
    def __init__(self, in_channels, out_channels, use_bn=True, activation=nn.ReLU(inplace=True)):
        super(FeatureCompressor, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            layers.append(activation)
        self.compressor = nn.Sequential(*layers)

    def forward(self, x):
        return self.compressor(x)


# ===== Example Usage =====
if __name__ == "__main__":
    x = torch.randn(2, 128, 64, 64)  # batch of 2, 128 channels, 64x64
    compressor = FeatureCompressor(in_channels=128, out_channels=32)
    out = compressor(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")  # Expected: [2, 32, 64, 64]
