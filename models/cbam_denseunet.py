import torch
import torch.nn as nn
from models.cbam import cbam
from models.dense import DenseBlock  # Your improved residual DenseBlock

class cbam_denseunet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, growth_rate=16, num_layers=4):
        super(cbam_denseunet, self).__init__()
        # Encoder: Conv → ReLU → DenseBlock → CBAM
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            DenseBlock(in_channels=base_channels, growth_rate=growth_rate, num_layers=num_layers),
            cbam(base_channels)  # Apply CBAM after DenseBlock
        )

        # Decoder: Conv → ReLU → CBAM → Conv → Sigmoid
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            cbam(base_channels),  # CBAM after ReLU
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec + x  # Residual connection

# ======= ✅ Test Example ======= #
if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    model = cbam_denseunet(in_channels=3, base_channels=32, growth_rate=16, num_layers=4)
    out = model(x)
    print("Input shape :", x.shape)
    print("Output shape:", out.shape)
