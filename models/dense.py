import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=16, num_layers=4):  # changed to 4
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.in_channels = in_channels

        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

        self.total_output_channels = in_channels + num_layers * growth_rate
        # Residual mapping if channels don't match
        if self.total_output_channels != in_channels:
            self.res_conv = nn.Conv2d(self.total_output_channels, in_channels, kernel_size=1, bias=False)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x):
        features = [x]
        for i in range(0, len(self.net), 2):
            out = self.net[i](torch.cat(features, dim=1))
            out = self.net[i + 1](out)
            features.append(out)
        dense_out = torch.cat(features, dim=1)
        residual = self.res_conv(dense_out)
        return x + residual  # Residual connection

# ======= ✅ Test Example ======= #
if __name__ == "__main__":
    x = torch.randn(1, 64, 128, 128)  # batch_size=1, channels=64, H=W=128
    block = DenseBlock(in_channels=64, growth_rate=16, num_layers=4)
    out = block(x)
    print("Input shape :", x.shape)
    print("Output shape:", out.shape)
