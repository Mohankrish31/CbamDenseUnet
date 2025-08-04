class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=12, layers=4):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1)
            ))

    def forward(self, x):
        features = [x]
        for layer in self.blocks:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)

class DenseUNet(UNet):
    def __init__(self, in_channels=3, out_channels=1, base=64):
        super().__init__(in_channels, out_channels, base)
        self.enc1 = DenseBlock(in_channels, base)
        self.enc2 = DenseBlock(base + 4 * 12, base*2)
        self.enc3 = DenseBlock(base*2 + 4 * 12, base*4)
        self.bottleneck = DenseBlock(base*4 + 4 * 12, base*8)
        self.dec3 = DenseBlock(base*8 + base*4 + 4 * 12, base*4)
        self.dec2 = DenseBlock(base*4 + base*2 + 4 * 12, base*2)
        self.dec1 = DenseBlock(base*2 + base + 4 * 12, base)
