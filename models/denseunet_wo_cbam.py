import torch
import torch.nn as nn

# --- Dense Block ---
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=12, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, growth_rate, kernel_size=3, padding=1, bias=False)
            ))
            channels += growth_rate

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)

# --- Transition Layer to adjust channels ---
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)

# --- DenseUNet ---
class DenseUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=32, growth_rate=12):
        super().__init__()
        # Encoder
        self.enc1 = DenseBlock(in_channels, growth_rate)
        self.trans1 = Transition(in_channels + 4 * growth_rate, base)
        self.pool = nn.MaxPool2d(2)

        self.enc2 = DenseBlock(base, growth_rate)
        self.trans2 = Transition(base + 4 * growth_rate, base * 2)

        self.enc3 = DenseBlock(base * 2, growth_rate)
        self.trans3 = Transition(base * 2 + 4 * growth_rate, base * 4)

        # Bottleneck
        self.bottleneck = DenseBlock(base * 4, growth_rate)
        self.bottleneck_trans = Transition(base * 4 + 4 * growth_rate, base * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = nn.Sequential(
            DenseBlock(base * 8, growth_rate),
            Transition(base * 8 + 4 * growth_rate, base * 4)
        )

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            DenseBlock(base * 4, growth_rate),
            Transition(base * 4 + 4 * growth_rate, base * 2)
        )

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = nn.Sequential(
            DenseBlock(base * 2, growth_rate),
            Transition(base * 2 + 4 * growth_rate, base)
        )

        self.final = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.trans1(self.enc1(x))
        e2 = self.trans2(self.enc2(self.pool(e1)))
        e3 = self.trans3(self.enc3(self.pool(e2)))
        # Bottleneck
        b = self.bottleneck_trans(self.bottleneck(self.pool(e3)))
        # Decoder
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)
