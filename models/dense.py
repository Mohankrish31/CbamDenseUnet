import torch
import torch.nn as nn
class denseblock(nn.Module):
    def __init__(self, in_channels, growth_rate=12, num_layers=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels + i * growth_rate, growth_rate, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        features = [x]
        for i in range(0, len(self.net), 2):
            out = self.net[i](torch.cat(features, dim=1))
            out = self.net[i + 1](out)
            features.append(out)
        return torch.cat(features, dim=1)
