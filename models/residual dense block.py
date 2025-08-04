class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels

        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(channels, growth_rate, 3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            channels += growth_rate

        self.local_fusion = nn.Conv2d(channels, in_channels, kernel_size=1)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        fused = self.local_fusion(torch.cat(features, dim=1))
        return x + fused  # Residual connection
