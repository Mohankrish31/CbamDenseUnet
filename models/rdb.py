# models/rdb.py
import torch
import torch.nn as nn
import torch.nn.functional as F
class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (RDB).
    Structure:
      - num_layers of: Conv3x3(in -> growth, padding=1) + ReLU, where input to each layer
        is concatenation of original input and all previous layer outputs.
      - Local Feature Fusion (LFF): 1x1 conv to reduce concatenated channels back to in_channels.
      - Residual connection: output = input + residual_scale * LFF_output.
    Args:
        in_channels (int): number of input channels.
        growth_channels (int): growth rate (channels added per conv layer).
        num_layers (int): number of dense conv layers inside the RDB.
        residual_scale (float): scale applied to the residual before adding back to input (default 0.2).
    """
    def __init__(self, in_channels, growth_channels=16, num_layers=4, residual_scale=0.2):
        super(ResidualDenseBlock, self).__init__()
        self.in_channels = in_channels
        self.growth_channels = growth_channels
        self.num_layers = num_layers
        self.residual_scale = residual_scale
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            # input channels to this conv = in_channels + i * growth_channels
            in_ch = in_channels + i * growth_channels
            self.convs.append(
                nn.Conv2d(in_ch, growth_channels, kernel_size=3, stride=1, padding=1, bias=True)
            )
        # Local Feature Fusion: compress concatenated (in + num_layers*growth) -> in_channels
        self.lff = nn.Conv2d(in_channels + num_layers * growth_channels, in_channels,
                             kernel_size=1, stride=1, padding=0, bias=True)
        # initialization (optional but helpful)
        self._init_weights()
    def _init_weights(self):
        # Kaiming for convs, small for LFF
        for m in self.convs:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if isinstance(self.lff, nn.Conv2d):
            nn.init.kaiming_normal_(self.lff.weight, a=0, mode='fan_in', nonlinearity='relu')
            if self.lff.bias is not None:
                nn.init.zeros_(self.lff.bias)
    def forward(self, x):
        features = [x]
        for conv in self.convs:
            inp = torch.cat(features, dim=1)
            out = F.relu(conv(inp), inplace=True)
            features.append(out)
        concat = torch.cat(features, dim=1)  # shape: in + num_layers*growth
        lff_out = self.lff(concat)
        return x + self.residual_scale * lff_out
