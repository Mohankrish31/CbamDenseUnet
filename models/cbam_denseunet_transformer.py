# cbam_denseunet_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cbam import cbam
from .dense import denseblock
from .rdb import ResidualDenseBlock
from .multiscale_pool import MultiScalePool

# -------------------------
# Basic conv block
# -------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# -------------------------
# Down block
# -------------------------
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_ch, out_ch, k=3, s=2, p=1),  # stride 2
            ConvBNAct(out_ch, out_ch, k=3, s=1, p=1)
        )
    def forward(self, x):
        return self.net(x)

# -------------------------
# Transformer helper (small)
# -------------------------
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # no strict divisibility requirement here; implementation pads if needed
    def forward(self, x):
        # x: (B,C,H,W)
        b, c, h, w = x.shape
        device = x.device
        pe = torch.zeros(1, c, h, w, device=device)
        c4 = max(1, c // 4)
        y = torch.arange(h, device=device).float().unsqueeze(1).repeat(1, w)
        x_pos = torch.arange(w, device=device).float().unsqueeze(0).repeat(h, 1)
        div = torch.exp(torch.arange(0, c4, device=device).float() * (-(torch.log(torch.tensor(10000.0)) / c4)))
        pe[:, 0:c4, :, :] = torch.sin(y.unsqueeze(0).unsqueeze(0) * div.view(1, -1, 1, 1))
        pe[:, c4:2*c4, :, :] = torch.cos(y.unsqueeze(0).unsqueeze(0) * div.view(1, -1, 1, 1))
        pe[:, 2*c4:3*c4, :, :] = torch.sin(x_pos.unsqueeze(0).unsqueeze(0) * div.view(1, -1, 1, 1))
        pe[:, 3*c4:4*c4, :, :] = torch.cos(x_pos.unsqueeze(0).unsqueeze(0) * div.view(1, -1, 1, 1))
        if c > 4*c4:
            pe[:, 4*c4:, :, :] = 0.0
        return x + pe

class SmallTransformer(nn.Module):
    def __init__(self, dim, heads=4, layers=1, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.pos = PositionalEncoding2D(dim)
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)   # (B, HW, C)
        x = self.enc(x)
        x = self.norm(x)
        x = x.transpose(1, 2).view(b, c, h, w)
        return x

# -------------------------
# Up block with CBAM + Transformer refinement
# -------------------------
class UpCBAMTrans(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, trans_heads=4):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            ConvBNAct(in_ch + skip_ch, out_ch, k=3, p=1),
            ConvBNAct(out_ch, out_ch, k=3, p=1)
        )
        self.cbam = cbam(out_ch)
        # small transformer: 1 layer to keep memory low
        self.trans = SmallTransformer(out_ch, heads=trans_heads, layers=1, mlp_ratio=2.0)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.cbam(x)
        x = self.trans(x)
        return x

# -------------------------
# Edge head
# -------------------------
class EdgeHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_ch, max(8, in_ch//2)),
            nn.Conv2d(max(8, in_ch//2), 1, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# -------------------------
# Main hybrid model
# -------------------------
class cbam_denseunet_transformer(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, growth_rate=12, dense_layers=3, trans_heads=4):
        """
        base_channels: controls model width (32 recommended for moderate GPUs)
        growth_rate, dense_layers: Dense block config
        """
        super().__init__()
        self.eps = 1e-5

        # input conv on log-illumination (1 channel)
        self.in_conv = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        # Encoder stage 1
        self.db1 = denseblock(base_channels, growth_rate=growth_rate, num_layers=dense_layers)
        ch1 = base_channels + dense_layers * growth_rate
        self.cbam1 = cbam(ch1)
        self.rdb1 = ResidualDenseBlock(ch1, growth_channels=16, num_layers=3)
        self.msp1 = MultiScalePool(ch1)
        self.down1 = Down(ch1, base_channels*2)

        # Encoder stage 2
        self.db2 = denseblock(base_channels*2, growth_rate=growth_rate, num_layers=dense_layers)
        ch2 = base_channels*2 + dense_layers * growth_rate
        self.cbam2 = cbam(ch2)
        self.rdb2 = ResidualDenseBlock(ch2, growth_channels=16, num_layers=3)
        self.msp2 = MultiScalePool(ch2)
        self.down2 = Down(ch2, base_channels*4)

        # Encoder / Bottleneck stage 3
        self.db3 = denseblock(base_channels*4, growth_rate=growth_rate, num_layers=dense_layers)
        ch3 = base_channels*4 + dense_layers * growth_rate
        self.cbam3 = cbam(ch3)
        self.rdb3 = ResidualDenseBlock(ch3, growth_channels=16, num_layers=3)
        self.msp3 = MultiScalePool(ch3)

        # Transformer bottleneck (global)
        self.trans_bottleneck = SmallTransformer(ch3, heads=trans_heads, layers=2, mlp_ratio=2.0)

        # Decoder (UNet-style) with CBAM + Transformer in each Up block
        self.up2 = UpCBAMTrans(in_ch=ch3, skip_ch=ch2, out_ch=base_channels*2, trans_heads=trans_heads)
        self.up1 = UpCBAMTrans(in_ch=base_channels*2, skip_ch=ch1, out_ch=base_channels, trans_heads=trans_heads)

        # refine
        self.refine = nn.Sequential(
            ConvBNAct(base_channels, base_channels),
            cbam(base_channels)
        )

        # illumination head & edge head
        self.illum_head = nn.Sequential(nn.Conv2d(base_channels, 1, kernel_size=1), nn.Sigmoid())
        self.edge_head = EdgeHead(base_channels)

        # learnable blend
        self._alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # x: (B,3,H,W) in [0,1]
        illum = x.max(dim=1, keepdim=True)[0] + self.eps   # (B,1,H,W)
        reflect = x / illum

        z = torch.log(illum)
        z = self.in_conv(z)   # (B, base, H, W)

        # enc1
        e1 = self.db1(z)
        e1 = self.cbam1(e1)
        e1 = self.rdb1(e1)
        e1 = self.msp1(e1)
        d1 = self.down1(e1)

        # enc2
        e2 = self.db2(d1)
        e2 = self.cbam2(e2)
        e2 = self.rdb2(e2)
        e2 = self.msp2(e2)
        d2 = self.down2(e2)

        # enc3 / bottleneck
        b = self.db3(d2)
        b = self.cbam3(b)
        b = self.rdb3(b)
        b = self.msp3(b)

        # bottleneck transformer
        b = self.trans_bottleneck(b)

        # decoder
        u2 = self.up2(b, e2)
        u1 = self.up1(u2, e1)

        feat = self.refine(u1)

        corrected_illum = self.illum_head(feat)
        edge_pred = self.edge_head(feat)

        enhanced = corrected_illum * reflect

        alpha = torch.sigmoid(self._alpha)
        y = alpha * enhanced + (1.0 - alpha) * x
        y = torch.clamp(y, 0.0, 1.0)

        return y, edge_pred

# -------------------------
# Sanity/test run
# -------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = cbam_denseunet_transformer(in_channels=3, base_channels=32, growth_rate=12, dense_layers=3, trans_heads=4).to(device)
    inp = torch.rand(2,3,224,224).to(device)
    out, edge = model(inp)
    print("Input:", inp.shape, "Output:", out.shape, "Edge:", edge.shape)
    print("alpha:", torch.sigmoid(model._alpha).item())
