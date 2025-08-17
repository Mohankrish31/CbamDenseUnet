import torch
import torch.nn as nn
import torch.nn.functional as F
from .cbam import cbam
from .dense import denseblock
from .rdb import ResidualDenseBlock
from .multiscale_pool import MultiScalePool
# =========================
# Decoder with CBAM inside
# =========================
class EnhancedDecoderCBAM(nn.Module):
    """
    A lightweight decoder/refiner block that applies conv -> CBAM -> conv.
    Keeps spatial size; use multiple stacks if you want deeper decoding.
    """
    def __init__(self, in_channels, mid_channels=None, out_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.attn = cbam(mid_channels)          # <<—— CBAM in decoder
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.attn(x)
        x = self.relu2(self.conv2(x))
        return x

# =========================
# Stronger Illumination Head
# =========================
class IlluminationCorrector(nn.Module):
    """
    Stronger, context-aware illumination predictor:
    RDB + CBAM -> 1x1 conv -> sigmoid
    """
    def __init__(self, in_channels, out_channels=1, growth=16):
        super().__init__()
        self.body = nn.Sequential(
            ResidualDenseBlock(in_channels, growth_channels=growth, num_layers=3),
            cbam(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.body(x)

# ===============================================
# Main Model: CBAM DenseUNet with Retinex + CBAM
# ===============================================
class cbam_denseunet_retinex(nn.Module):
    """
    - Encoder side: Dense + CBAM + RDB + MultiScalePool on log-illumination.
    - Decoder side: EnhancedDecoderCBAM to refine features before predicting illumination.
    - IlluminationCorrector: stronger head for robust, natural exposure.
    - Learnable skip scaling (alpha): blends enhanced result with original input to avoid
      under/over enhancement. alpha in (0,1) via sigmoid.
    """
    def __init__(self, in_channels=3, base_channels=48):
        super().__init__()
        self.eps = 1e-5

        # Dense block output channels (match your denseblock config)
        growth_rate = 12
        num_layers = 3
        dense_out_channels = base_channels + num_layers * growth_rate

        # -------- Encoder / feature extractor on log-illumination (1ch) --------
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            denseblock(base_channels, growth_rate=growth_rate, num_layers=num_layers),
            cbam(dense_out_channels),
            ResidualDenseBlock(dense_out_channels, growth_channels=16, num_layers=3),
            MultiScalePool(dense_out_channels)   # should keep spatial size; if it changes, add upsampling below
        )

        # -------- Decoder with CBAM to refine illumination features --------
        self.decoder = EnhancedDecoderCBAM(
            in_channels=dense_out_channels,
            mid_channels=dense_out_channels,
            out_channels=dense_out_channels
        )

        # -------- Illumination corrector (predicts 1-channel correction) --------
        self.illumination_corrector = IlluminationCorrector(dense_out_channels, out_channels=1)

        # -------- Learnable skip scaling (blend original vs enhanced) --------
        # Initialize near 0.0 so sigmoid ≈ 0.5 (balanced start)
        self._alpha = nn.Parameter(torch.tensor(0.0))  # alpha = sigmoid(_alpha) ∈ (0,1)

    def forward(self, x):
        """
        x: RGB in [0,1], shape (B,3,H,W)
        returns: enhanced RGB in [0,1]
        """
        # --- Retinex split ---
        illumination = x.max(dim=1, keepdim=True)[0] + self.eps        # (B,1,H,W)
        reflectance = x / illumination                                  # (B,3,H,W)

        # --- Feature extraction on log-illumination ---
        log_illum = torch.log(illumination)
        f = self.feature_extractor(log_illum)

        # If MultiScalePool changes spatial size, upsample back:
        if f.shape[-2:] != x.shape[-2:]:
            f = F.interpolate(f, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # --- Decoder with CBAM ---
        f_refined = self.decoder(f)

        # --- Predict corrected illumination (0..1) ---
        corrected_illum = self.illumination_corrector(f_refined)

        # --- Recompose enhanced image via Retinex ---
        enhanced = corrected_illum * reflectance

        # --- Learnable skip scaling: blend with original input ---
        alpha = torch.sigmoid(self._alpha)  # scalar in (0,1)
        y = alpha * enhanced + (1.0 - alpha) * x

        return torch.clamp(y, 0.0, 1.0)

# =========================
# Quick sanity test
# =========================
if __name__ == "__main__":
    print("Running test of cbam_denseunet_retinex (with decoder CBAM + skip scaling)...")
    model = cbam_denseunet_retinex(in_channels=3, base_channels=48)
    inp = torch.rand(2, 3, 224, 224) * 0.4  # simulate darker inputs
    out = model(inp)
    print("Input:", inp.shape, "Output:", out.shape)
    print("alpha (blend):", torch.sigmoid(model._alpha).item())
    print("Input mean:", inp.mean().item(), "Output mean:", out.mean().item())
