import torch
import torch.nn.functional as F
from math import log10
from lpips import LPIPS
from pytorch_msssim import ssim
import numpy as np
# Initialize LPIPS once (VGG-based)
lpips_fn = LPIPS(net='vgg').cuda() if torch.cuda.is_available() else LPIPS(net='vgg')
def compute_mse(pred, target):
    return F.mse_loss(pred, target).item()
def compute_psnr(pred, target):
    mse = compute_mse(pred, target)
    if mse == 0:
        return float('inf')
    return 10 * log10(1.0 / mse)
def compute_ssim(pred, target):
    with torch.no_grad():
        return ssim(pred, target, data_range=1.0, size_average=True).item()
def compute_lpips(pred, target):
    with torch.no_grad():
        return lpips_fn(pred, target).mean().item()
def compute_edge_loss(pred, target):
    """Approximate EBCM with Sobel-based edge loss."""
    def sobel(img):
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
        if img.shape[1] != 1:
            img = img.mean(dim=1, keepdim=True)  # Convert to grayscale
        gx = F.conv2d(img, sobel_x, padding=1)
        gy = F.conv2d(img, sobel_y, padding=1)
        edge = torch.sqrt(gx ** 2 + gy ** 2)
        return edge
    pred_edge = sobel(pred)
    target_edge = sobel(target)
    return F.l1_loss(pred_edge, target_edge).item()
def evaluate_metrics(pred, target):
    """
    Returns all key metrics in a dictionary.
    """
    return {
        "C-PSNR": compute_psnr(pred, target),
        "SSIM": compute_ssim(pred, target),
        "LPIPS": compute_lpips(pred, target),
        "Edge Loss": compute_edge_loss(pred, target)
    }
