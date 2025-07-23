import torch
import torch.nn.functional as F
import lpips
import kornia
from pytorch_msssim import ssim

# ✅ MSE Loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target)

# ✅ SSIM Loss
def ssim_loss(pred, target):
    return 1 - ssim(pred, target, data_range=1.0, size_average=True)

# ✅ LPIPS Loss using VGG
lpips_vgg = lpips.LPIPS(net='vgg').to('cuda' if torch.cuda.is_available() else 'cpu')
def lpips_loss_vgg(pred, target):
    return lpips_vgg(pred, target).mean()

# ✅ Sobel Edge Loss
def edge_loss_sobel(pred, target):
    pred_edges = kornia.filters.Sobel()(pred)
    target_edges = kornia.filters.Sobel()(target)
    return F.l1_loss(pred_edges, target_edges)
