import numpy as np
import cv2
import lpips
import torch
from skimage.metrics import structural_similarity as ssim_fn

# Initialize LPIPS loss once
loss_fn_alex = lpips.LPIPS(net='vgg').to("cuda" if torch.cuda.is_available() else "cpu")

def cpsnr(img1, img2):
    """Compute Contrast-aware PSNR"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    psnr = 10 * np.log10(1.0 / mse)
    return psnr

def ssim(img1, img2):
    """Compute Structural Similarity"""
    return ssim_fn(img1, img2, channel_axis=2, data_range=1.0)

def ebcm(img1, img2):
    """Edge-based contrast metric (Sobel edge difference)"""
    sobel = lambda x: cv2.Sobel(x, cv2.CV_64F, 1, 1, ksize=3)
    gray1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    edge1 = sobel(gray1)
    edge2 = sobel(gray2)
    return np.mean((edge1 - edge2) ** 2)

def lpips_loss_vgg(pred, target):
    """Compute LPIPS using VGG network (inputs must be normalized tensors [B,C,H,W])"""
    return loss_fn_alex(pred, target).mean().item()

def compute_metrics(pred_np, target_np, pred_tensor=None, target_tensor=None):
    """
    pred_np, target_np: numpy images in [H, W, 3], range [0,1]
    pred_tensor, target_tensor: torch tensors in [B,3,H,W], range [-1,1] (for LPIPS)
    """
    c_psnr = cpsnr(pred_np, target_np)
    ssim_val = ssim(pred_np, target_np)
    ebcm_val = ebcm(pred_np, target_np)
    
    lpips_val = None
    if pred_tensor is not None and target_tensor is not None:
        lpips_val = lpips_loss_vgg(pred_tensor, target_tensor)
    
    return {
        "C-PSNR": c_psnr,
        "SSIM": ssim_val,
        "EBCM": ebcm_val,
        "LPIPS": lpips_val
    }
