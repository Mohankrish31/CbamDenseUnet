import numpy as np
import cv2
import lpips
import torch
from skimage.metrics import structural_similarity as ssim_fn

# C-PSNR
def cpsnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    psnr = 10 * np.log10(1.0 / mse)
    return psnr

# SSIM
def ssim(img1, img2):
    return ssim_fn(img1, img2, channel_axis=2)

# EBCM (Edge-based contrast metric) - placeholder
def ebcm(img1, img2):
    sobel = lambda x: cv2.Sobel(x, cv2.CV_64F, 1, 1, ksize=3)
    gray1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    edge1 = sobel(gray1)
    edge2 = sobel(gray2)
    return np.mean((edge1 - edge2) ** 2)

# LPIPS using VGG
loss_fn = lpips.LPIPS(net='vgg').to("cuda" if torch.cuda.is_available() else "cpu")

def lpips_loss_vgg(pred, target):
    return loss_fn(pred, target).mean()
