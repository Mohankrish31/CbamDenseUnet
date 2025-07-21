# loss_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, pred, target):
        return ssim(pred, target, data_range=1.0, size_average=True)

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.kernel = torch.tensor([[-1, -1, -1],
                                    [-1,  8, -1],
                                    [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel = self.kernel.repeat(3, 1, 1, 1)
        self.kernel.requires_grad = False

    def forward(self, pred, target):
        if pred.shape[1] == 1:  # Grayscale
            pred_edge = F.conv2d(pred, self.kernel[0:1].to(pred.device), padding=1)
            target_edge = F.conv2d(target, self.kernel[0:1].to(target.device), padding=1)
        else:  # RGB
            pred_edge = F.conv2d(pred, self.kernel.to(pred.device), groups=3, padding=1)
            target_edge = F.conv2d(target, self.kernel.to(target.device), groups=3, padding=1)

        return F.l1_loss(pred_edge, target_edge)
