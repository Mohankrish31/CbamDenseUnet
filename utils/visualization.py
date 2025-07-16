import os
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid, save_image
def show_image_grid(tensors, titles=None, nrow=3, save_path=None):
    """
    Display or save a grid of images from a batch of tensors.
    Args:
        tensors (list of torch.Tensor): List of image tensors (B, C, H, W) or (C, H, W)
        titles (list of str): Optional titles for each image.
        nrow (int): Number of images per row in the grid.
        save_path (str): If provided, saves the grid image.
    """
    if isinstance(tensors[0], torch.Tensor) and tensors[0].dim() == 3:
        tensors = [t.unsqueeze(0) for t in tensors]  # Make (1, C, H, W)
    grid = make_grid(torch.cat(tensors, dim=0), nrow=nrow, normalize=True, scale_each=True)
    plt.figure(figsize=(12, 6))
    plt.axis('off')
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    if titles:
        plt.title(" | ".join(titles))
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
def plot_metrics(history, save_path=None):
    """
    Plot training/validation losses and metrics over epochs.

    Args:
        history (dict): Dictionary with keys like 'loss', 'ssim', 'psnr', etc.
        save_path (str): If given, saves the plot to a file.
    """
    plt.figure(figsize=(10, 6))
    for key, values in history.items():
        plt.plot(values, label=key)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Metrics Over Time')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"📊 Saved plot to: {save_path}")
    else:
        plt.show()
def save_comparison(input_tensor, output_tensor, target_tensor, out_path):
    """
    Save a side-by-side comparison of input, output, and target.
    Args:
        input_tensor: low-light input (1, C, H, W)
        output_tensor: enhanced output (1, C, H, W)
        target_tensor: ground truth (1, C, H, W)
        out_path: output image file path
    """
    comparison = torch.cat([input_tensor, output_tensor, target_tensor], dim=0)
    grid = make_grid(comparison, nrow=3, normalize=True)
    save_image(grid, out_path)
    print(f"Saved comparison to: {out_path}")
