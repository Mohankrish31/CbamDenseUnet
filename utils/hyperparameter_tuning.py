import torch
import os
# Updated loss weights
LOSS_WEIGHTS = {
    "mse": 1.0,
    "ssim": 0.4,
    "lpips": 0.3,
    "edge": 0.2  # General edge loss (e.g., Sobel or Laplacian)
}
def get_optimizer(model, lr=1e-4, weight_decay=0):
    """
    Returns Adam optimizer.
    """
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
def get_scheduler(optimizer, scheduler_type="plateau", patience=5, factor=0.5):
    """
    Returns a learning rate scheduler.
    """
    if scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience, verbose=True)
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5)
    else:
        return None
def update_loss_weights(epoch, total_epochs, start_weights, end_weights):
    """
    Linearly interpolate loss weights over training.
    Useful if you want to focus more on perceptual loss over time.
    """
    updated_weights = {}
    for key in start_weights:
        start = start_weights[key]
        end = end_weights.get(key, start)
        interpolated = start + (end - start) * (epoch / total_epochs)
        updated_weights[key] = interpolated
    return updated_weights
def create_save_path(base_path="saved_models/", model_name="CBAM_DenseUNet.pth"):
    """
    Ensures model save path exists.
    """
    os.makedirs(base_path, exist_ok=True)
    return os.path.join(base_path, model_name)

