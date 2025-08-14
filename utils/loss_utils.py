# === Imports and Setup ===
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import lpips
import numpy as np
# Ensure your model is correctly imported from its path
from models.cbam_denseunet import cbam_denseunet
# === Dataset Class ===
class cvccolondbsplitDataset(Dataset):
    def __init__(self, enhanced_dir, high_dir, transform=None):
        self.enhanced_dir = enhanced_dir
        self.high_dir = high_dir
        self.transform = transform
        self.image_names = sorted(os.listdir(enhanced_dir))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        enhanced_path = os.path.join(self.enhanced_dir, self.image_names[idx])
        high_path = os.path.join(self.high_dir, self.image_names[idx])
        enhanced_img = Image.open(enhanced_path).convert("RGB")
        high_img = Image.open(high_path).convert("RGB")
        if self.transform:
            enhanced_img = self.transform(enhanced_img)
            high_img = self.transform(high_img)
        return enhanced_img, high_img

# === Loss Functions ===
mse_loss_fn = nn.MSELoss()
mae_loss_fn = nn.L1Loss() # MAE is L1Loss in PyTorch

def total_loss_fn(pred, target, w_mae, w_mse, w_lpips, lpips_model):
    """Calculates a total loss from MAE, MSE, and LPIPS components."""
    mae = mae_loss_fn(pred, target)
    mse = mse_loss_fn(pred, target)
    # LPIPS expects input in range [-1, 1], so we scale it
    lp = lpips_model(2 * pred - 1, 2 * target - 1).mean()
    total = w_mae * mae + w_mse * mse + w_lpips * lp
    return total, mae, mse, lp

# === Early Stopping ===
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# === Paths ===
train_enhanced_dir = "/content/outputs/train_enhanced"
train_high_dir = "/content/cvccolondbsplit/train/high"
val_enhanced_dir = "/content/outputs/val_enhanced"
val_high_dir = "/content/cvccolondbsplit/val/high"

# === Hyperparameters ===
learning_rate = 1e-4
weight_decay = 1e-5
num_epochs = 100
batch_size = 8

# === Loss Weights ===
w_mae = 1.0
w_mse = 0.5
w_lpips = 0.1

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Dataloaders ===
train_dataset = cvccolondbsplitDataset(train_enhanced_dir, train_high_dir, transform)
val_dataset = cvccolondbsplitDataset(val_enhanced_dir, val_high_dir, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# === Model ===
model = cbam_denseunet().to(device)

# === Optimizer ===
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# === Early Stopping ===
early_stopping = EarlyStopping(patience=10)

# === Loss History ===
train_losses = []
val_losses = []

# === Training Loop ===
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    running_mse = 0.0
    running_lpips = 0.0

    for input_img, target_img in train_loader:
        input_img, target_img = input_img.to(device), target_img.to(device)
        optimizer.zero_grad()
        output = model(input_img)
        total_loss, mae_val, mse_val, lpips_val = total_loss_fn(
            output, target_img, w_mae, w_mse, w_lpips, lpips_loss_fn)
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()
        running_mae += mae_val.item()
        running_mse += mse_val.item()
        running_lpips += lpips_val.item()

    avg_train_loss = running_loss / len(train_loader)
    avg_train_mae = running_mae / len(train_loader)
    avg_train_mse = running_mse / len(train_loader)
    avg_train_lpips = running_lpips / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    val_mse = 0.0
    val_lpips = 0.0

    with torch.no_grad():
        for input_img, target_img in val_loader:
            input_img, target_img = input_img.to(device), target_img.to(device)
            output = model(input_img)
            total_loss, mae_val, mse_val, lpips_val = total_loss_fn(
                output, target_img, w_mae, w_mse, w_lpips, lpips_loss_fn)
            val_loss += total_loss.item()
            val_mae += mae_val.item()
            val_mse += mse_val.item()
            val_lpips += lpips_val.item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_mae = val_mae / len(val_loader)
    avg_val_mse = val_mse / len(val_loader)
    avg_val_lpips = val_lpips / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f}")
    print(f"Train MAE: {avg_train_mae:.4f}, Train MSE: {avg_train_mse:.4f}, Train LPIPS: {avg_train_lpips:.4f}")
    print(f"Val MAE: {avg_val_mae:.4f}, Val MSE: {avg_val_mse:.4f}, Val LPIPS: {avg_val_lpips:.4f}")

    early_stopping(avg_val_loss)
    if avg_val_loss < early_stopping.best_val_loss - early_stopping.min_delta:
        torch.save(model.state_dict(), 'best_cbam_denseunet.pth')
        print("✔️ Saved best model.")

    if early_stopping.early_stop:
        print(f"⏹️ Early stopping at epoch {epoch+1}")
        break
