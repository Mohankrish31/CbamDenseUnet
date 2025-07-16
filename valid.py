import torch
from torch.utils.data import DataLoader
from models.cbam_denseunet import CBAM_DenseUNet
from dataset.cvccolondb import cvccolondb
from utils.loss_utils import TotalLoss  # ✅ Import your total loss function
import os
import json
# Load config
with open("config.json") as f:
    config = json.load(f)
# Init model
model = CBAM_DenseUNet(**config["model"]["which_model"]["args"]).cuda()
model.load_state_dict(torch.load(os.path.join(config["train"]["model_path"], config["train"]["model_name"])))
model.eval()
# Init validation set and loader
val_data = cvccolondb(**config["val"]["dataset"]["args"])
val_loader = DataLoader(val_data, **config["val"]["dataloader"]["args"])
# Init loss function
loss_fn = TotalLoss ()
# Track total and average metrics
total_loss, total_mse, total_ssim, total_lpips, total_edge = 0, 0, 0, 0, 0
# Run validation
with torch.no_grad():
    for batch in val_loader:
        low = batch["low"].cuda()
        high = batch["high"].cuda()
        output = model(low)
        loss, mse, ssim_val, lpips_val, edge = loss_fn(output, high)
        total_loss += loss.item()
        total_mse += mse.item()
        total_ssim += ssim_val.item()
        total_lpips += lpips_val.item()
        total_edge += edge.item()
# Compute averages
num_batches = len(val_loader)
print(f"\n Validation Results (avg over {num_batches} batches):")
print(f"  Total Loss : {total_loss / num_batches:.4f}")
print(f"  MSE        : {total_mse / num_batches:.4f}")
print(f"  SSIM Loss  : {total_ssim / num_batches:.4f}")
print(f"  LPIPS Loss : {total_lpips / num_batches:.4f}")
print(f"  Edge Loss  : {total_edge / num_batches:.4f}")

