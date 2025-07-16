import torch
from torch.utils.data import DataLoader
from models.cbam_denseunet import CBAM_DenseUNet
from dataset import cvccolondb
from utils.loss_utils import TotalLoss # ✅ Your custom total loss function
import os
import json
# Load configuration
with open("config.json") as f:
    config = json.load(f)
# Initialize model
model = CBAM_DenseUNet(**config["model"]["which_model"]["args"]).cuda()
# Dataset and DataLoader
train_data = cvccolondb(**config["train"]["dataset"]["args"])
train_loader = DataLoader(train_data, **config["train"]["dataloader"]["args"])
# Optimizer and Loss
optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
loss_fn = totalloss()  # ✅ Replacing CombinedLoss
# Training Loop
for epoch in range(config["train"]["n_epoch"]):
    model.train()
    running_total, running_mse, running_ssim, running_lpips, running_edge = 0, 0, 0, 0, 0
    for batch in train_loader:
        low = batch["low"].cuda()
        high = batch["high"].cuda()
        optimizer.zero_grad()
        output = model(low)
        total_loss, mse, ssim_loss, lpips_loss, edge_loss = loss_fn(output, high)
        total_loss.backward()
        optimizer.step()
        running_total += total_loss.item()
        running_mse += mse.item()
        running_ssim += ssim_loss.item()
        running_lpips += lpips_loss.item()
        running_edge += edge_loss.item()
    # Logging average per epoch
    num_batches = len(train_loader)
    print(f"\n📘 Epoch {epoch+1}/{config['train']['n_epoch']}")
    print(f"   Total Loss : {running_total / num_batches:.4f}")
    print(f"   MSE        : {running_mse / num_batches:.4f}")
    print(f"   SSIM Loss  : {running_ssim / num_batches:.4f}")
    print(f"   LPIPS Loss : {running_lpips / num_batches:.4f}")
    print(f"   Edge Loss  : {running_edge / num_batches:.4f}")
# Save the trained model
os.makedirs(config["train"]["model_path"], exist_ok=True)
torch.save(model.state_dict(), os.path.join(config["train"]["model_path"], config["train"]["model_name"]))
print(f"\n✅ Model saved to {config['train']['model_path']}{config['train']['model_name']}")
