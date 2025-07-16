import torch
from torch.utils.data import DataLoader
from models.cbam_denseunet import CBAM_DenseUNet
from dataset import cvc-colondb
from utils.loss_utils import TotalLoss # ✅ Custom total loss function
from utils.hyperparameter_tuning import get_optimizer, get_scheduler, update_loss_weights
from utils.reproducibility import set_seed
from utils.logger import Logger  # ✅ Logger added
import os
import json
# ✅ Set seed for reproducibility
set_seed(42)
# ✅ Load configuration
with open("config.json") as f:
    config = json.load(f)
# ✅ Initialize model
model = CBAM_DenseUNet(**config["model"]["which_model"]["args"]).cuda()
# ✅ Dataset and DataLoader
train_data = cvc-colondb(**config["train"]["dataset"]["args"])
train_loader = DataLoader(train_data, **config["train"]["dataloader"]["args"])
# ✅ Optimizer and Scheduler
optimizer = get_optimizer(model, lr=config["train"]["lr"])
scheduler = get_scheduler(optimizer)
# ✅ Loss Function
loss_fn = totalloss()
# ✅ Logger
logger = Logger()
# ✅ Training Loop
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
    # Optional scheduler step (e.g., ReduceLROnPlateau)
    if scheduler is not None:
        scheduler.step(running_total / len(train_loader))
    # Logging average metrics per epoch
    num_batches = len(train_loader)
    avg_total = running_total / num_batches
    avg_mse = running_mse / num_batches
    avg_ssim = running_ssim / num_batches
    avg_lpips = running_lpips / num_batches
    avg_edge = running_edge / num_batches
    print(f"\n📘 Epoch {epoch+1}/{config['train']['n_epoch']}")
    print(f"   Total Loss : {avg_total:.4f}")
    print(f"   MSE        : {avg_mse:.4f}")
    print(f"   SSIM Loss  : {avg_ssim:.4f}")
    print(f"   LPIPS Loss : {avg_lpips:.4f}")
    print(f"   Edge Loss  : {avg_edge:.4f}")
    # ✅ Log values
    logger.log({
        "epoch": epoch + 1,
        "loss": avg_total,
        "mse": avg_mse,
        "ssim": avg_ssim,
        "lpips": avg_lpips,
        "edge": avg_edge
    })
# ✅ Save the trained model
os.makedirs(config["train"]["model_path"], exist_ok=True)
save_path = os.path.join(config["train"]["model_path"], config["train"]["model_name"])
torch.save(model.state_dict(), save_path)
print(f"\n✅ Model saved to {save_path}")

