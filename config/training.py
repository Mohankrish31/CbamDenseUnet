# ✅ train.py
import torch
from torch.utils.data import DataLoader
from models.cbam_denseunet import CBAM_DenseUNet
from dataset import cvccolondb
from loss_utils import CombinedLoss  # Your custom loss
import os, json
# Load config
with open("config.json") as f:
    config = json.load(f)
# Init model
model = CBAM_DenseUNet(**config["model"]["which_model"]["args"]).cuda()
# Dataset and DataLoader
train_data = cvccolondb(**config["train"]["dataset"]["args"])
train_loader = DataLoader(train_data, **config["train"]["dataloader"]["args"])
# Optimizer, Loss
optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
loss_fn = CombinedLoss()
# Training Loop
for epoch in range(config["train"]["n_epoch"]):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        low = batch["low"].cuda()
        high = batch["high"].cuda()
        optimizer.zero_grad()
        output = model(low)
        loss = loss_fn(output, high)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{config['train']['n_epoch']} Loss: {running_loss/len(train_loader):.4f}")

# Save Model
os.makedirs(config["train"]["model_path"], exist_ok=True)
torch.save(model.state_dict(), os.path.join(config["train"]["model_path"], config["train"]["model_name"]))

