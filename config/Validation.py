# ✅ val.py
import torch
from torch.utils.data import DataLoader
from models.cbam_denseunet import cbam_denseunet
from dataset import cvccolondb
import json
# Load config
with open("config.json") as f:
    config = json.load(f)
# Init model
model = cbam_denseunet(**config["model"]["which_model"]["args"]).cuda()
model.load_state_dict(torch.load(os.path.join(config["train"]["model_path"], config["train"]["model_name"])))
model.eval()
# Validation set
val_data = cvccolondb(**config["val"]["dataset"]["args"])
val_loader = DataLoader(val_data, **config["val"]["dataloader"]["args"])
# Metrics placeholder
val_loss = 0
with torch.no_grad():
    for batch in val_loader:
        low = batch["low"].cuda()
        high = batch["high"].cuda()
        output = model(low)
        val_loss += torch.nn.functional.mse_loss(output, high).item()
print(f"Validation MSE: {val_loss / len(val_loader):.4f}")

