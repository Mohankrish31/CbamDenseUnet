# ✅ test.py
import torch
from torch.utils.data import DataLoader
from models.CBAM_DenseUNet import CBAM_DenseUNet
from dataset import cvccolondb
from torchvision.utils import save_image
import os, json
# Load config
with open("config.json") as f:
    config = json.load(f)
# Init model
model =  CBAM_DenseUNet(**config["model"]["which_model"]["args"]).cuda()
model.load_state_dict(torch.load(os.path.join(config["test"]["model_path"], config["test"]["model_name"])))
model.eval()
# Test set
test_data = cvccolondb(**config["test"]["dataset"]["args"])
test_loader = DataLoader(test_data, **config["test"]["dataloader"]["args"])
os.makedirs(config["test"]["output_images_path"], exist_ok=True)
# Run inference and save results
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        low = batch["low"].cuda()
        output = model(low)
        save_image(output, os.path.join(config["test"]["output_images_path"], f"output_{i}.png"))
print("Test images saved to:", config["test"]["output_images_path"])
