import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.cbam_denseunet import cbam_denseunet
from dataset import cvccolondb
from loss_utils import CombinedLoss
# Load config
with open("config.json") as f:
    config = json.load(f)
def train():
    print(" Training started...")
    model = cbam_denseunet(**config["model"]["which_model"]["args"]).cuda()
    train_data = cvccolondb(**config["train"]["dataset"]["args"])
    train_loader = DataLoader(train_data, **config["train"]["dataloader"]["args"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    loss_fn = CombinedLoss()
    for epoch in range(config["train"]["n_epoch"]):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            low, high = batch["low"].cuda(), batch["high"].cuda()
            optimizer.zero_grad()
            output = model(low)
            loss = loss_fn(output, high)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{config['train']['n_epoch']} Loss: {epoch_loss / len(train_loader):.4f}")
    os.makedirs(config["train"]["model_path"], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(config["train"]["model_path"], config["train"]["model_name"]))
    print("✅ Training complete and model saved.")
def validate():
    print("🔵 Validation started...")
    model = cbam_denseunet(**config["model"]["which_model"]["args"]).cuda()
    model.load_state_dict(torch.load(os.path.join(config["train"]["model_path"], config["train"]["model_name"])))
    model.eval()
    val_data = cvccolondb(**config["val"]["dataset"]["args"])
    val_loader = DataLoader(val_data, **config["val"]["dataloader"]["args"])
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            low = batch["low"].cuda()
            high = batch["high"].cuda()
            output = model(low)
            val_loss += torch.nn.functional.mse_loss(output, high).item()
    print(f"✅ Validation MSE: {val_loss / len(val_loader):.4f}")
def test():
    print("🟣 Testing started...")
    model = cbam_denseunet(**config["model"]["which_model"]["args"]).cuda()
    model.load_state_dict(torch.load(os.path.join(config["test"]["model_path"], config["test"]["model_name"])))
    model.eval()
    test_data = cvccolondb(**config["test"]["dataset"]["args"])
    test_loader = DataLoader(test_data, **config["test"]["dataloader"]["args"])
    os.makedirs(config["test"]["output_images_path"], exist_ok=True)
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            low = batch["low"].cuda()
            output = model(low)
            save_image(output, os.path.join(config["test"]["output_images_path"], f"output_{i}.png"))
    print("✅ Test images saved to:", config["test"]["output_images_path"])
# Argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CBAM-DenseUNet Pipeline")
    parser.add_argument("--mode", type=str, choices=["train", "val", "test"], required=True,
                        help="Run mode: train / val / test")
    args = parser.parse_args()
    if args.mode == "train":
        train()
    elif args.mode == "val":
        validate()
    elif args.mode == "test":
        test()
