import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.cvccolondbsplit import cvccolondbsplit  # Make sure this matches your folder name
from models.CBAM_DenseUNet import CBAM_DenseUNet
from totalloss import TotaLoss  # Your custom loss
from utils import save_results, validate_model  # Optional utilities
import json
import os
from train_utils import train_one_epoch, evaluate
def main():
    config_path = "config/training.json"
    with open(config_path) as f:
        config = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = config["mode"]
    model = CBAM_DenseUNet(in_channels=3, out_channels=3, features=64).to(device)
    # Dataset & DataLoader
    if mode == "train":
        train_dataset = cvccolondbsplit(config["dataset"]["args"]["low"], config["dataset"]["args"]["high"])
        train_loader = DataLoader(train_dataset, batch_size=config["training_info"]["batch_size"], shuffle=True)
        val_dataset = cvccolondbsplit(config["dataset"]["args"]["val_low"], config["dataset"]["args"]["val_high"])
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        # Loss and Optimizer
        loss_fn = TotaLoss()  # Combined MSE, SSIM, LPIPS, Edge Loss
        optimizer = torch.optim.Adam(model.parameters(), lr=config["training_info"]["learning_rate"])
        # Training Loop
        for epoch in range(config["training_info"]["epoch"]):
            print(f"\nEpoch {epoch+1}/{config['training_info']['epoch']}")
            train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
            # Validation
            evaluate(model, val_loader, loss_fn, device)
    elif mode == "test":
        test_dataset = cvccolondbsplit(config["dataset"]["args"]["test_low"], config["dataset"]["args"]["test_high"])
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        model.load_state_dict(torch.load(config["testing"]["model_path"], map_location=device))
        model.eval()
        # Save enhanced output
        save_results(model, test_loader, device, config["testing"]["save_dir"])
if __name__ == "__main__":
    main()
