import argparse
import json
import torch
import sys
import os

# ✅ Import modules in the same folder
import training  # training.py must be in the same folder as run.py
import test
# import validation  # Uncomment if you have validation.py

from data.dataset import PairedDataset
from torch.utils.data import DataLoader
from models.cbam_denseunet import cbam_denseunet
from utils.loss_utils import totalloss
from utils.hyperparameter import LOSS_WEIGHTS  # Ensure this exists

# ✅ Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="CBAM-DenseUNet Runner")
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'validate'], required=True)
    parser.add_argument('--config', type=str, default='config/training.json', help='Path to config file')
    return parser.parse_args()

# ✅ Main logic
def main_runner():
    args = parse_args()

    # Load training config
    with open(args.config, 'r') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == 'train':
        # Dataset and DataLoader
        train_dataset = PairedDataset(
            config["train"]["low_light_root"],
            config["train"]["normal_light_root"],
            transform=None
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=True,
            num_workers=2
        )

        # Model
        model = cbam_denseunet().to(device)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])

        # Loss Function
        criterion = totalloss(
            device,
            w_mse=LOSS_WEIGHTS["mse"],
            w_ssim=LOSS_WEIGHTS["ssim"],
            w_lpips=LOSS_WEIGHTS["lpips"],
            w_edge=LOSS_WEIGHTS["edge"]
        )

        # ✅ Call train function from training.py
        training.train(config, train_loader, optimizer, criterion, device, model)

    elif args.mode == 'test':
        test.test(config)

    elif args.mode == 'validate':
        import validation
        validation.validate(config)

# ✅ Entry point
if __name__ == '__main__':
    main_runner()
