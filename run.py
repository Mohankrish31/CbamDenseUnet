import argparse
import json
import torch
import sys
import os
# Append base directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import main   # ✅ Use main instead of training
import test
# import validation  # Uncomment if validation.py exists

from data.dataset import PairedDataset
from torch.utils.data import DataLoader
from models.cbam_denseunet import cbam_denseunet
from utils.loss_utils import totalloss
from utils.hyperparameter import LOSS_WEIGHTS  # ✅ Make sure this exists and is defined

def parse_args():
    parser = argparse.ArgumentParser(description="CBAM-DenseUNet Runner")
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'validate'], required=True)
    parser.add_argument('--config', type=str, default='config/training.json', help='Path to config file')
    return parser.parse_args()

def main_runner():   # Renamed to avoid conflict with `main` module
    args = parse_args()

    # Load config JSON
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

        # ✅ Call train from main module
        main.train(config, train_loader, optimizer, criterion, device, model)

    elif args.mode == 'test':
        test.test(config)

    elif args.mode == 'validate':
        import validation  # Only if you have validation.py
        validation.validate(config)

if __name__ == '__main__':
    main_runner()

