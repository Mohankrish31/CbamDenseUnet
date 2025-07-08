
import sys
sys.path.append('/content/CBAMDenseUNet')

import argparse
import json
import torch
import training
import test
# import validation  # Uncomment if validation.py exists

from data.dataset import PairedDataset
from torch.utils.data import DataLoader
from models.cbam_denseunet import CBAM_DenseUNet
from utils.loss_utils import TotalLoss
from utils.hyperparameter import LOSS_WEIGHTS  # ✅ Should exist and be defined properly

def parse_args():
    parser = argparse.ArgumentParser(description="CBAM-DenseUNet Runner")
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'validate'], required=True)
    parser.add_argument('--config', type=str, default='config/training.json', help='Path to config file')
    return parser.parse_args()

def main():
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
        model = CBAM_DenseUNet().to(device)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])

        # Loss Function
        criterion = TotalLoss(
            device,
            w_mse=LOSS_WEIGHTS["mse"],
            w_ssim=LOSS_WEIGHTS["ssim"],
            w_lpips=LOSS_WEIGHTS["lpips"],
            w_edge=LOSS_WEIGHTS["edge"]
        )

        # Train
        training.train(config, train_loader, optimizer, criterion, device)

    elif args.mode == 'test':
        test.test(config)

    elif args.mode == 'validate':
        import validation  # Only if you have this module
        validation.validate(config)

if __name__ == '__main__':
    main()
