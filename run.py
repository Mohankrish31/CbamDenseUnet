import sys
import os
sys.path.append('/content/CbamDenseUnet') 
import argparse
import json
import torch
# ✅ Import local modules
import main as training
import test
# import validation  # Uncomment if needed
from data.dataset import PairedDataset
from torch.utils.data import DataLoader
from models.cbam_denseunet import cbam_denseunet
from utils.loss_utils import totalloss
from utils.hyperparameter_tuning import LOSS_WEIGHTS
def parse_args():
    parser = argparse.ArgumentParser(description="CBAM-DenseUNet Runner")
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'validate'], required=True)
    parser.add_argument('--config', type=str, default='config/training/training.json', help='Path to config file')
    return parser.parse_args()
def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == 'train':
        # Load Dataset
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

        # Load Model
        model = cbam_denseunet().to(device)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])

        # Custom Loss Function
        criterion = totalloss(
            device,
            w_mse=LOSS_WEIGHTS["mse"],
            w_ssim=LOSS_WEIGHTS["ssim"],
            w_lpips=LOSS_WEIGHTS["lpips"],
            w_edge=LOSS_WEIGHTS["edge"]
        )

        # Train Model
        training.train(config, train_loader, optimizer, criterion, device, model)

    elif args.mode == 'test':
        test.test(config)
    elif args.mode == 'validate':
        import validation
        validation.validate(config)
if __name__ == '__main__':
    main()
