import sys
import os
import torch
#Add project path (for Colab or custom path)
sys.path.append('/content/CbamDenseUNet') 
#Import local modules
import main as train
import test
#import validation  # Uncomment if needed
from data.dataset import PairedDataset 
from torch.utils.data import DataLoader
from models.cbam_denseunet import CBAM_DenseUNet
from utils.loss_utils import TotalLoss
from utils.hyperparameter_tuning import LOSS_WEIGHTS
from utils.parser import get_config  #Load config from parser
def main():
    #Get config (with argparse and json loading)
    config = get_config()
    device = torch.device(config.get("train", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    #Determine operation mode
    mode = config.get("run_mode", "train")  # Default to train if not specified
    if mode == 'train':
        # >>>––––– BEGIN DATA SECTION ––––––>>>
        train_dataset = PairedDataset(
            low_light_root    = config["train"]["dataset"]["args"]["low_light_root"],
            normal_light_root = config["train"]["dataset"]["args"]["normal_light_root"],
            image_size        = config["train"]["dataset"]["args"].get("image_size", [224, 224])
        )
        print("Length of PairedDataset:", len(train_dataset))
        train_loader = DataLoader(
            train_dataset,
            batch_size  = config["train"]["dataloader"]["args"]["batch_size"],
            shuffle     = config["train"]["dataloader"]["args"]["shuffle"],
            num_workers = config["train"]["dataloader"]["args"]["num_workers"],
            pin_memory  = True
        )
        # <<<––––– END DATA SECTION ––––––<<<
        # Load Model
        model = CBAM_DenseUNet().to(device)
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
        # Loss Function
        criterion = TotalLoss(
            device,
            w_mse   = LOSS_WEIGHTS["mse"],
            w_ssim  = LOSS_WEIGHTS["ssim"],
            w_lpips = LOSS_WEIGHTS["lpips"],
            w_edge  = LOSS_WEIGHTS["edge"]
        )
        # Train Model
        train.train(config, train_loader, optimizer, criterion, device, model)
    elif mode == 'test':
        test.test(config)
    elif mode == 'validate':
        import validation
        valid.validate(config)
if __name__ == '__main__':
    main()
