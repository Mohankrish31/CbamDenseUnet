import os
import json
import torch
from torch.utils.data import DataLoader
from dataset import cvccolondbsplit  # Replace with your actual dataset class file name
from models.CBAM_DenseUNet import CBAM_DenseUNet  # Make sure this path is correct
from loss_utils import MSE_SSIM_Loss  # Replace with your loss combination
from utils import save_results, validate_model  # Implement as needed (saving output, metrics etc.)
from train_utils import train_one_epoch  # Utility for training one epoch
def load_model(config):
    args = config["model"]["which_model"]["args"]
    model = CBAM_DenseUNet(**args)
    return model
def load_dataset(config):
    dataset_args = config["dataset"]["args"]
    return cvccolondbsplit (**dataset_args)
def load_dataloader(config, dataset):
    dataloader_args = config["dataloader"]["args"]
    return DataLoader(dataset, **dataloader_args)
def main(config_path):
    # Load JSON config
    with open(config_path, "r") as f:
        config = json.load(f)
    mode = config["mode"]
    device = torch.device(config["device"]["name"] if torch.cuda.is_available() else "cpu")
    model = load_model(config).to(device)
    # Load dataset and dataloader
    dataset = load_dataset(config)
    dataloader = load_dataloader(config, dataset)
    # Load model weights if not training
    if mode != "train":
        model_path = config.get("model_path", os.path.join(config["model_path"], config["model_name"]))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    if mode == "train":
        epochs = config["training_info"]["epoch"]
        lr = config["training_info"]["learning_rate"]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = MSE_SSIM_Loss()
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, dataloader, optimizer, loss_fn, device)
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}")
            # Optionally save checkpoints
            torch.save(model.state_dict(), config["model_path"])
    elif mode == "test":
        output_dir = config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        save_results(model, dataloader, output_dir, device)
    elif mode == "valid":
        validate_model(model, dataloader, device)
    else:
        print("Invalid mode! Use one of: train, valid, test.")
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python run.py path_to_config.json")
        exit(1)
    config_file = sys.argv[1]
    main(config_file)
