import os
import sys
sys.path.append(os.getcwd())
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lpips
from pytorch_msssim import ssim
from torchvision import transforms
import sys
import os
#  Fix path for 'data.dataset'
sys.path.append(os.path.dirname(__file__))
from data.dataset import PairedDataset, UnpairedDataset
from models.cbam_denseunet import Cbam_DenseUnet
from utils.augmentation import get_train_transforms, get_test_transforms
from utils.reproducibility import set_seed
from utils.hyperparameter import params
import argparse
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for low_light, normal_light in train_loader:
        low_light = low_light.to(device)
        normal_light = normal_light.to(device)
        optimizer.zero_grad()
        output = model(low_light)
        loss = criterion(output, normal_light)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)
def test(model, test_loader, device, output_path):
    model.eval()
    os.makedirs(output_path, exist_ok=True)
    to_pil = transforms.ToPILImage()
    with torch.no_grad():
        for i, (low_light, fname) in enumerate(test_loader):
            low_light = low_light.to(device)
            output = model(low_light)
            output = output.cpu()
            for j in range(output.shape[0]):
                save_path = os.path.join(output_path, fname[j])
                out_img = to_pil(output[j].clamp(0, 1))
                out_img.save(save_path)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
    args = parser.parse_args()
    set_seed(42)
    config = load_config(args.config)
    device = torch.device(config[args.mode]['device'])
    model = CBAM_DenseUNet().to(device)
    if args.mode == 'train':
        dataset_args = config['train']['dataset']['args']
        train_dataset = PairedDataset(
            low_light_root=dataset_args['low_light_root'],
            normal_light_root=dataset_args['normal_light_root'],
            image_size=params['img_size']
        )
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        criterion = nn.MSELoss()
for epoch in range(params['epochs']):
    loss = train(model, train_loader, optimizer, criterion, device)
    print(f"Epoch {epoch+1}/{params['epochs']} - Loss: {loss:.4f}") 
    torch.save(
        model.state_dict(),
        os.path.join(config['train']['model_path'], config['train']['model_name'])
    )
    else:  # test mode
        dataset_args = config['test']['dataset']['args']
        test_dataset = UnpairedDataset(
            low_light_root=dataset_args['low_light_root'],
            image_size=params['img_size']
        )
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
        model.load_state_dict(torch.load(os.path.join(config['test']['model_path'], config['test']['model_name']), map_location=device))
        output_path = config['test']['output_images_path']
        test(model, test_loader, device, output_path)
if __name__ == '__main__':
    main()
