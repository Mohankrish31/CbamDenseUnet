%%writefile /content/CbamDenseUnet/CbamDenseUnet/main.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import PairedDataset
from models.cbam_denseunet import CBAM_DenseUNet
import yaml

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        low = batch['low'].to(device)
        normal = batch['normal'].to(device)

        optimizer.zero_grad()
        output = model(low)
        loss = criterion(output, normal)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

if __name__ == "__main__":
    config_path = "/content/CbamDenseUnet/CbamDenseUnet/config/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    params = config['train']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CBAM_DenseUNet().to(device)

    train_dataset = PairedDataset(
        low_light_dir=config['dataset']['train_low'],
        normal_light_dir=config['dataset']['train_normal']
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
