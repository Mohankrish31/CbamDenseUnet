# /content/CBAMDenseUNet/data/dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PairedDataset(Dataset):
    def __init__(self, low_light_root, normal_light_root, image_size=[200, 200]):
        super().__init__()
        self.low_light_dataset = sorted([
            os.path.join(low_light_root, image)
            for image in os.listdir(low_light_root)
            if image.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.normal_light_dataset = sorted([
            os.path.join(normal_light_root, image)
            for image in os.listdir(normal_light_root)
            if image.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        assert len(self.low_light_dataset) == len(self.normal_light_dataset), \
            f"Mismatch: {len(self.low_light_dataset)} low vs {len(self.normal_light_dataset)} normal"

        self.transforms = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.low_light_dataset)

    def __getitem__(self, idx):
        low_img = Image.open(self.low_light_dataset[idx]).convert('RGB')
        normal_img = Image.open(self.normal_light_dataset[idx]).convert('RGB')
        return self.transforms(low_img), self.transforms(normal_img)


class UnpairedDataset(Dataset):
    def __init__(self, low_light_root, image_size=[200, 200]):
        super().__init__()
        self.low_light_dataset = sorted([
            os.path.join(low_light_root, image)
            for image in os.listdir(low_light_root)
            if image.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.transforms = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.low_light_dataset)

    def __getitem__(self, idx):
        image = Image.open(self.low_light_dataset[idx]).convert('RGB')
        return self.transforms(image), os.path.basename(self.low_light_dataset[idx])
