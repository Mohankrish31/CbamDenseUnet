# dataset.py

import os
import cv2
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def clahe_rgb(img):
    """Apply CLAHE to the V-channel of an RGB image."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

class cvccolondb(Dataset):
    def __init__(self, root, mode="train", size=(224, 224), use_clahe=True):
        """
        Args:
            root: Root path to dataset/
            mode: 'train', 'val', or 'test'
            size: Tuple image resize size (H, W)
            use_clahe: Apply CLAHE to low-light images
        """
        self.low_paths = sorted(glob.glob(os.path.join(root, mode, "low", "*")))
        self.high_paths = sorted(glob.glob(os.path.join(root, mode, "high", "*")))
        assert len(self.low_paths) == len(self.high_paths), "Mismatch in image pairs."

        self.size = size
        self.use_clahe = use_clahe
        self.tfs = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        low_img  = np.array(Image.open(self.low_paths[idx]).convert("RGB"))
        high_img = np.array(Image.open(self.high_paths[idx]).convert("RGB"))

        # Resize images
        low_img  = cv2.resize(low_img,  self.size)
        high_img = cv2.resize(high_img, self.size)

        if self.use_clahe:
            low_img = clahe_rgb(low_img)

        low_tensor  = self.tfs(low_img)
        high_tensor = self.tfs(high_img)

        return {
            "low": low_tensor,
            "high": high_tensor
        }
