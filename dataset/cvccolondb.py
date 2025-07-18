import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
class cvccolondbsplitDataset(Dataset):
    def __init__(self, low_root, high_root, transform=None):
        self.low_root = low_root
        self.high_root = high_root
        self.transform = transform
        self.filenames = sorted(os.listdir(low_root))
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        low_img_path = os.path.join(self.low_root, self.filenames[idx])
        high_img_path = os.path.join(self.high_root, self.filenames[idx])
        low = Image.open(low_img_path).convert('RGB')
        high = Image.open(high_img_path).convert('RGB')
        if self.transform:
            low = self.transform(low)
            high = self.transform(high)
        return low, high
