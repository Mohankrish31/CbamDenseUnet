import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE to a PIL Image or NumPy array (expects RGB)."""
    if isinstance(img, Image.Image):
        img = np.array(img)
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    img_lab_clahe = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(img_lab_clahe, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img_clahe)
def get_transforms(size=(224, 224), use_clahe=False):
    """Return transformation pipeline with optional CLAHE."""
    def preprocess(img):
        if use_clahe:
            img = apply_clahe(img)
        return img
    return transforms.Compose([
        transforms.Lambda(preprocess),
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
def get_train_augmentations(size=(224, 224), use_clahe=False):
    """Transformations for training with augmentation."""
    def preprocess(img):
        if use_clahe:
            img = apply_clahe(img)
        return img
    return transforms.Compose([
        transforms.Lambda(preprocess),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])
# Added function for backward compatibility
def some_transformation_function(size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

