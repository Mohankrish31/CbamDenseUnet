from torchvision import transforms
def get_train_transforms(img_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2, hue=0.05),
        transforms.ToTensor()
    ])
def get_test_transforms(img_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
