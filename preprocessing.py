import os
from PIL import Image, ImageEnhance
from tqdm import tqdm
# Paths
input_dir = '/content/cvccolondb'  # Raw cvc-colondb images
output_low = '/content/cvccolondb/train/low'
output_high = '/content/cvccolondb/train/high'
# Create directories
os.makedirs(output_low, exist_ok=True)
os.makedirs(output_high, exist_ok=True)
def simulate_low_light(img):
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(0.3)  # Reduce brightness
# Generate dataset
for fname in tqdm(os.listdir(input_dir)):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_dir, fname)
        img = Image.open(img_path).convert('RGB')
        low_img = simulate_low_light(img)
        img.save(os.path.join(output_high, fname))     # Original → ground truth
        low_img.save(os.path.join(output_low, fname))  # Darkened → input

