import torch
from torch.utils.data import DataLoader
from models.cbam_denseunet import CBAM_DenseUNet
from dataset.cvccolondb import cvccolondb
from torchvision.utils import save_image
import os
import json
# ✅ Import metrics and visualization utilities
from utils.metrics import evaluate_metrics
from utils.visualization import save_comparison, plot_metrics
# --- Configuration Loading ---
try:
    with open("config.json") as f:
        config = json.load(f)
except FileNotFoundError:
    print("Error: config.json not found. Please ensure it's in the same directory.")
    exit() # Exit if config file is missing
# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# --- Model Initialization and Loading ---
# Check for model configuration
if "model" not in config or "which_model" not in config["model"] or "args" not in config["model"]["which_model"]:
    print("Error: Model configuration missing in config.json. Please check 'model.which_model.args'.")
    exit()
model = CBAM_DenseUNet(**config["model"]["which_model"]["args"]).to(device) # Move model to device
# Check for test configuration and model path
if "test" not in config or "model_path" not in config["test"] or "model_name" not in config["test"]:
    print("Error: Test model path or name missing in config.json. Please check 'test.model_path' and 'test.model_name'.")
    exit()
model_full_path = os.path.join(config["test"]["model_path"], config["test"]["model_name"])
try:
    model.load_state_dict(torch.load(model_full_path, map_location=device)) # Map to appropriate device
    print(f"Model loaded successfully from: {model_full_path}")
except FileNotFoundError:
    print(f"Error: Model checkpoint not found at {model_full_path}. Please check the path.")
    exit()
except Exception as e:
    print(f"Error loading model state dict: {e}")
    exit()
model.eval() # Set model to evaluation mode
# --- Test Dataset and DataLoader ---
# Check for dataset configuration
if "test" not in config or "dataset" not in config["test"] or "args" not in config["test"]["dataset"]:
    print("Error: Test dataset configuration missing in config.json. Please check 'test.dataset.args'.")
    exit()
test_data = cvccolondb(**config["test"]["dataset"]["args"])
# Check for dataloader configuration
if "test" not in config or "dataloader" not in config["test"] or "args" not in config["test"]["dataloader"]:
    print("Error: Test dataloader configuration missing in config.json. Please check 'test.dataloader.args'.")
    exit()
test_loader = DataLoader(test_data, **config["test"]["dataloader"]["args"])
# --- Output Directory Creation ---
if "test" not in config or "output_images_path" not in config["test"]:
    print("Error: Output images path missing in config.json. Please check 'test.output_images_path'.")
    exit()
output_dir = config["test"]["output_images_path"]
os.makedirs(output_dir, exist_ok=True)
print(f"Output images will be saved to: {output_dir}")
# --- Metrics Collection ---
all_metrics = []
# --- Run Inference and Save Results ---
with torch.no_grad(): # Disable gradient calculations for inference
    for i, batch in enumerate(test_loader):
        # Ensure tensors are on the correct device
        low = batch["low"].to(device)
        high = batch["high"].to(device) # High-resolution ground truth
        output = model(low)
        # ✅ Save output image (predicted high-resolution)
        save_path = os.path.join(output_dir, f"output_{i}.png")
        # Ensure image is in a savable format (e.g., float between 0-1, or uint8)
        # Assuming output is already in [0, 1] range. If not, normalize or clamp.
        save_image(output.cpu(), save_path) # Move to CPU before saving
        # ✅ Evaluate metrics (output vs. ground truth high)
        # Detach tensors from the GPU and convert to numpy if evaluate_metrics expects numpy
        # Or ensure evaluate_metrics handles torch tensors on device.
        # Assuming evaluate_metrics can take torch tensors, but usually they expect CPU numpy.
        metrics = evaluate_metrics(output.cpu(), high.cpu())
        all_metrics.append(metrics)
        # ✅ Save comparison plot (low, predicted high, ground truth high)
        compare_plot_path = os.path.join(output_dir, f"compare_{i}.png")
        # Ensure tensors are on CPU for visualization if save_comparison expects it
        save_comparison(low.cpu(), output.cpu(), high.cpu(), compare_plot_path)
        print(f"Processed batch {i+1}/{len(test_loader)}. Metrics: {metrics}")
# ✅ Plot aggregated metrics (optional summary)
if all_metrics: # Only plot if there are metrics to plot
    metrics_plot_save_path = os.path.join(output_dir, "metrics_plot.png")
    plot_metrics(all_metrics, save_path=metrics_plot_save_path)
    print(f"Aggregated metrics plot saved to: {metrics_plot_save_path}")
else:
    print("No metrics collected to plot.")
print(f"✅ Test images and comparisons saved to: {output_dir}")
print("Testing complete!")
