# ✅ test.py
import torch
from torch.utils.data import DataLoader
from models.cbam_denseunet import CBAM_DenseUNet
from dataset import cvc-colondb
from torchvision.utils import save_image
import os
import json
# ✅ Import metrics and visualization utilities
from utils.metrics import evaluate_metrics 
from utils.visualization import save_comparison, plot_metrics
# Load config
with open("config.json") as f:
    config = json.load(f)
# Init model
model = CBAM_DenseUNet(**config["model"]["which_model"]["args"]).cuda()
model.load_state_dict(torch.load(os.path.join(config["test"]["model_path"], config["test"]["model_name"])))
model.eval()
# Test set
test_data = cvc-colondb(**config["test"]["dataset"]["args"])
test_loader = DataLoader(test_data, **config["test"]["dataloader"]["args"])
# Create output directory
os.makedirs(config["test"]["output_images_path"], exist_ok=True)
# Metrics collection
all_metrics = []
# Run inference and save results
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        low = batch["low"].cuda()
        high = batch["high"].cuda()
        output = model(low)
        # ✅ Save output image
        save_path = os.path.join(config["test"]["output_images_path"], f"output_{i}.png")
        save_image(output, save_path)
        # ✅ Evaluate metrics
        metrics = evaluate_metrics(output, high)
        all_metrics.append(metrics)
        # ✅ Save comparison plot
        compare_plot_path = os.path.join(config["test"]["output_images_path"], f"compare_{i}.png")
        save_comparison(low, output, high, compare_plot_path)
# ✅ Plot aggregated metrics (optional summary)
plot_metrics(all_metrics, save_path=os.path.join(config["test"]["output_images_path"], "metrics_plot.png"))
print("✅ Test images saved to:", config["test"]["output_images_path"])

