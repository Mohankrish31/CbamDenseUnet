import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Load your saved metrics CSV ===
df = pd.read_csv("metrics_results.csv")

# === Line plot: Per-image metrics ===
plt.figure(figsize=(12, 6))
plt.plot(df["filename"], df["PSNR"], marker='o', label="PSNR")
plt.plot(df["filename"], df["SSIM"], marker='s', label="SSIM")
plt.plot(df["filename"], df["LPIPS"], marker='^', label="LPIPS")
plt.xticks(rotation=90)
plt.xlabel("Image")
plt.ylabel("Metric Value")
plt.title("Per-image Metrics")
plt.legend()
plt.tight_layout()
plt.savefig("per_image_metrics.png", dpi=300)
plt.show()

# === Bar chart: Mean metrics ===
mean_values = [df["PSNR"].mean(), df["SSIM"].mean(), df["LPIPS"].mean()]
plt.figure(figsize=(6, 4))
bars = plt.bar(["PSNR", "SSIM", "LPIPS"], mean_values, color=['skyblue', 'lightgreen', 'salmon'])
plt.ylabel("Mean Value")
plt.title("Mean Metrics Across Dataset")
for bar, val in zip(bars, mean_values):
    plt.text(bar.get_x() + bar.get_width()/2, val, f"{val:.4f}", ha='center', va='bottom', fontsize=9)
plt.savefig("mean_metrics.png", dpi=300)
plt.show()
