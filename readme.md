# CBAM-DenseUNet-Retinex (WIP)

Low-light colonoscopy image enhancement using **Convolutional Block Attention Module (CBAM)**, **DenseUNet backbone**, and **Retinex-based decomposition**.

> **Status:** Work in Progress – Training, validation, and testing scripts are functional. Results will be updated after experiments.

---

## 🚀 Features
- **CBAM Attention:** Enhances feature maps via channel and spatial attention.
- **DenseUNet Backbone:** Dense connections for improved feature reuse and gradient flow.
- **Retinex Decomposition:** Separates illumination and reflectance to boost low-light details.
- **Multi-Loss Optimization:** Combines MSE, SSIM, LPIPS, and Edge loss for improved perceptual quality.

---

## 📂 Dataset
This project is trained and evaluated on the **CVC-ColonDB** dataset.

- **Training & Validation Resolution:** Images resized to `224×224`
- **Testing Resolution:** Original dimensions `574×500`
- **Augmentations:** Random crop, flip, rotation
- **Split:** Train / Validation / Test

---

## 📊 Evaluation Metrics
Implemented:
- **PSNR (Peak Signal-to-Noise Ratio)**
- **SSIM (Structural Similarity Index)**
- **LPIPS (Learned Perceptual Image Patch Similarity)**

---

## 🛠 Installation
```bash
git clone https://github.com/<your-username>/cbam-denseunet-retinex.git
cd cbam-denseunet-retinex
pip install -r requirements.txt
