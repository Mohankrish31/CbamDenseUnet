# CBAM-DenseUNet (WIP)
Work-in-progress: Low-light colonoscopy image enhancement using CBAM and DenseUNet.
---
## ✅ Current Goals
- ✅ Build model (`CBAM_DenseUNet`)
- ✅ Train on **CVC-ColonDB** dataset
- ✅ Integrate multi-loss:
  - Structural Similarity (SSIM)
  - Learned Perceptual Image Patch Similarity (LPIPS)
  - Sobel Edge Loss
---
### 📊 Evaluation Metrics (To Do)

| Metric        | Status |
|---------------|--------|
| Total Loss    | ✅ Done |
| C-PSNR        | ❌ Pending |
| SSIM          | ❌ Pending |
| EBCM          | ❌ Pending |
| LPIPS         | ❌ Pending |
---
## 🔄 Next Steps
- [ ] Finalize validation loop and compute missing metrics
- [ ] Save final `.pt` model and validate on real data
- [ ] Plot metrics vs. epoch:
  - [ ] Total Loss
  - [ ] SSIM
  - [ ] LPIPS
  - [ ] EBCM
  - [x] C-PSNR
- [ ] Write full documentation
- [ ] Finalize `README.md`
---
📌 _Note: This is a work-in-progress. Full documentation and performance results will be added once initial experiments are completed._

