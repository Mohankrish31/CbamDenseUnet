# CBAM-DenseUNet (WIP)

Work in progress on low-light colonoscopy image enhancement using CBAM and DenseUNet.

## ✅ Current Goals:
- [x] Build model ✅
- [x] Train on CVC-ColonDB ✅
- [x] Add SSIM, LPIPS, and Sobel loss ✅
- [ ] Evaluate with:
  - [x] Total Loss
  - [ ] C-PSNR ❌
  - [ ] SSIM ❌
  - [ ] EBCM ❌
  - [ ] LPIPS ❌
- [ ] Write manuscript ❌

---

## 🔄 Next Steps
- [ ] Finish validation and compute missing evaluation metrics
- [ ] Save `.pt` and validate on real data
- [ ] Prepare plots:
  - [ ] Total Loss vs Epoch
  - [ ] SSIM vs Epoch
  - [ ] LPIPS vs Epoch
  - [ ] EBCM vs Epoch
  - [ ] ✅ **C-PSNR vs Epoch**
- [ ] Write complete documentation and final `README.md`

> Note: This is a work-in-progress draft. Full documentation will be added after initial experiments.
