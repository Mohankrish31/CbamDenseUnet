# /content/CBAMDenseUNet/utils/hyperparameter.py

# ✅ Used in run.py for weighted loss calculation
LOSS_WEIGHTS = {
    "mse": 1.0,
    "ssim": 1.0,
    "lpips": 0.5,
    "edge": 0.2
}

# ✅ Used in training.py or other configs that expect global parameters
params = {
    "learning_rate": 1e-4,
    "batch_size": 16,
    "num_epochs": 100,
    "model_name": "CBAM_DenseUNet"
}
