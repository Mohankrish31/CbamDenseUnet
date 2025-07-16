import os
import random
import numpy as np
import torch
def set_seed(seed=42):
    """
    Sets seeds for reproducibility across random, numpy, and torch (CPU & CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For single-GPU
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    # Ensures deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Optional: control hash seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Reproducibility seed set to: {seed}")
