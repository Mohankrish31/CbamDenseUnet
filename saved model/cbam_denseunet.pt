from models.cbam_denseunet import cbam_denseunet
import torch
import os
# Create model
model = cbam_denseunet()
dummy_weights_path = "cbam_denseunet.pt"
# Save random initialized weights
torch.save(model.state_dict(), dummy_weights_path)
print(f"Saved dummy pretrained model to {dummy_weights_path}")
