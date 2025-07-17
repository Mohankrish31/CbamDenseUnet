import torch
from torch.utils.data import DataLoader
# Assuming 'models.cbam_denseunet' and 'dataset.cvccolondb' are correctly in your project structure
from models.cbam_denseunet import CBAM_DenseUNet
from dataset.cvccolondb import cvccolondb
# Assuming 'utils.loss_utils' contains the TotalLoss class
from utils.loss_utils import TotalLoss # Corrected: Assuming TotalLoss is directly importable
from utils.hyperparameter_tuning import get_optimizer, get_scheduler
from utils.reproducibility import set_seed
from utils.logger import Logger
import os
import json
# ✅ Set seed for reproducibility
set_seed(42)
# ✅ Load configuration
try:
    with open("config.json") as f:
        config = json.load(f)
except FileNotFoundError:
    print("Error: config.json not found. Please ensure it's in the same directory.")
    exit() # Exit if config file is missing
# Ensure 'cuda' is available, otherwise use 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# ✅ Initialize model
# Check if "which_model" and "args" exist in config["model"]
if "model" in config and "which_model" in config["model"] and "args" in config["model"]["which_model"]:
    model = CBAM_DenseUNet(**config["model"]["which_model"]["args"]).to(device) # Moved model to device
else:
    print("Error: Model configuration missing in config.json. Please check 'model.which_model.args'.")
    exit()
# ✅ Dataset and DataLoader
# Check if "dataset" and "args" exist in config["train"]
if "train" in config and "dataset" in config["train"] and "args" in config["train"]["dataset"]:
    train_data = cvccolondb(**config["train"]["dataset"]["args"])
else:
    print("Error: Train dataset configuration missing in config.json. Please check 'train.dataset.args'.")
    exit()
# Check if "dataloader" and "args" exist in config["train"]
if "train" in config and "dataloader" in config["train"] and "args" in config["train"]["dataloader"]:
    train_loader = DataLoader(train_data, **config["train"]["dataloader"]["args"])
else:
    print("Error: Train dataloader configuration missing in config.json. Please check 'train.dataloader.args'.")
    exit()
# ✅ Optimizer and Scheduler
# Check if "lr" exists in config["train"]
if "train" in config and "lr" in config["train"]:
    optimizer = get_optimizer(model, lr=config["train"]["lr"])
else:
    print("Error: Learning rate (lr) missing in config.json. Please check 'train.lr'.")
    exit()
scheduler = get_scheduler(optimizer)
# ✅ Loss Function
# Assuming TotalLoss does not require arguments or they are handled internally.
# If TotalLoss requires arguments, you would pass them here, e.g., loss_fn = TotalLoss(weights=...)
try:
    loss_fn = TotalLoss()
except NameError:
    print("Error: TotalLoss class not found. Make sure it's correctly imported from utils.loss_utils.")
    exit()
# ✅ Logger
logger = Logger() # Initialize Logger
# ✅ Training Loop
# Check if "n_epoch" exists in config["train"]
if "train" in config and "n_epoch" in config["train"]:
    num_epochs = config["train"]["n_epoch"]
else:
    print("Error: Number of epochs (n_epoch) missing in config.json. Please check 'train.n_epoch'.")
    exit()
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    print(f"Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        # Calculate loss
        loss = loss_fn(outputs, targets) # Assuming TotalLoss takes outputs and targets
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        # Log batch loss (optional)
        if batch_idx % 10 == 0: # Log every 10 batches
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    # Calculate epoch loss
    epoch_loss = running_loss / len(train_data)
    print(f"Epoch {epoch+1} finished, Average Loss: {epoch_loss:.4f}")
    # Step the scheduler (if using a learning rate scheduler)
    scheduler.step(epoch_loss) # Or scheduler.step() depending on your scheduler type
    # Log epoch metrics using the logger (assuming logger has a log_epoch method)
    logger.log_epoch(epoch, epoch_loss)
    # You might want to add validation/evaluation here
    # model.eval()
    # with torch.no_grad():
    #     validation_loss = 0.0
    #     for inputs_val, targets_val in val_loader:
    #         inputs_val = inputs_val.to(device)
    #         targets_val = targets_val.to(device)
    #         outputs_val = model(inputs_val)
    #         val_loss = loss_fn(outputs_val, targets_val)
    #         validation_loss += val_loss.item() * inputs_val.size(0)
    #     avg_val_loss = validation_loss / len(val_data)
    #     print(f"  Validation Loss: {avg_val_loss:.4f}")
    #     logger.log_validation(epoch, avg_val_loss)
    # Save model checkpoint (optional)
    if (epoch + 1) % 5 == 0: # Save every 5 epochs
        save_path = os.path.join(config.get("checkpoint_dir", "./checkpoints"), f"model_epoch_{epoch+1}.pth")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
print("Training finished!")
