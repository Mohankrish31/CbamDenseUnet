import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(train_losses, 'o-', label='Train Loss')
plt.plot(val_losses, 's-', label='Validation Loss')
plt.xlabel("Epoch Number", fontsize=14)
plt.ylabel("Total Loss", fontsize=14)
plt.title("Training and Validation Loss Curve for CBAM-DenseUNet", fontsize=16)
plt.legend(loc="upper right", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("training_validation_loss_curve.png", dpi=300)
plt.show()

