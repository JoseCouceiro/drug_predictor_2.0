import json
import matplotlib.pyplot as plt
import numpy as np

# Load training history
with open('data/08_reporting/multitask_training_history.json', 'r') as f:
    history = json.load(f)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Multitask Model Training History', fontsize=16, fontweight='bold')

epochs = range(1, len(history['loss']) + 1)

# Plot 1: Total Loss
ax1 = axes[0, 0]
ax1.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
ax1.set_title('Total Loss', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Drug Output (Binary Classification)
ax2 = axes[0, 1]
ax2.plot(epochs, history['drug_output_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
ax2.plot(epochs, history['val_drug_output_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
ax2.plot(epochs, history['drug_output_loss'], 'b--', label='Train Loss', alpha=0.6)
ax2.plot(epochs, history['val_drug_output_loss'], 'r--', label='Val Loss', alpha=0.6)
ax2.set_title('Drug Output (Drug vs Non-Drug)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Metric Value')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: ATC Output (Multiclass Classification)
ax3 = axes[1, 0]
ax3.plot(epochs, history['atc_output_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
ax3.plot(epochs, history['val_atc_output_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
ax3.plot(epochs, history['atc_output_loss'], 'b--', label='Train Loss', alpha=0.6)
ax3.plot(epochs, history['val_atc_output_loss'], 'r--', label='Val Loss', alpha=0.6)
ax3.set_title('ATC Output (17-Class Classification)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Metric Value')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Accuracy Comparison
ax4 = axes[1, 1]
ax4.plot(epochs, history['drug_output_accuracy'], 'b-', label='Drug Train', linewidth=2)
ax4.plot(epochs, history['val_drug_output_accuracy'], 'b--', label='Drug Val', linewidth=2)
ax4.plot(epochs, history['atc_output_accuracy'], 'g-', label='ATC Train', linewidth=2)
ax4.plot(epochs, history['val_atc_output_accuracy'], 'g--', label='ATC Val', linewidth=2)
ax4.set_title('Accuracy Comparison: Drug vs ATC', fontsize=12, fontweight='bold')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Accuracy')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0.3, 0.8])

plt.tight_layout()
plt.savefig('data/08_reporting/training_history_plot.png', dpi=300, bbox_inches='tight')
print("Training history plot saved to: data/08_reporting/training_history_plot.png")

# Print final metrics
print("\n" + "="*60)
print("FINAL TRAINING METRICS (Epoch 50)")
print("="*60)
print(f"Total Loss          : {history['loss'][-1]:.4f} (train) | {history['val_loss'][-1]:.4f} (val)")
print(f"Drug Accuracy       : {history['drug_output_accuracy'][-1]:.4f} (train) | {history['val_drug_output_accuracy'][-1]:.4f} (val)")
print(f"ATC Accuracy        : {history['atc_output_accuracy'][-1]:.4f} (train) | {history['val_atc_output_accuracy'][-1]:.4f} (val)")
print(f"Drug Loss           : {history['drug_output_loss'][-1]:.4f} (train) | {history['val_drug_output_loss'][-1]:.4f} (val)")
print(f"ATC Loss            : {history['atc_output_loss'][-1]:.4f} (train) | {history['val_atc_output_loss'][-1]:.4f} (val)")
print("="*60)

# Show best validation performance
best_drug_acc_epoch = np.argmax(history['val_drug_output_accuracy']) + 1
best_atc_acc_epoch = np.argmax(history['val_atc_output_accuracy']) + 1

print(f"\nBest Drug Accuracy  : {max(history['val_drug_output_accuracy']):.4f} at epoch {best_drug_acc_epoch}")
print(f"Best ATC Accuracy   : {max(history['val_atc_output_accuracy']):.4f} at epoch {best_atc_acc_epoch}")
print("="*60)

plt.show()
