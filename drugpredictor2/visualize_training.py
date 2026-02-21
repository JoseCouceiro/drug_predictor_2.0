import json
import matplotlib.pyplot as plt
import numpy as np

# Load training histories for both models
with open('data/08_reporting/drug_classifier_history.json', 'r') as f:
    drug_history = json.load(f)

with open('data/08_reporting/atc_classifier_history.json', 'r') as f:
    atc_history = json.load(f)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Model Training History', fontsize=16, fontweight='bold')

# ── Plot 1: Drug Classifier Loss ──
drug_epochs = range(1, len(drug_history['loss']) + 1)
ax1 = axes[0, 0]
ax1.plot(drug_epochs, drug_history['loss'], 'b-', label='Train Loss', linewidth=2)
ax1.plot(drug_epochs, drug_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
ax1.set_title('Drug Classifier — Loss', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ── Plot 2: Drug Classifier Accuracy ──
ax2 = axes[0, 1]
ax2.plot(drug_epochs, drug_history['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
ax2.plot(drug_epochs, drug_history['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
ax2.set_title('Drug Classifier — Accuracy', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ── Plot 3: ATC Classifier Loss ──
atc_epochs = range(1, len(atc_history['loss']) + 1)
ax3 = axes[1, 0]
ax3.plot(atc_epochs, atc_history['loss'], 'b-', label='Train Loss', linewidth=2)
ax3.plot(atc_epochs, atc_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
ax3.set_title('ATC Classifier — Loss', fontsize=12, fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ── Plot 4: ATC Classifier Accuracy ──
ax4 = axes[1, 1]
ax4.plot(atc_epochs, atc_history['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
ax4.plot(atc_epochs, atc_history['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
ax4.set_title('ATC Classifier — Accuracy', fontsize=12, fontweight='bold')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Accuracy')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/08_reporting/training_history_plot.png', dpi=300, bbox_inches='tight')
print("Training history plot saved to: data/08_reporting/training_history_plot.png")

# Print final metrics
print("\n" + "="*60)
print("DRUG CLASSIFIER — Final Metrics")
print("="*60)
print(f"Loss     : {drug_history['loss'][-1]:.4f} (train) | {drug_history['val_loss'][-1]:.4f} (val)")
print(f"Accuracy : {drug_history['accuracy'][-1]:.4f} (train) | {drug_history['val_accuracy'][-1]:.4f} (val)")
best_drug_epoch = np.argmax(drug_history['val_accuracy']) + 1
print(f"Best Val Accuracy: {max(drug_history['val_accuracy']):.4f} at epoch {best_drug_epoch}")
print(f"Epochs run: {len(drug_history['loss'])}")

print("\n" + "="*60)
print("ATC CLASSIFIER — Final Metrics")
print("="*60)
print(f"Loss     : {atc_history['loss'][-1]:.4f} (train) | {atc_history['val_loss'][-1]:.4f} (val)")
print(f"Accuracy : {atc_history['accuracy'][-1]:.4f} (train) | {atc_history['val_accuracy'][-1]:.4f} (val)")
best_atc_epoch = np.argmax(atc_history['val_accuracy']) + 1
print(f"Best Val Accuracy: {max(atc_history['val_accuracy']):.4f} at epoch {best_atc_epoch}")
print(f"Epochs run: {len(atc_history['loss'])}")
print("="*60)

plt.show()
