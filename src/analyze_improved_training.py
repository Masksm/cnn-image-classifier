from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

HISTORY_PATH = Path("results/improved_training_history.npy")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# Load training history
# ============================================================

history = np.load(
    HISTORY_PATH,
    allow_pickle=True
).item()


# ============================================================
# Extract metrics
# ============================================================

accuracy = history["accuracy"]
val_accuracy = history["val_accuracy"]

loss = history["loss"]
val_loss = history["val_loss"]

epochs = range(1, len(accuracy) + 1)


# ============================================================
# Find best epoch
# ============================================================

best_epoch = np.argmin(val_loss) + 1

best_val_loss = min(val_loss)
best_val_accuracy = max(val_accuracy)


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 70)
print("IMPROVED CNN TRAINING ANALYSIS")
print("=" * 70)

print(f"\nTotal epochs completed: {len(accuracy)}")

print(f"Best epoch based on validation loss: {best_epoch}")

print(f"Best validation loss: {best_val_loss:.4f}")

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4%}"
)

print("\nTraining history:")
print("-" * 70)

for i in range(len(accuracy)):
    print(
        f"Epoch {i + 1:2d} | "
        f"Accuracy: {accuracy[i]:.4f} | "
        f"Loss: {loss[i]:.4f} | "
        f"Val Accuracy: {val_accuracy[i]:.4f} | "
        f"Val Loss: {val_loss[i]:.4f}"
    )


# ============================================================
# Plot accuracy
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    accuracy,
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_accuracy,
    label="Validation Accuracy"
)

plt.axvline(
    best_epoch,
    linestyle="--",
    label=f"Best Epoch ({best_epoch})"
)

plt.title("Improved CNN Training and Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

accuracy_path = RESULTS_DIR / "improved_accuracy.png"

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Plot loss
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    loss,
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    label="Validation Loss"
)

plt.axvline(
    best_epoch,
    linestyle="--",
    label=f"Best Epoch ({best_epoch})"
)

plt.title("Improved CNN Training and Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

loss_path = RESULTS_DIR / "improved_loss.png"

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Completion message
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print(accuracy_path)

print(loss_path)