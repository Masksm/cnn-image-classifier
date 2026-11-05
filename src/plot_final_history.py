from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

HISTORY_PATH = Path(
    "results/inception_full_finetune_history.npy"
)

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
# Extract values
# ============================================================

accuracy = history["accuracy"]
val_accuracy = history["val_accuracy"]

loss = history["loss"]
val_loss = history["val_loss"]

epochs = range(1, len(accuracy) + 1)


# ============================================================
# Accuracy graph
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    accuracy,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_accuracy,
    marker="o",
    label="Validation Accuracy"
)

plt.title(
    "Final InceptionV3 Training and Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.xticks(list(epochs))

plt.legend()

plt.grid(True)

accuracy_path = (
    RESULTS_DIR /
    "final_training_validation_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Loss graph
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    loss,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    marker="o",
    label="Validation Loss"
)

plt.title(
    "Final InceptionV3 Training and Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.xticks(list(epochs))

plt.legend()

plt.grid(True)

loss_path = (
    RESULTS_DIR /
    "final_training_validation_loss.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Print summary
# ============================================================

print("\n" + "=" * 70)
print("FINAL TRAINING GRAPHS GENERATED")
print("=" * 70)

print("\nAccuracy graph:")
print(accuracy_path)

print("\nLoss graph:")
print(loss_path)

print(
    f"\nTotal epochs recorded: "
    f"{len(accuracy)}"
)

print(
    f"Best training accuracy: "
    f"{max(accuracy):.4%}"
)

print(
    f"Best validation accuracy: "
    f"{max(val_accuracy):.4%}"
)