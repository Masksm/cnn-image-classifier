from pathlib import Path

import numpy as np
import tensorflow as tf

from data_loader import load_datasets
from improved_model import (
    build_improved_cnn,
    compile_improved_model
)


# ============================================================
# Configuration
# ============================================================

EPOCHS = 20
MODEL_DIR = Path("model")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "best_improved_cnn.keras"
FINAL_MODEL_PATH = MODEL_DIR / "improved_cnn_final.keras"

HISTORY_PATH = RESULTS_DIR / "improved_training_history.npy"

CSV_LOG_PATH = RESULTS_DIR / "improved_training_log.csv"


# ============================================================
# Load datasets
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASETS")
print("=" * 70)

train_dataset, validation_dataset, test_dataset = load_datasets()

print("\nDatasets loaded successfully.")


# ============================================================
# Build model
# ============================================================

print("\n" + "=" * 70)
print("BUILDING IMPROVED CNN")
print("=" * 70)

model = build_improved_cnn()

model = compile_improved_model(model)

model.summary()


# ============================================================
# Callbacks
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=BEST_MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


csv_logger = tf.keras.callbacks.CSVLogger(
    CSV_LOG_PATH,
    append=False
)


# ============================================================
# Train model
# ============================================================

print("\n" + "=" * 70)
print("TRAINING IMPROVED CNN")
print("=" * 70)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        csv_logger
    ]
)


# ============================================================
# Save final model
# ============================================================

print("\n" + "=" * 70)
print("SAVING FINAL MODEL")
print("=" * 70)

model.save(FINAL_MODEL_PATH)

print(f"\nFinal model saved to:")
print(FINAL_MODEL_PATH)


# ============================================================
# Save training history
# ============================================================

np.save(
    HISTORY_PATH,
    history.history,
    allow_pickle=True
)

print(f"\nTraining history saved to:")
print(HISTORY_PATH)


# ============================================================
# Training summary
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

best_epoch = np.argmin(history.history["val_loss"]) + 1

best_val_loss = min(
    history.history["val_loss"]
)

best_val_accuracy = max(
    history.history["val_accuracy"]
)

print(f"\nBest epoch: {best_epoch}")

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4%}"
)

print(f"\nBest model:")
print(BEST_MODEL_PATH)

print(f"\nFinal model:")
print(FINAL_MODEL_PATH)

print("\nTraining history files:")
print(HISTORY_PATH)
print(CSV_LOG_PATH)