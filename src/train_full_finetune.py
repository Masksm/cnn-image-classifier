from pathlib import Path

import numpy as np
import tensorflow as tf

from data_loader_v2 import load_datasets
from full_finetune_model import (
    build_full_finetuning_model,
    compile_full_finetuning_model
)


# ============================================================
# Configuration
# ============================================================

EPOCHS = 8

MODEL_DIR = Path("model")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


BEST_MODEL_PATH = (
    MODEL_DIR /
    "best_inception_full_finetuned.keras"
)

FINAL_MODEL_PATH = (
    MODEL_DIR /
    "inception_full_finetuned_final.keras"
)

HISTORY_PATH = (
    RESULTS_DIR /
    "inception_full_finetune_history.npy"
)

CSV_LOG_PATH = (
    RESULTS_DIR /
    "inception_full_finetune_log.csv"
)


# ============================================================
# Load datasets
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASETS")
print("=" * 70)

train_dataset, validation_dataset, test_dataset = (
    load_datasets()
)

print("\nDatasets loaded successfully.")


# ============================================================
# Build model
# ============================================================

print("\n" + "=" * 70)
print("BUILDING FULL FINE-TUNING MODEL")
print("=" * 70)

model = build_full_finetuning_model()

model = compile_full_finetuning_model(
    model
)

model.summary()


# ============================================================
# Parameter count
# ============================================================

trainable_parameters = sum(
    tf.keras.backend.count_params(weight)
    for weight in model.trainable_weights
)

non_trainable_parameters = sum(
    tf.keras.backend.count_params(weight)
    for weight in model.non_trainable_weights
)

print(
    f"\nTrainable parameters: "
    f"{trainable_parameters:,}"
)

print(
    f"Non-trainable parameters: "
    f"{non_trainable_parameters:,}"
)


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
    patience=2,
    restore_best_weights=True,
    verbose=1
)


csv_logger = tf.keras.callbacks.CSVLogger(
    CSV_LOG_PATH,
    append=False
)


# ============================================================
# Train
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FULL INCEPTIONV3")
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
print("SAVING FULL FINE-TUNED MODEL")
print("=" * 70)

model.save(
    FINAL_MODEL_PATH
)

print("\nFinal model saved to:")
print(FINAL_MODEL_PATH)


# ============================================================
# Save history
# ============================================================

np.save(
    HISTORY_PATH,
    history.history,
    allow_pickle=True
)

print("\nTraining history saved to:")
print(HISTORY_PATH)


# ============================================================
# Summary
# ============================================================

best_epoch = (
    np.argmin(
        history.history["val_loss"]
    ) + 1
)

best_val_loss = min(
    history.history["val_loss"]
)

best_val_accuracy = max(
    history.history["val_accuracy"]
)


print("\n" + "=" * 70)
print("FULL FINE-TUNING COMPLETE")
print("=" * 70)

print(
    f"\nEpochs completed: "
    f"{len(history.history['accuracy'])}"
)

print(
    f"Best epoch: "
    f"{best_epoch}"
)

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4%}"
)

print("\nBest model:")
print(BEST_MODEL_PATH)

print("\nFinal model:")
print(FINAL_MODEL_PATH)

print("\nHistory:")
print(HISTORY_PATH)

print("\nCSV log:")
print(CSV_LOG_PATH)