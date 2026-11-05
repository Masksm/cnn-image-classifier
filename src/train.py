from pathlib import Path

import tensorflow as tf

from data_loader import load_datasets
from model import build_cnn_model, compile_model


# ============================================================
# Configuration
# ============================================================

EPOCHS = 30

MODEL_DIR = Path("model")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# Load datasets
# ============================================================

print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)

train_dataset, validation_dataset, test_dataset = load_datasets()


# ============================================================
# Build model
# ============================================================

print("\n" + "=" * 70)
print("BUILDING CNN MODEL")
print("=" * 70)

model = build_cnn_model()
model = compile_model(model)

model.summary()


# ============================================================
# Callbacks
# ============================================================

checkpoint_path = MODEL_DIR / "best_cnn_model.keras"

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


# ============================================================
# Train model
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)


# ============================================================
# Save final model
# ============================================================

final_model_path = MODEL_DIR / "cnn_image_classifier.keras"

model.save(final_model_path)


# ============================================================
# Save training history
# ============================================================

history_path = RESULTS_DIR / "training_history.npy"

import numpy as np

np.save(
    history_path,
    history.history,
    allow_pickle=True
)


# ============================================================
# Training complete
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"\nBest model saved to:")
print(checkpoint_path)

print(f"\nFinal model saved to:")
print(final_model_path)

print(f"\nTraining history saved to:")
print(history_path)

print(f"\nEpochs completed: {len(history.history['loss'])}")