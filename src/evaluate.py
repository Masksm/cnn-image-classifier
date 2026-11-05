from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from data_loader import load_datasets


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = Path("model/best_cnn_model.keras")

CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]


# ============================================================
# Load trained model
# ============================================================

print("=" * 70)
print("LOADING TRAINED MODEL")
print("=" * 70)

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully.")


# ============================================================
# Load datasets
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATASET")
print("=" * 70)

train_dataset, validation_dataset, test_dataset = load_datasets()


# ============================================================
# Evaluate test dataset
# ============================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print("\nTest Loss:")
print(f"{test_loss:.4f}")

print("\nTest Accuracy:")
print(f"{test_accuracy:.4%}")


# ============================================================
# Generate predictions
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

y_true = []
y_pred = []


for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# Calculate metrics
# ============================================================

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 70)
print("CLASSIFICATION METRICS")
print("=" * 70)

print(f"\nPrecision : {precision:.4%}")
print(f"Recall    : {recall:.4%}")
print(f"F1-score  : {f1:.4%}")


# ============================================================
# Detailed classification report
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# Confusion matrix
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n")
print(cm)