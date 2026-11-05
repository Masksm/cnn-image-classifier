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

from data_loader_v2 import load_datasets


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = Path(
    "model/best_inception_finetuned.keras"
)

RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# Load dataset
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATASET")
print("=" * 70)

train_dataset, validation_dataset, test_dataset = (
    load_datasets()
)

class_names = test_dataset.class_names


print("\nClasses:")

for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")


# ============================================================
# Load model
# ============================================================

print("\n" + "=" * 70)
print("LOADING FINE-TUNED INCEPTIONV3")
print("=" * 70)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("\nModel loaded from:")
print(MODEL_PATH)


# ============================================================
# Evaluate
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING FINE-TUNED MODEL")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)


print("\nTest Results")
print("-" * 70)

print(
    f"Test Loss:     {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.4%}"
)


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

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# Metrics
# ============================================================

precision = precision_score(
    y_true,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)


print("\n" + "=" * 70)
print("FINE-TUNED CLASSIFICATION METRICS")
print("=" * 70)

print(
    f"\nPrecision: {precision:.4%}"
)

print(
    f"Recall:    {recall:.4%}"
)

print(
    f"F1-score:  {f1:.4%}"
)


# ============================================================
# Classification report
# ============================================================

print("\n" + "=" * 70)
print("FINE-TUNED CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print("\n")
print(report)


report_path = (
    RESULTS_DIR /
    "inception_finetuned_classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


# ============================================================
# Confusion matrix
# ============================================================

print("\n" + "=" * 70)
print("FINE-TUNED CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# Save confusion matrix
# ============================================================

cm_path = (
    RESULTS_DIR /
    "inception_finetuned_confusion_matrix.npy"
)

np.save(
    cm_path,
    cm
)


# ============================================================
# Save predictions
# ============================================================

predictions_path = (
    RESULTS_DIR /
    "inception_finetuned_predictions.npy"
)

np.save(
    predictions_path,
    {
        "y_true": y_true,
        "y_pred": y_pred,
        "class_names": class_names
    },
    allow_pickle=True
)


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 70)
print("FINE-TUNED EVALUATION COMPLETE")
print("=" * 70)

print(
    f"\nTest accuracy: {test_accuracy:.4%}"
)

print(
    f"Precision:     {precision:.4%}"
)

print(
    f"Recall:        {recall:.4%}"
)

print(
    f"F1-score:      {f1:.4%}"
)

print("\nGenerated files:")

print(report_path)

print(cm_path)

print(predictions_path)