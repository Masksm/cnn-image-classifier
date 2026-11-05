from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from data_loader_v2 import load_datasets


# ============================================================
# Paths
# ============================================================

MODEL_PATH = Path(
    "model/best_improved_cnn_v2.keras"
)

RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# Load datasets
# ============================================================

print("\n" + "=" * 70)
print("LOADING V2 TEST DATASET")
print("=" * 70)

train_dataset, validation_dataset, test_dataset = load_datasets()

class_names = test_dataset.class_names

print("\nClasses:")

for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")


# ============================================================
# Load V2 model
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST IMPROVED CNN V2")
print("=" * 70)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("\nModel loaded from:")
print(MODEL_PATH)


# ============================================================
# Evaluate model
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING V2 ON TEST DATA")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)


print("\nV2 Test Results")
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
print("GENERATING V2 PREDICTIONS")
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
# Calculate metrics
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
print("V2 CLASSIFICATION METRICS")
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
print("V2 CLASSIFICATION REPORT")
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
    "improved_v2_classification_report.txt"
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
print("V2 CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# Plot confusion matrix
# ============================================================

plt.figure(
    figsize=(9, 7)
)

plt.imshow(cm)

plt.title(
    "Improved CNN V2 Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(class_names)),
    class_names
)


for i in range(len(class_names)):

    for j in range(len(class_names)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.colorbar()

plt.tight_layout()


cm_path = (
    RESULTS_DIR /
    "improved_v2_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Save predictions
# ============================================================

predictions_path = (
    RESULTS_DIR /
    "improved_v2_predictions.npy"
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
print("V2 EVALUATION COMPLETE")
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