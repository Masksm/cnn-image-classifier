from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


MODEL_PATH = Path("model/best_cnn_model.keras")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

TEST_DIR = "dataset/test"

CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]


# ============================================================
# Load model
# ============================================================

print("=" * 70)
print("LOADING MODEL")
print("=" * 70)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# Load test dataset
# ============================================================

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(128, 128),
    batch_size=32,
    shuffle=False
)


# ============================================================
# Generate predictions
# ============================================================

print("\nGenerating predictions...")

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
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion matrix:")
print(cm)


# ============================================================
# Plot confusion matrix
# ============================================================

plt.figure(figsize=(9, 7))

plt.imshow(cm)

plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES
)

for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

confusion_path = RESULTS_DIR / "confusion_matrix.png"

plt.savefig(
    confusion_path,
    dpi=300
)

plt.close()

print(f"\nConfusion matrix saved to:")
print(confusion_path)