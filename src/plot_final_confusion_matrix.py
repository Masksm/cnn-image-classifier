import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Final confusion matrix
# ============================================================

cm = np.array([
    [403, 0, 0, 1, 6, 27],
    [1, 466, 2, 2, 3, 0],
    [1, 2, 458, 72, 19, 1],
    [0, 1, 53, 460, 10, 1],
    [4, 0, 3, 4, 499, 0],
    [31, 0, 0, 2, 3, 465]
])


class_names = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]


RESULTS_DIR = Path("results")

output_path = (
    RESULTS_DIR /
    "final_confusion_matrix.png"
)


# ============================================================
# Create figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 8)
)

image = ax.imshow(cm)

ax.set_xticks(
    np.arange(len(class_names))
)

ax.set_yticks(
    np.arange(len(class_names))
)

ax.set_xticklabels(
    class_names,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(
    class_names
)


ax.set_xlabel(
    "Predicted Class"
)

ax.set_ylabel(
    "Actual Class"
)

ax.set_title(
    "Confusion Matrix - Final InceptionV3 Model"
)


# ============================================================
# Add values
# ============================================================

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\n" + "=" * 70)
print("FINAL CONFUSION MATRIX GENERATED")
print("=" * 70)

print("\nSaved to:")
print(output_path)