import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Model results
# ============================================================

models = [
    "Custom CNN V2",
    "Frozen InceptionV3",
    "Partial Fine-Tuning",
    "Full Fine-Tuning"
]


accuracy = [
    77.40,
    89.9333,
    91.0667,
    91.7000
]


RESULTS_DIR = Path("results")

output_path = (
    RESULTS_DIR /
    "model_accuracy_comparison.png"
)


# ============================================================
# Create chart
# ============================================================

plt.figure(
    figsize=(10, 6)
)

bars = plt.bar(
    models,
    accuracy
)


plt.title(
    "Comparison of CNN Model Accuracy"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Test Accuracy (%)"
)

plt.ylim(
    0,
    100
)

plt.xticks(
    rotation=20,
    ha="right"
)


# ============================================================
# Add percentages
# ============================================================

for bar, value in zip(
    bars,
    accuracy
):

    plt.text(
        bar.get_x() +
        bar.get_width() / 2,

        value + 1,

        f"{value:.2f}%",

        ha="center"
    )


plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\n" + "=" * 70)
print("MODEL COMPARISON GENERATED")
print("=" * 70)

print("\nSaved to:")
print(output_path)