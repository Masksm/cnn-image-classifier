from pathlib import Path
import tensorflow as tf


# ============================================================
# Dataset paths
# ============================================================

DATASET_ROOT = Path("dataset")

TRAIN_DIR = DATASET_ROOT / "train"
VALIDATION_DIR = DATASET_ROOT / "validation"
TEST_DIR = DATASET_ROOT / "test"


# ============================================================
# Image configuration
# ============================================================

# V2 uses the original dataset resolution
IMAGE_SIZE = (150, 150)

BATCH_SIZE = 32


# ============================================================
# Data augmentation
# ============================================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10),
], name="data_augmentation_v2")


# ============================================================
# Dataset loader
# ============================================================

def load_datasets():
    """
    Load training, validation, and test datasets
    using 150 x 150 image resolution.
    """

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=42
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset
    )


# ============================================================
# Test loader
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("V2 DATASET LOADER TEST")
    print("=" * 70)

    train_dataset, validation_dataset, test_dataset = load_datasets()

    print("\nClass names:")
    print(train_dataset.class_names)

    print("\nNumber of classes:")
    print(len(train_dataset.class_names))

    print("\nImage size:")

    for images, labels in train_dataset.take(1):

        print(f"Batch image shape: {images.shape}")
        print(f"Batch label shape: {labels.shape}")

    print("\nV2 dataset loader test completed successfully.")