from pathlib import Path
import tensorflow as tf


# Dataset paths
DATASET_ROOT = Path("dataset")

TRAIN_DIR = DATASET_ROOT / "train"
VALIDATION_DIR = DATASET_ROOT / "validation"
TEST_DIR = DATASET_ROOT / "test"


# Image configuration
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32


# Data augmentation for training images
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10),
], name="data_augmentation")


def load_datasets():
    """
    Load training, validation, and test datasets
    from their respective directory structures.
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

    return train_dataset, validation_dataset, test_dataset