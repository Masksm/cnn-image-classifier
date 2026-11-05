from pathlib import Path
import shutil
import random

# Original dataset
SOURCE_ROOT = Path(r"C:\Users\marvel\Downloads\archive")

TRAIN_SOURCE = SOURCE_ROOT / "seg_train" / "seg_train"
TEST_SOURCE = SOURCE_ROOT / "seg_test" / "seg_test"

# Project dataset directory
PROJECT_DATASET = Path("dataset")

TRAIN_DEST = PROJECT_DATASET / "train"
VALIDATION_DEST = PROJECT_DATASET / "validation"
TEST_DEST = PROJECT_DATASET / "test"

# Dataset classes
CLASSES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

# Percentage of training images used for validation
VALIDATION_RATIO = 0.20

# Makes the split reproducible
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def create_directories():
    """Create the required dataset directories."""

    for class_name in CLASSES:
        (TRAIN_DEST / class_name).mkdir(parents=True, exist_ok=True)
        (VALIDATION_DEST / class_name).mkdir(parents=True, exist_ok=True)
        (TEST_DEST / class_name).mkdir(parents=True, exist_ok=True)


def split_training_data():
    """Split original training images into training and validation sets."""

    for class_name in CLASSES:

        source_class = TRAIN_SOURCE / class_name

        images = list(source_class.glob("*.jpg"))

        random.shuffle(images)

        validation_count = int(len(images) * VALIDATION_RATIO)

        validation_images = images[:validation_count]
        training_images = images[validation_count:]

        print(f"\nClass: {class_name}")
        print(f"Total: {len(images)}")
        print(f"Training: {len(training_images)}")
        print(f"Validation: {len(validation_images)}")

        for image in training_images:
            shutil.copy2(
                image,
                TRAIN_DEST / class_name / image.name
            )

        for image in validation_images:
            shutil.copy2(
                image,
                VALIDATION_DEST / class_name / image.name
            )


def copy_test_data():
    """Copy the original test images into the project test directory."""

    for class_name in CLASSES:

        source_class = TEST_SOURCE / class_name

        images = list(source_class.glob("*.jpg"))

        print(f"\nTest class: {class_name}")
        print(f"Test images: {len(images)}")

        for image in images:
            shutil.copy2(
                image,
                TEST_DEST / class_name / image.name
            )


def main():
    print("=" * 60)
    print("DATASET ORGANIZATION")
    print("=" * 60)

    create_directories()

    print("\nSplitting training data...")
    split_training_data()

    print("\nCopying test data...")
    copy_test_data()

    print("\n" + "=" * 60)
    print("DATASET ORGANIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()