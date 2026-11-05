from pathlib import Path

DATASET_ROOT = Path("dataset")

SPLITS = ["train", "validation", "test"]

CLASSES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]


def count_images(folder):
    """Count JPG images inside a folder."""
    return len(list(folder.glob("*.jpg")))


print("=" * 60)
print("FINAL DATASET VERIFICATION")
print("=" * 60)

grand_total = 0

for split in SPLITS:

    print(f"\n{split.upper()}")
    print("-" * 40)

    split_total = 0

    for class_name in CLASSES:

        class_folder = DATASET_ROOT / split / class_name

        count = count_images(class_folder)

        print(f"{class_name:12} : {count}")

        split_total += count

    print(f"{'TOTAL':12} : {split_total}")

    grand_total += split_total


print("\n" + "=" * 60)
print(f"GRAND TOTAL: {grand_total}")
print("=" * 60)