from pathlib import Path
from PIL import Image
from collections import Counter

DATASET_PATH = Path(
    r"C:\Users\marvel\Downloads\archive\seg_train\seg_train"
)

dimensions = Counter()
image_modes = Counter()
corrupted = []

images = list(DATASET_PATH.rglob("*.jpg"))

for image_path in images:
    try:
        with Image.open(image_path) as img:
            # Check whether the image is valid
            img.verify()

        # Reopen the image because verify() invalidates the image object
        with Image.open(image_path) as img:
            dimensions[img.size] += 1
            image_modes[img.mode] += 1

    except Exception as e:
        corrupted.append((str(image_path), str(e)))


# =========================
# DATASET INSPECTION REPORT
# =========================

print("=" * 50)
print("DATASET INSPECTION REPORT")
print("=" * 50)

print(f"\nTotal JPG files found: {len(images)}")
print(f"Valid images: {len(images) - len(corrupted)}")
print(f"Corrupted images: {len(corrupted)}")

print("\nImage dimensions:")
for dimension, count in dimensions.most_common():
    print(f"  {dimension}: {count}")

print("\nImage color modes:")
for mode, count in image_modes.most_common():
    print(f"  {mode}: {count}")

print(f"\nTotal images successfully inspected: {sum(dimensions.values())}")


# =========================
# CORRUPTED IMAGE DETAILS
# =========================

if corrupted:
    print("\n" + "=" * 50)
    print("CORRUPTED / UNREADABLE IMAGES")
    print("=" * 50)

    for image_path, error in corrupted:
        print(f"\nFile: {image_path}")
        print(f"Error: {error}")
else:
    print("\nNo corrupted images found.")

print("\n" + "=" * 50)