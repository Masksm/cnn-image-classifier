from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "best_inception_full_finetuned.keras"

IMAGE_SIZE = (150, 150)

CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".jfif",
    ".webp",
}


# ============================================================
# Model loading
# ============================================================

@lru_cache(maxsize=1)
def load_classifier_model():
    """Load the trained model once and reuse it for later predictions."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Make sure Git LFS is installed and run 'git lfs pull'."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    output_units = model.output_shape[-1]
    if output_units != len(CLASS_NAMES):
        raise ValueError(
            f"Model has {output_units} output classes, "
            f"but {len(CLASS_NAMES)} class names are configured."
        )

    return model


# ============================================================
# Image preparation
# ============================================================

def _open_image(image_input):
    """Open an image supplied as bytes, a path, or a file-like object."""
    if isinstance(image_input, (bytes, bytearray, memoryview)):
        source = BytesIO(image_input)
    else:
        source = image_input

    try:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image)
            return image.convert("RGB").resize(
                (IMAGE_SIZE[1], IMAGE_SIZE[0]),
                Image.Resampling.LANCZOS,
            )
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError(
            "The uploaded file is not a valid, readable image."
        ) from error


# ============================================================
# Prediction
# ============================================================

def predict_image(image_input):
    """
    Classify an image using the trained CNN.

    image_input may be image bytes, a file-like object, or a filesystem path.
    Returns the three highest-scoring classes and their model scores.
    """
    if isinstance(image_input, (str, Path)):
        image_path = Path(image_input)

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported image format: {image_path.suffix}. "
                f"Supported formats: {supported}"
            )

    image = _open_image(image_input)
    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    model = load_classifier_model()
    predictions = np.asarray(
        model.predict(image_array, verbose=0)[0],
        dtype=np.float64,
    )

    if predictions.shape != (len(CLASS_NAMES),):
        raise ValueError(
            "The model returned an unexpected prediction shape."
        )

    if not np.all(np.isfinite(predictions)):
        raise ValueError("The model returned invalid prediction scores.")

    top_indices = np.argsort(predictions)[::-1][:3]

    return [
        {
            "class": CLASS_NAMES[index],
            "confidence": float(predictions[index]),
        }
        for index in top_indices
    ]


# ============================================================
# Command-line interface
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CNN IMAGE CLASSIFIER")
    print("=" * 70)

    image_path = input("Enter the path to an image: ").strip().strip('"')

    try:
        results = predict_image(image_path)

        print("\nPREDICTION RESULTS")
        for position, result in enumerate(results, start=1):
            print(
                f"{position}. {result['class'].title()}: "
                f"{result['confidence']:.2%}"
            )

    except Exception as error:
        print(f"\nPREDICTION FAILED: {error}")
