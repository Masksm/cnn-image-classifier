from pathlib import Path

import numpy as np
import tensorflow as tf


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
    "street"
]

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".jfif",
    ".webp"
}


# ============================================================
# Load model
# ============================================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# Prediction function
# ============================================================

def predict_image(image_path):
    """
    Classify an image using the trained CNN.

    Returns:
        list of dictionaries containing
        class names and confidence scores.
    """

    image_path = Path(image_path)

    # --------------------------------------------------------
    # Check file exists
    # --------------------------------------------------------

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # --------------------------------------------------------
    # Check file extension
    # --------------------------------------------------------

    if image_path.suffix.lower() not in (
        SUPPORTED_EXTENSIONS
    ):

        supported = ", ".join(
            sorted(SUPPORTED_EXTENSIONS)
        )

        raise ValueError(
            f"Unsupported image format: "
            f"{image_path.suffix}\n"
            f"Supported formats: {supported}"
        )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    try:

        image = tf.keras.utils.load_img(
            image_path,
            target_size=IMAGE_SIZE
        )

    except Exception as error:

        raise ValueError(
            f"Could not read the image. "
            f"It may be corrupted or invalid.\n"
            f"Details: {error}"
        )

    # --------------------------------------------------------
    # Convert image to array
    # --------------------------------------------------------

    image_array = tf.keras.utils.img_to_array(
        image
    )

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    # --------------------------------------------------------
    # Find top 3 predictions
    # --------------------------------------------------------

    top_indices = np.argsort(
        predictions
    )[::-1][:3]

    results = []

    for index in top_indices:

        results.append({
            "class": CLASS_NAMES[index],
            "confidence": float(
                predictions[index]
            )
        })

    return results


# ============================================================
# Command-line interface
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("CNN IMAGE CLASSIFIER")
    print("=" * 70)

    image_path = input(
        "\nEnter the path to an image: "
    ).strip()

    try:

        results = predict_image(
            image_path
        )

        print("\n" + "=" * 70)
        print("PREDICTION RESULTS")
        print("=" * 70)

        for position, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\n{position}. "
                f"{result['class']}"
            )

            print(
                f"   Confidence: "
                f"{result['confidence']:.2%}"
            )

    except Exception as error:

        print("\n" + "=" * 70)
        print("PREDICTION FAILED")
        print("=" * 70)

        print(
            f"\n{error}"
        )