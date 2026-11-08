from pathlib import Path
import sys

import streamlit as st
from PIL import Image, ImageOps

# Make the src directory importable when Streamlit is launched from the
# repository root with: streamlit run app/app.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from predict import predict_image  # noqa: E402


st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🖼️",
    layout="centered",
)

CONFIDENCE_THRESHOLD = 0.49
CLASS_DESCRIPTION = (
    "Buildings · Forest · Glacier · Mountain · Sea · Street"
)

st.title("CNN Image Classifier")
st.caption("A six-class image recognition system powered by a fine-tuned InceptionV3 model.")
st.write(CLASS_DESCRIPTION)

with st.sidebar:
    st.header("About the model")
    st.metric("Test accuracy", "91.70%")
    st.write("Input resolution: **150 × 150 pixels**")
    st.write("Architecture: **InceptionV3, full fine-tuning**")
    st.caption(
        "The displayed score is the model's prediction score, not a guarantee "
        "that a prediction is correct. The 49% rejection threshold is a "
        "simple heuristic, not an out-of-distribution detector."
    )

uploaded_file = st.file_uploader(
    "Upload an image to classify",
    type=["jpg", "jpeg", "png", "bmp", "jfif", "webp"],
    help="Choose a clear image of a building, forest, glacier, mountain, sea, or street.",
)

if uploaded_file is None:
    st.info("Upload an image to get started.")
    st.stop()

try:
    image_bytes = uploaded_file.getvalue()
    preview = Image.open(uploaded_file)
    preview = ImageOps.exif_transpose(preview).convert("RGB")

    st.image(
        preview,
        caption=f"Selected image: {uploaded_file.name}",
        width="stretch",
    )
except Exception:
    st.error("The selected file could not be opened as a valid image.")
    st.stop()

if st.button("Classify image", type="primary", use_container_width=True):
    try:
        with st.spinner("Analyzing image..."):
            results = predict_image(image_bytes)

        if not results:
            st.warning("The model did not return a prediction.")
            st.stop()

        best_result = results[0]
        best_class = best_result["class"].title()
        confidence = float(best_result["confidence"])

        st.divider()
        st.subheader("Prediction")

        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(
                "The model is not confident enough to assign a category. "
                "Try a clearer image containing one of the supported scene types."
            )
        else:
            st.success(f"Predicted category: **{best_class}**")

        st.metric("Highest prediction score", f"{confidence:.2%}")

        st.subheader("Top 3 predictions")
        for rank, result in enumerate(results, start=1):
            score = float(result["confidence"])
            label_col, score_col = st.columns([3, 1])
            label_col.write(f"**{rank}. {result['class'].title()}**")
            score_col.write(f"**{score:.2%}**")
            st.progress(max(0.0, min(1.0, score)))

        st.caption(
            "Predictions are limited to the six categories used during training. "
            "Images outside those categories may still receive a high score."
        )

    except FileNotFoundError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Classification failed: {error}")
