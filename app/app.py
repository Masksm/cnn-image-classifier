from pathlib import Path
import sys

import streamlit as st
from PIL import Image


# ============================================================
# Allow Python to find the src directory
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))


# ============================================================
# Import prediction function
# ============================================================

from predict import predict_image


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🖼️",
    layout="centered"
)


# ============================================================
# Confidence threshold
# ============================================================

CONFIDENCE_THRESHOLD = 0.49


# ============================================================
# Page title
# ============================================================

st.title("CNN Image Classification System")

st.write(
    "Upload an image and the trained CNN will predict its class."
)

st.write(
    "The system classifies images into six categories: "
    "buildings, forest, glacier, mountain, sea, and street."
)


# ============================================================
# File uploader
# ============================================================

uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "jfif",
        "webp"
    ]
)


# ============================================================
# Process uploaded image
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Display uploaded image
    # --------------------------------------------------------

    try:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            width="stretch"
        )

    except Exception as error:
        st.error(f"Unable to open this image: {error}")
        st.stop()

    # --------------------------------------------------------
    # Classification button
    # --------------------------------------------------------

    if st.button(
        "Classify Image",
        type="primary"
    ):

        # ----------------------------------------------------
        # Save uploaded image temporarily
        # ----------------------------------------------------

        temporary_path = (
            PROJECT_ROOT
            / "app"
            / f"temp_uploaded_image{Path(uploaded_file.name).suffix}"
        )

        try:

            with open(temporary_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            # ------------------------------------------------
            # Run prediction
            # ------------------------------------------------

            with st.spinner("Classifying image..."):

                results = predict_image(temporary_path)

            # ------------------------------------------------
            # Check prediction results
            # ------------------------------------------------

            if not results:

                st.warning(
                    "The model could not produce a prediction "
                    "for this image."
                )

            else:

                # --------------------------------------------
                # Get the highest prediction
                # --------------------------------------------

                top_result = results[0]

                confidence = float(
                    top_result["confidence"]
                )

                predicted_class = top_result["class"].title()

                # --------------------------------------------
                # Apply the 49% confidence threshold
                # --------------------------------------------

                if confidence < CONFIDENCE_THRESHOLD:

                    st.warning(
                        "No suitable category found. "
                        "This image may not fit any of the "
                        "available categories, or the model "
                        "could not classify it confidently."
                    )

                    st.metric(
                        "Highest Confidence",
                        f"{confidence:.2%}"
                    )

                    st.info(
                        "Please upload an image of a building, "
                        "forest, glacier, mountain, sea, or street."
                    )

                else:

                    # ----------------------------------------
                    # Display main prediction
                    # ----------------------------------------

                    st.success(
                        f"Predicted Category: {predicted_class}"
                    )

                    st.metric(
                        "Confidence",
                        f"{confidence:.2%}"
                    )

                    # ----------------------------------------
                    # Display top predictions
                    # ----------------------------------------

                    st.subheader("Top Predictions")

                    for position, result in enumerate(
                        results[:3],
                        start=1
                    ):

                        result_confidence = float(
                            result["confidence"]
                        )

                        # Keep progress values within 0 and 1
                        progress_value = max(
                            0.0,
                            min(1.0, result_confidence)
                        )

                        st.write(
                            f"**{position}. "
                            f"{result['class'].title()}**"
                        )

                        st.progress(
                            progress_value,
                            text=f"{result_confidence:.2%}"
                        )

        except Exception as error:

            st.error(
                f"Classification failed: {error}"
            )

        finally:

            # -----------------------------------------------
            # Delete temporary file
            # -----------------------------------------------

            if temporary_path.exists():

                temporary_path.unlink()
