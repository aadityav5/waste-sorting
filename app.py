import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "best.pt"


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_NAMES = {
    0: "Non-Recyclable",
    1: "Recyclable"
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    """Load the YOLO model."""
    try:
        return YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


model = load_model()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sortyx Waste Detector",
    page_icon="♻️",
    layout="centered"
)


# ============================================================
# MAIN UI
# ============================================================

st.title("♻️ Sortyx Waste Detector")

st.subheader(
    "AI-Based Waste Classification"
)

st.write(
    "Upload a waste image and the AI will classify it as "
    "**Recyclable** or **Non-Recyclable**."
)


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file is not None:

    # --------------------------------------------------------
    # Display uploaded image
    # --------------------------------------------------------

    st.image(
        uploaded_file,
        caption="Uploaded Waste Image",
        use_container_width=True
    )


    # --------------------------------------------------------
    # Analyze button
    # --------------------------------------------------------

    submitted = st.button(
        "🔍 Analyze Waste",
        type="primary",
        use_container_width=True
    )


    if submitted:

        image_bytes = uploaded_file.getvalue()

        nparr = np.frombuffer(
            image_bytes,
            np.uint8
        )

        img = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )


        # ----------------------------------------------------
        # Run YOLO
        # ----------------------------------------------------

        with st.spinner("🔍 Analyzing waste..."):

            if model is not None:

                results = model(
                    img,
                    conf=0.25,
                    verbose=False
                )[0]

            else:

                st.error(
                    "Model not loaded. Check that best.pt is present."
                )

                results = None


        # ====================================================
        # PROCESS RESULTS
        # ====================================================

        if (
            results is not None
            and results.boxes is not None
            and len(results.boxes) > 0
        ):

            # ------------------------------------------------
            # Get predictions
            # ------------------------------------------------

            predictions = []

            for box in results.boxes:

                cls = int(box.cls[0])

                conf = float(box.conf[0])

                label = CLASS_NAMES.get(
                    cls,
                    "Unknown"
                )

                predictions.append({
                    "class_id": cls,
                    "label": label,
                    "confidence": conf
                })


            # ------------------------------------------------
            # Select highest-confidence detection
            # ------------------------------------------------

            best_prediction = max(
                predictions,
                key=lambda x: x["confidence"]
            )


            final_class = best_prediction["class_id"]

            final_label = best_prediction["label"]

            final_confidence = best_prediction["confidence"]


            # =================================================
            # ANNOTATED IMAGE
            # =================================================

            annotated = results.plot(
                line_width=2
            )

            st.image(
                annotated,
                channels="BGR",
                caption="AI Detection Result",
                use_container_width=True
            )


            # =================================================
            # FINAL RESULT
            # =================================================

            st.markdown("## 🧠 Final Result")


            if final_class == 1:

                st.success(
                    f"♻️ RECYCLABLE"
                )

            else:

                st.error(
                    f"🗑️ NON-RECYCLABLE"
                )


            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            st.metric(
                "AI Confidence",
                f"{final_confidence:.2%}"
            )


            # =================================================
            # DETECTION DETAILS
            # =================================================

            st.markdown(
                "### 🔍 Detection Details"
            )


            for i, prediction in enumerate(
                predictions,
                start=1
            ):

                st.write(
                    f"**Item {i}:** "
                    f"{prediction['label']} — "
                    f"{prediction['confidence']:.2%}"
                )


        else:

            # ------------------------------------------------
            # No detection
            # ------------------------------------------------

            st.warning(
                "⚠️ No waste detected. "
                "Please upload a clearer image."
            )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "### ℹ️ About Sortyx"
)

st.sidebar.info(
    """
This AI system classifies waste into two categories:

♻️ **Recyclable** — Class 1

🗑️ **Non-Recyclable** — Class 0

The trained YOLO model analyzes the uploaded
image and determines the final waste category.
"""
)


st.sidebar.markdown(
    "### 🎯 Classes"
)

st.sidebar.write(
    "**Class 0:** 🗑️ Non-Recyclable"
)

st.sidebar.write(
    "**Class 1:** ♻️ Recyclable"
)