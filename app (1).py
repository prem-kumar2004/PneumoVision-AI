import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import cv2

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="PneumoVision AI",
    page_icon="🫁",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pneumonia_efficientnetb0.keras"
IMG_SIZE = (224, 224)


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        text-align: center;
        padding: 0.5rem 0 1.5rem 0;
    }
    .hero h1 {
        font-size: 2.6rem;
        margin-bottom: 0.25rem;
    }
    .hero p {
        color: #6b7280;
        font-size: 1.05rem;
    }
    .metric-card {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px 12px;
        text-align: center;
        min-height: 120px;
        background: #ffffff;
    }
    .metric-label {
        color: #6b7280;
        font-size: 0.9rem;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Make sure the .keras model is in the same GitHub folder as app.py."
        )
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )


try:
    model = load_model()
except Exception as exc:
    st.error("The trained model could not be loaded.")
    st.exception(exc)
    st.stop()


# ------------------------------------------------------------
# Grad-CAM
# ------------------------------------------------------------

def make_gradcam_heatmap(input_tensor):
    base_model = model.get_layer("efficientnetb0")
    last_conv = base_model.get_layer("top_conv")

    grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=last_conv.output,
    )

    with tf.GradientTape() as tape:
        augmented = model.get_layer(
            "data_augmentation"
        )(input_tensor, training=False)

        conv_output = grad_model(augmented)

        x = model.get_layer(
            "global_average_pooling2d"
        )(conv_output)
        x = model.get_layer("dropout")(x, training=False)
        x = model.get_layer("dense")(x)
        x = model.get_layer("dropout_1")(x, training=False)
        prediction = model.get_layer("dense_1")(x)[:, 0]

    gradients = tape.gradient(prediction, conv_output)

    if gradients is None:
        return None

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2),
    )

    conv_output = conv_output[0]
    pooled_gradients = pooled_gradients[0]

    heatmap = tf.reduce_sum(
        conv_output * pooled_gradients,
        axis=-1,
    )

    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)

    if float(maximum) == 0:
        return None

    return (heatmap / maximum).numpy()


def create_explanation(original_image, heatmap):
    if heatmap is None:
        return None, None, None

    height, width = original_image.shape[:2]

    heatmap_resized = cv2.resize(
        heatmap,
        (width, height),
    )

    mask = np.uint8(heatmap_resized >= 0.55) * 255

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    marked = original_image.copy()
    box = None

    if contours:
        contour = max(
            contours,
            key=cv2.contourArea,
        )

        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            box = (x, y, w, h)

            cv2.rectangle(
                marked,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                4,
            )

            cv2.putText(
                marked,
                "AI-highlighted region",
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

    return marked, heatmap_resized, box


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🫁 PneumoVision AI</h1>
        <p>Chest X-Ray Pneumonia Detection with EfficientNetB0 + Grad-CAM</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("📋 Model Information")
    st.write("**Architecture:** EfficientNetB0")
    st.write("**Task:** Binary classification")
    st.write("**Classes:** NORMAL / PNEUMONIA")
    st.write("**Input:** 224 × 224 RGB")

    st.divider()

    st.header("📊 Test Performance")
    st.metric("Accuracy", "86.38%")
    st.metric("ROC-AUC", "95.04%")
    st.metric("Pneumonia Recall", "97%")

    st.divider()

    st.caption(
        "Grad-CAM shows regions that influenced the model. "
        "It is not a clinically validated lesion detector."
    )


# ------------------------------------------------------------
# Upload + analysis
# ------------------------------------------------------------

st.subheader("📤 Upload Chest X-Ray")

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG, or PNG chest X-ray",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    preview_col, info_col = st.columns(
        [1.15, 0.85],
        gap="large",
    )

    with preview_col:
        st.subheader("🩻 Image Preview")
        st.image(
            image,
            use_container_width=True,
        )

    with info_col:
        st.subheader("🔍 Analysis")

        st.info(
            "Upload the X-ray, then press the button below. "
            "The model will classify it and generate an explanation."
        )

        analyze = st.button(
            "🔍 Analyze X-Ray",
            type="primary",
            use_container_width=True,
        )

    if analyze:
        with st.spinner("Analyzing X-ray..."):
            resized = image.resize(IMG_SIZE)
            image_array = np.asarray(
                resized,
                dtype=np.float32,
            )
            input_tensor = tf.convert_to_tensor(
                np.expand_dims(image_array, axis=0)
            )

            prediction = float(
                model.predict(
                    input_tensor,
                    verbose=0,
                )[0][0]
            )

            pneumonia_probability = prediction
            normal_probability = 1.0 - prediction

            if prediction >= 0.5:
                result = "PNEUMONIA"
                confidence = pneumonia_probability
            else:
                result = "NORMAL"
                confidence = normal_probability

            heatmap = make_gradcam_heatmap(
                input_tensor
            )

            marked, resized_heatmap, box = (
                create_explanation(
                    np.asarray(image),
                    heatmap,
                )
            )

        st.divider()
        st.subheader("🤖 Analysis Result")

        c1, c2, c3 = st.columns(
            3,
            gap="large",
        )

        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Prediction</div>
                    <div class="metric-value">{result}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{confidence:.2%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Pneumonia Probability</div>
                    <div class="metric-value">{pneumonia_probability:.2%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.subheader("📊 Class Probabilities")
        st.write(f"**NORMAL:** {normal_probability:.2%}")
        st.progress(normal_probability)
        st.write(f"**PNEUMONIA:** {pneumonia_probability:.2%}")
        st.progress(pneumonia_probability)

        st.divider()
        st.subheader("🔥 Explainable AI")

        if resized_heatmap is not None and marked is not None:
            heat_col, mark_col = st.columns(
                2,
                gap="large",
            )

            with heat_col:
                st.markdown("**Attention Heatmap**")
                overlay = np.asarray(image).copy()
                overlay_heatmap = np.uint8(
                    255 * resized_heatmap
                )
                overlay_heatmap = cv2.applyColorMap(
                    overlay_heatmap,
                    cv2.COLORMAP_JET,
                )
                overlay_heatmap = cv2.cvtColor(
                    overlay_heatmap,
                    cv2.COLOR_BGR2RGB,
                )

                blended = cv2.addWeighted(
                    overlay,
                    0.60,
                    overlay_heatmap,
                    0.40,
                    0,
                )

                st.image(
                    blended,
                    use_container_width=True,
                )

            with mark_col:
                st.markdown("**AI-Highlighted Region**")
                st.image(
                    marked,
                    use_container_width=True,
                )

            if box is not None:
                x, y, w, h = box
                st.success(
                    f"Strongest localized activation found "
                    f"in an approximate {w} × {h} pixel region."
                )
            else:
                st.info(
                    "No sufficiently strong localized activation "
                    "was found."
                )
        else:
            st.info(
                "Grad-CAM could not generate a reliable heatmap "
                "for this image."
            )

        st.divider()
        st.subheader("🧠 Interpretation")

        if result == "PNEUMONIA":
            st.warning(
                f"The model classified this X-ray as "
                f"**PNEUMONIA** with **{confidence:.2%} confidence**."
            )
        else:
            st.success(
                f"The model classified this X-ray as "
                f"**NORMAL** with **{confidence:.2%} confidence**."
            )

        st.warning(
            "⚠️ Medical disclaimer: This is an educational "
            "machine-learning project and has not been clinically "
            "validated. The prediction and highlighted region "
            "must not be used for diagnosis or treatment decisions."
        )

else:
    st.info(
        "👆 Upload a chest X-ray above to begin the analysis."
    )
