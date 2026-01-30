import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# -------------------------------------------------
# Page settings
# -------------------------------------------------
st.set_page_config(
    page_title="Cotton Leaf Disease Detection",
    layout="centered"
)

st.title("🌱 Cotton Leaf Disease Detection System")
st.write("Upload a cotton leaf image to check plant health and disease type")

# -------------------------------------------------
# Load models (cached for speed)
# -------------------------------------------------
@st.cache_resource
def load_models():
    modelA = tf.keras.models.load_model("MobileNet_PartA_BEST.h5")  # Healthy / Diseased
    modelB = tf.keras.models.load_model("MobileNet_PartB_BEST.h5")  # Disease type
    return modelA, modelB

modelA, modelB = load_models()

# -------------------------------------------------
# Part-B class names (same order as training)
# -------------------------------------------------
class_names_B = [
    "Aphids",
    "BacterialBlight",
    "BollRot",
    "Healthy",
    "LeafCurl",
    "PowderyMildew"
]

# -------------------------------------------------
# Image uploader
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose a cotton leaf image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------
# Prediction pipeline
# -------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing (same for both models)
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    st.markdown("---")
    st.subheader("🔍 Analysis Result")

    # -------------------------
    # PART-A: Healthy vs Diseased
    # -------------------------
    predA = modelA.predict(img)[0][0]

    if predA > 0.5:
        st.success("🌿 Leaf Status: **Healthy**")
        st.info("No disease detected. The plant leaf appears healthy.")

    else:
        st.error("⚠️ Leaf Status: **Diseased**")

        # -------------------------
        # PART-B: Disease classification
        # -------------------------
        predsB = modelB.predict(img)
        disease_id = np.argmax(predsB)
        confidence = np.max(predsB)

        st.subheader("🧪 Detected Disease")
        st.write(f"**Disease Name:** {class_names_B[disease_id]}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")

st.markdown("---")
st.caption("Developed using Deep Learning (MobileNet) and Streamlit")
