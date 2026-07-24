import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import joblib

st.set_page_config(page_title="COVID-19 Chest X-Ray Classification", page_icon="🩺", layout="wide")

IMG_HEIGHT = 224
IMG_WIDTH = 224

@st.cache_resource
def load_artifacts():
    model = load_model("covid_classifier.keras")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

model, label_encoder = load_artifacts()

st.title("🩺 COVID-19 Chest X-Ray Classification System")
st.caption("Deep Learning | MobileNetV2 + Keras Tuner")

with st.sidebar:
    st.header("Project Information")
    st.markdown("""
**Model:** MobileNetV2 + Keras Tuner

**Dataset Classes**
- Covid
- Normal
- Viral Pneumonia

**Performance**
- Test Accuracy: **90.91%**
- ROC-AUC: **0.9884**

This application is for educational and research purposes only.
""")

st.write("""
Upload a chest X-ray image in **JPG**, **JPEG**, or **PNG** format.
The application will classify the image into one of the following categories:

- Covid
- Normal
- Viral Pneumonia
""")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

def prepare_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(image).astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

disease_info = {
    "Covid": "The model detected imaging features most consistent with COVID-19.",
    "Normal": "The model did not detect significant abnormalities associated with the trained disease classes.",
    "Viral Pneumonia": "The model detected imaging features more consistent with Viral Pneumonia."
}

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1,1])

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        if st.button("🔍 Analyze Chest X-ray", type="primary"):
            with st.spinner("Analyzing image..."):
                img = prepare_image(image)
                probabilities = model.predict(img, verbose=0)[0]

            pred_index = int(np.argmax(probabilities))
            pred_label = label_encoder.inverse_transform([pred_index])[0]
            confidence = probabilities[pred_index] * 100

            st.success(f"### Prediction: {pred_label}")
            st.metric("Confidence", f"{confidence:.2f}%")

            st.subheader("Class Probabilities")

            results = pd.DataFrame({
                "Disease": label_encoder.classes_,
                "Probability (%)": np.round(probabilities * 100, 2)
            })

            st.dataframe(results, use_container_width=True, hide_index=True)
            chart = results.set_index("Disease")
            st.bar_chart(chart)

            st.subheader("Prediction Interpretation")
            st.info(disease_info[pred_label])

st.divider()

with st.expander("Model Details"):
    st.markdown("""
- **Architecture:** MobileNetV2 (Transfer Learning)
- **Optimization:** Keras Tuner
- **Framework:** TensorFlow / Keras
- **Image Size:** 224 × 224
- **Classes:** 3
""")

st.warning(
    "Disclaimer: This application is intended for educational and research purposes only. "
    "It must not be used as a substitute for professional medical diagnosis."
)
