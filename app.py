
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
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

st.markdown("""
<style>

.hero-container{
    background: linear-gradient(90deg, #0F4C81, #2196F3);
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
}

.hero-title{
    font-size:42px;
    font-weight:bold;
    margin-bottom:8px;
}

.hero-subtitle{
    font-size:20px;
    opacity:0.95;
}

</style>

<div class="hero-container">

<div class="hero-title">
🩺 COVID-19 Chest X-Ray Classification System
</div>

<div class="hero-subtitle">
Deep Learning using <b>MobileNetV2 + Keras Tuner</b>
</div>

</div>

""", unsafe_allow_html=True)


# =====================================================
# Model Performance Dashboard
# =====================================================

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric(
        label="🎯 Test Accuracy",
        value="90.91%"
    )

with metric2:
    st.metric(
        label="📈 ROC-AUC",
        value="0.9884"
    )

with metric3:
    st.metric(
        label="🧠 Architecture",
        value="MobileNetV2"
    )

with metric4:
    st.metric(
        label="🏷️ Classes",
        value="3"
    )

st.divider()



st.caption("Deep Learning | MobileNetV2 + Keras Tuner")

with st.sidebar:

    st.header("📋 Project Overview")

    st.info(
        """
        **COVID-19 Chest X-ray Classification**

        AI-powered chest X-ray image classification using
        Transfer Learning and Deep Learning.
        """
    )

    st.divider()

    st.subheader("🤖 Model")

    st.write("**Architecture:** MobileNetV2")
    st.write("**Optimization:** Keras Tuner")
    st.write("**Framework:** TensorFlow / Keras")

    st.divider()

    st.subheader("📊 Model Performance")

    st.metric(
        label="Test Accuracy",
        value="90.91%"
    )

    st.metric(
        label="ROC-AUC",
        value="0.9884"
    )

    st.metric(
        label="Classes",
        value="3"
    )

    st.divider()

    st.subheader("🧪 Dataset Classes")

    st.success("🟢 Normal")
    st.error("🔴 Covid")
    st.warning("🟠 Viral Pneumonia")

    st.divider()

    st.caption(
        "⚠️ This application is intended for educational and research purposes only."
    )

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

    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, caption="Uploaded Chest X-ray", use_container_width=True)

    with col2:
        if st.button("🔍 Analyze Chest X-ray", type="primary"):
            with st.spinner("Analyzing image..."):
                img = prepare_image(image)
                probabilities = model.predict(img, verbose=0)[0]

            pred_index = int(np.argmax(probabilities))
            pred_label = label_encoder.inverse_transform([pred_index])[0]
            confidence = probabilities[pred_index] * 100
            

            st.subheader("🧠 AI Prediction Dashboard")

            # ================================
            # Prediction Result Card
            # ================================

            if pred_label == "Covid":
                bg_color = "#fdecea"
                border_color = "#d32f2f"
                icon = "🦠"

            elif pred_label == "Normal":
                bg_color = "#e8f5e9"
                border_color = "#2e7d32"
                icon = "✅"

            else:
                bg_color = "#fff3e0"
                border_color = "#ef6c00"
                icon = "🫁"

            st.markdown(
                f"""
                <div style="
                    background-color:{bg_color};
                    border-left:8px solid {border_color};
                    padding:20px;
                    border-radius:12px;
                    box-shadow:0px 2px 8px rgba(0,0,0,0.15);
                    margin-bottom:20px;
                ">

                <h2 style="margin:0;">
                    {icon} Prediction Result
                </h2>

                <h1 style="margin-top:10px; color:{border_color};">
                    {pred_label}
                </h1>

                <h3>
                    Confidence Score: {confidence:.2f}%
                </h3>

                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.subheader("Class Probabilities")

            # ============================================
            # Confidence Progress Bars
            # ============================================

            st.markdown("### 📈 Confidence for Each Class")

            for disease, prob in zip(label_encoder.classes_, probabilities):

                st.write(f"**{disease}**")

                st.progress(float(prob))

                st.caption(f"{prob * 100:.2f}%")


            results = pd.DataFrame({
                "Disease": label_encoder.classes_,
                "Probability (%)": np.round(probabilities * 100, 2)
            })

            # ============================================
            # Top Predictions
            # ============================================

            ranked_results = results.sort_values(
                by="Probability (%)",
                ascending=False
            ).reset_index(drop=True)

            medals = ["🥇", "🥈", "🥉"]

            ranked_results.insert(
                0,
                "Rank",
                medals[:len(ranked_results)]
            )

            st.subheader("🏆 Top Predictions")

            st.dataframe(
                ranked_results,
                use_container_width=True,
                hide_index=True
            )

            chart = results.set_index("Disease")
            fig = px.bar(
                results,
                x="Disease",
                y="Probability (%)",
                color="Probability (%)",
                text="Probability (%)",
                color_continuous_scale="Blues"
            )

            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")

            fig.update_layout(
                template="plotly_white",
                height=450,
                title="Prediction Probability Distribution",
                yaxis_title="Probability (%)",
                xaxis_title=""
            )

            st.plotly_chart(fig, use_container_width=True)

            # ============================================
            # AI Prediction Summary
            # ============================================

            st.subheader("📝 AI Prediction Summary")

            if confidence >= 95:
                confidence_level = "Very High"

            elif confidence >= 80:
                confidence_level = "High"

            elif confidence >= 60:
                confidence_level = "Moderate"

            else:
                confidence_level = "Low"

            summary = f"""
            ### 🩺 AI Assessment

            The uploaded chest X-ray image has been classified as **{pred_label}**
            with a **{confidence_level} confidence** of **{confidence:.2f}%**.

            **Model Interpretation**

            {disease_info[pred_label]}

            ---

            **Important Note**

            This prediction is generated by a deep learning model trained for
            educational and research purposes.

            It should **not** be considered a substitute for professional medical
            diagnosis or clinical evaluation.
            """

            st.info(summary)

# =====================================================
# Model Comparison
# =====================================================

st.divider()

st.subheader("📊 Model Comparison")

comparison_df = pd.DataFrame({

    "Model": [
        "Baseline CNN",
        "Deep CNN",
        "VGG16 Transfer Learning",
        "ResNet50 Transfer Learning",
        "MobileNetV2",
        "MobileNetV2 + Keras Tuner ⭐"
    ],

    "Test Accuracy": [
        "65.15%",
        "-",
        "86.36%",
        "81.82%",
        "89.39%",
        "90.91%"
    ],

    "ROC-AUC": [
        "0.8593",
        "-",
        "0.9866",
        "0.9474",
        "0.9790",
        "0.9884"
    ]

})

st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True
)



with st.expander("🔬 AI Model Information", expanded=False):

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🧠 Model Details")

        st.markdown("""
- **Architecture:** MobileNetV2
- **Transfer Learning:** Yes
- **Hyperparameter Tuning:** Keras Tuner
- **Framework:** TensorFlow / Keras
- **Input Size:** 224 × 224 RGB
- **Output Classes:** 3
        """)

    with col2:

        st.markdown("### 📊 Final Performance")

        st.markdown("""
- **Test Accuracy:** 90.91%
- **ROC-AUC Score:** 0.9884
- **Final Model:** MobileNetV2 + Keras Tuner
- **Deployment:** Streamlit
- **Image Preprocessing:** MobileNetV2 preprocess_input()
        """)

    st.divider()

    st.markdown("### 📚 Dataset")

    st.info(
        """
The model was trained to classify chest X-ray images into the following categories:

- 🦠 Covid
- ✅ Normal
- 🫁 Viral Pneumonia

The final deployed model was selected after comparing multiple deep learning architectures, including Baseline CNN, Deep CNN, VGG16, ResNet50, and MobileNetV2.
"""
    )


st.warning(
    "Disclaimer: This application is intended for educational and research purposes only. "
    "It must not be used as a substitute for professional medical diagnosis."
)
