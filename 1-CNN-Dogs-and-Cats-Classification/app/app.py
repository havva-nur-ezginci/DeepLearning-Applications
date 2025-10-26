import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config(page_title="Cat - Dog Classifier", page_icon="🐶", layout="centered")

st.title("🐶🐱 Cat - Dog Image Classifier")
st.markdown("Upload an image and let the trained CNN model predict whether it’s a **Cat** or a **Dog**!")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Load model
    model_path = "models/cat-and-dog.keras"
    try:
        model = tf.keras.models.load_model(model_path)
        pred = model.predict(img_array)[0][0]
        label = "🐶 Dog" if pred > 0.5 else "🐱 Cat"
        confidence = pred if pred > 0.5 else 1 - pred

        st.markdown(f"### Prediction: {label}")
        st.progress(float(confidence))
        st.write(f"**Confidence:** {confidence:.2%}")

    except Exception as e:
        st.error("Model could not be loaded. Please check the model path.")
        st.exception(e)
