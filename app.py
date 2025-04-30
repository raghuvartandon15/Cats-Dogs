import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cats_vs_dogs_model.h5")

model = load_model()

# Title
st.title("🐶🐱 Cat vs Dog Classifier")

# File uploader
uploaded_file = st.file_uploader("Upload an image of a cat or dog", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).resize((100, 100))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 100, 100, 3)

    # Show a smaller image
    st.image(image, caption="Uploaded Image", width=150)  # Removed use_column_width

    # Predict
    prediction = model.predict(img_array)[0][0]  # Get scalar from prediction

    label = "Dog 🐶" if prediction < 0.5 else "Cat 🐱"
    confidence = 1 - prediction if prediction < 0.5 else prediction

    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"**Confidence:** {confidence:.2%}")
