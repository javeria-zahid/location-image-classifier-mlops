import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Load the model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('location_classifier_final.keras')

model = load_model()

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

st.title("Location Image Classifier")
st.write("Provide URL of Location Image for image classification")

image_url = st.text_input("Enter Image URL to Classify..")

if image_url:
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')

        # Preprocess
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        st.write("Predicted Class:")
        st.write(f"**{predicted_class}** ({confidence:.1f}% confidence)")
        st.image(img, use_container_width=True)

    except Exception as e:
        st.error(f"Couldn't load or classify image: {e}")