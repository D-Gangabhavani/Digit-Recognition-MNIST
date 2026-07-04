import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# 1. Page Configuration
st.set_page_config(page_title="MNIST Digit Recognition", layout="centered")
st.title("Handwritten Digit Recognition using MNIST")
st.write("Draw a digit (0-9) inside the box below and click Predict.")

# 2. Load Model Safely
@st.cache_resource
def load_digit_model():
    return tf.keras.models.load_model('models/digit_model.h5')

try:
    model = load_digit_model()
except Exception as e:
    st.error("Model file not found. Please run 'train.py' first.")

# 3. Canvas Setup (Black background, White stroke)
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=14,            # Perfect stroke width for 28x28 resize
    stroke_color="#FFFFFF",      # White ink
    background_color="#000000",  # Black background
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# 4. Prediction Logic
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert canvas drawing to numpy array (Shape: 280, 280, 4)
        img_raw = np.array(canvas_result.image_data, dtype=np.uint8)
        
        # FOOLPROOF METHOD: Convert RGBA to Grayscale using standard formula
        # This guarantees we capture the white drawings correctly across all library versions
        img_gray = (0.2989 * img_raw[:,:,0] + 0.5870 * img_raw[:,:,1] + 0.1140 * img_raw[:,:,2]).astype(np.uint8)
        
        # Check if user actually drew something (if max pixel value is greater than 0)
        if np.max(img_gray) == 0:
            st.warning("Please draw something on the canvas first.")
        else:
            # Resize directly using PIL to 28x28
            img_pil = Image.fromarray(img_gray)
            img_resized = img_pil.resize((28, 28), Image.Resampling.BILINEAR)
            
            # Convert back to numpy and normalize (0.0 to 1.0)
            img_final = np.array(img_resized) / 255.0
            
            # Reshape for CNN input: (1, 28, 28, 1)
            img_final = img_final.reshape(1, 28, 28, 1)
            
            # Make prediction
            prediction = model.predict(img_final)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            # Display Final Outputs
            st.success(f"Predicted Digit: {predicted_digit}")
            st.info(f"Confidence Level: {confidence:.2f}%")
    else:
        st.warning("Please draw something on the canvas first.")