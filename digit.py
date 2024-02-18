import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import cv2

# Load the digit recognition model
model = load_model('digit.h5')

# Define prediction function
def predict_digit(image):
    img = ImageOps.grayscale(image).resize((28, 28))
    img_array = np.array(img) #/ 255.0
    img_array = img_array.reshape( 1, 28, 28, 1)
    st.image(img_array)
    prediction = np.argmax(model.predict(img_array))
    return prediction

# Define Streamlit app layout
st.title('Digit Recognition App')
st.write('Draw a digit on the canvas below and click the "Predict" button to see the prediction.')

# Add streamlit-drawable-canvas widget
from streamlit_drawable_canvas import st_canvas

canvas_result = st_canvas(
    fill_color="rgb(255, 255, 255)",  # Fixed fill color with a white background
    stroke_width=10,
    stroke_color="rgb(255,255,255)",  # Black color for drawing
    background_color="rgb(0, 0, 0)",
    height=150,
    width=150,
    drawing_mode="freedraw",
    key="canvas",
)



# Predict button
if st.button('Predict'):
    if canvas_result.image_data is not None:
        #image_data = canvas_result.image_data.astype(np.uint8)
        #img = Image.fromarray(image_data)
         # Apply Gaussian blurring
        image_data = cv2.GaussianBlur(canvas_result.image_data.astype(np.uint8), (5, 5), 0)
        
        # Convert to grayscale and resize
        img = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY))
        
        prediction = predict_digit(img)
        st.success(f'Prediction: {prediction}')
    else:
        st.write("Please draw a digit on the canvas before predicting.")
