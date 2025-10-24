import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import os

st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")
st.title("🧠 MNIST Digit Recognizer")

# Load or train model
model_path = "mnist_model.h5"

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    with st.spinner("Training model... please wait ⏳"):
        from tensorflow.keras.datasets import mnist
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Flatten, Dense
        from tensorflow.keras.utils import to_categorical

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        y_train, y_test = to_categorical(y_train), to_categorical(y_test)

        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
        model.save(model_path)
    st.success("Model trained and saved!")

# Canvas for digit input
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    width=192,
    height=192,
    drawing_mode="freedraw",
    key="canvas",
)

# Preprocess image
def preprocess_image(canvas_data):
    if canvas_data.image_data is not None:
        img = Image.fromarray(canvas_data.image_data.astype("uint8"))
        img = ImageOps.grayscale(img)
        img = img.resize((28, 28))
        img = ImageOps.invert(img)
        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28, 1)
        return img
    return None

# Predict and display result
if canvas_result.image_data is not None:
    input_img = preprocess_image(canvas_result)
    if input_img is not None:
        prediction = model.predict(input_img)
        predicted_digit = np.argmax(prediction)
        st.subheader(f"🔢 Predicted Digit: {predicted_digit}")
        st.caption("Confidence scores:")
        st.bar_chart(prediction[0])