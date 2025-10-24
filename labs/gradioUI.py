#!/usr/bin/env python3


# !pip install -q gradio==4.44.0 opencv-python
# pip install tensorflow matplotlib

import gradio as gr
import numpy as np
import cv2
import tensorflow as tf

# Create and train a simple MNIST model
def create_mnist_model():
    """Create and train a simple MNIST MLP model"""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape to flatten the images
    x_train = x_train.reshape(-1, 28*28)
    x_test = x_test.reshape(-1, 28*28)
    
    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # Create model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Training model...")
    # Train model (just a few epochs for demo)
    model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=1, validation_split=0.1)
    
    # Save model
    model.save("mnist_mlp.keras")
    print("Model saved as mnist_mlp.keras")
    
    return model

# Load or create model
try:
    model = tf.keras.models.load_model("mnist_mlp.keras")
    print("Loaded existing model")
except:
    print("No existing model found, creating new one...")
    model = create_mnist_model()

def preprocess_img(img: np.ndarray) -> np.ndarray:
    """
    img: HxWx3 or HxW (uint8), white background with black strokes from sketchpad.
    Returns: (1, 784) float32 in [0,1]
    """
    if img is None:
        return None
    # Convert to grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    # Invert if background is dark and digit is light (sketchpad often gives black bg, white ink)
    # Heuristic: compare means
    if gray.mean() < 127:
        gray = 255 - gray

    # Threshold to get a clean digit
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Find bounding box of the digit
    ys, xs = np.where(th > 0)
    if len(xs) == 0 or len(ys) == 0:
        # Empty drawing -> return blank
        roi = np.zeros((28, 28), dtype=np.uint8)
    else:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        crop = th[y1:y2+1, x1:x2+1]

        # Make square by padding
        h, w = crop.shape
        s = max(h, w)
        pad_y = (s - h) // 2
        pad_x = (s - w) // 2
        crop_sq = cv2.copyMakeBorder(crop, pad_y, s - h - pad_y, pad_x, s - w - pad_x,
                                     cv2.BORDER_CONSTANT, value=0)

        # Resize to 28x28
        roi = cv2.resize(crop_sq, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize to [0,1] and flatten
    roi = roi.astype("float32") / 255.0
    roi = roi.reshape(1, 28*28)
    return roi

def predict_draw(img):
    roi = preprocess_img(img)
    if roi is None:
        return "No image"
    probs = model.predict(roi, verbose=0)[0]
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    # Return top-3 nicely
    top3_idx = probs.argsort()[-3:][::-1]
    top3 = [(int(i), float(probs[i])) for i in top3_idx]
    return f"Predicted: {pred}  (confidence: {conf:.2f})\nTop-3: {top3}"

demo = gr.Interface(
    fn=predict_draw,
    inputs=gr.Sketchpad(shape=(280, 280), brush=gr.Brush(default_color="white", size=20),  # white ink on black
                        bg_color="black", type="numpy"),
    outputs="text",
    title="Draw a Digit (0-9) and Predict",
    description="Draw a big, centered digit. Try different brush sizes if needed."
)

demo.launch(debug=False)

