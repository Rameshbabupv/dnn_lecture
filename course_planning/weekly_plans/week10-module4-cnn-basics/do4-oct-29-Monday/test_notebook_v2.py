#!/usr/bin/env python
"""
Test script V2 - Alternative approach for creating feature extraction model
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("=" * 70)
print("TUTORIAL T10 - TESTING ALTERNATIVE FEATURE EXTRACTION APPROACH")
print("=" * 70)

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print(f"\n✓ TensorFlow version: {tf.__version__}")

# Quick data load
print("\n[1/4] Loading minimal dataset for testing...")
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_test = X_test.astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)
print("✓ Data loaded")

# Build model
print("\n[2/4] Building model...")
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'),
    layers.MaxPooling2D((2, 2), name='pool1'),
    layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
    layers.MaxPooling2D((2, 2), name='pool2'),
    layers.Flatten(name='flatten'),
    layers.Dense(64, activation='relu', name='dense1'),
    layers.Dense(10, activation='softmax', name='output')
], name='Fashion_MNIST_CNN')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("✓ Model built")

# Test different approaches
print("\n[3/4] Testing feature extraction approaches...")
print("=" * 70)

# Approach 1: Try direct model.input (will fail)
print("\n❌ Approach 1: Direct model.input access")
try:
    activation_model = keras.Model(inputs=model.input, outputs=model.layers[0].output)
    print("   SUCCESS (unexpected)")
except AttributeError as e:
    print(f"   FAILED as expected: {str(e)[:60]}...")

# Approach 2: Call model first, then access
print("\n⚠️  Approach 2: Call predict() then access model.input")
try:
    _ = model.predict(X_test[:1], verbose=0)
    activation_model = keras.Model(inputs=model.input, outputs=model.layers[0].output)
    print("   SUCCESS")
except AttributeError as e:
    print(f"   FAILED: {str(e)[:60]}...")

# Approach 3: Build with explicit Input layer
print("\n✅ Approach 3: Rebuild model with explicit Input (RECOMMENDED)")
try:
    # Create a new input tensor
    inputs = keras.Input(shape=(28, 28, 1))

    # Apply each layer sequentially
    x = inputs
    layer_outputs = {}
    for layer in model.layers:
        x = layer(x)
        layer_outputs[layer.name] = x

    # Create feature extraction model
    layer_names = ['conv1', 'pool1', 'conv2', 'pool2']
    outputs_list = [layer_outputs[name] for name in layer_names]

    activation_model = keras.Model(inputs=inputs, outputs=outputs_list)

    # Test it
    sample_image = X_test[0:1]
    activations = activation_model.predict(sample_image, verbose=0)

    print("   ✅ SUCCESS! Feature extraction working")
    print(f"   Generated {len(activations)} feature maps:")
    for name, activation in zip(layer_names, activations):
        print(f"      - {name}: {activation.shape}")

except Exception as e:
    print(f"   FAILED: {type(e).__name__}: {e}")

print("\n" + "=" * 70)
print("✅ Approach 3 is the correct solution for TensorFlow 2.16+")
print("=" * 70)
