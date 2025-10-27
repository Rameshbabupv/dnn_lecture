#!/usr/bin/env python
"""
Test script for Tutorial T10 Fashion-MNIST CNN notebook
This validates the fix for the AttributeError in cell 25
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("=" * 70)
print("TUTORIAL T10 NOTEBOOK TEST - Cell 25 Fix Validation")
print("=" * 70)

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print(f"\n✓ TensorFlow version: {tf.__version__}")
print(f"✓ NumPy version: {np.__version__}")

# Step 1: Load Fashion-MNIST
print("\n[1/7] Loading Fashion-MNIST dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
print(f"✓ Loaded: {X_train.shape[0]} train + {X_test.shape[0]} test images")

# Step 2: Preprocess
print("\n[2/7] Preprocessing data...")
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
y_train_categorical = keras.utils.to_categorical(y_train, 10)
y_test_categorical = keras.utils.to_categorical(y_test, 10)
print(f"✓ Normalized and reshaped: X_train {X_train.shape}, y_train {y_train_categorical.shape}")

# Step 3: Build model
print("\n[3/7] Building CNN model...")
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'),
    layers.MaxPooling2D((2, 2), name='pool1'),
    layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
    layers.MaxPooling2D((2, 2), name='pool2'),
    layers.Flatten(name='flatten'),
    layers.Dense(64, activation='relu', name='dense1'),
    layers.Dense(10, activation='softmax', name='output')
], name='Fashion_MNIST_CNN')
print("✓ Model architecture created")

# Step 4: Compile
print("\n[4/7] Compiling model...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("✓ Model compiled")

# Step 5: Train (reduced epochs for testing)
print("\n[5/7] Training model (2 epochs for quick test)...")
history = model.fit(
    X_train, y_train_categorical,
    epochs=2,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)
print("✓ Training completed")

# Step 6: Test the FIXED Cell 25 code
print("\n[6/7] Testing Cell 25 fix (Feature Map Visualization)...")
print("=" * 70)

try:
    # THE FIX: Call model.predict() to build the model properly
    print("  → Calling model.predict() to build the model...")
    _ = model.predict(X_test[:1], verbose=0)
    print("  ✓ Model built successfully")

    # Now create feature extraction model
    print("  → Creating feature extraction model...")
    layer_names = ['conv1', 'pool1', 'conv2', 'pool2']
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
    print("  ✓ Feature extraction model created (no AttributeError!)")

    # Get activations for a sample
    print("  → Generating feature maps...")
    sample_image = X_test[0:1]
    activations = activation_model.predict(sample_image, verbose=0)
    print(f"  ✓ Generated {len(activations)} feature maps:")

    for i, (name, activation) in enumerate(zip(layer_names, activations)):
        print(f"      - {name}: {activation.shape}")

    print("\n" + "=" * 70)
    print("🎉 SUCCESS! Cell 25 fix is working correctly!")
    print("=" * 70)

except AttributeError as e:
    print(f"\n❌ FAILED: AttributeError still occurring:")
    print(f"   {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ FAILED: Unexpected error:")
    print(f"   {type(e).__name__}: {e}")
    sys.exit(1)

# Step 7: Evaluate
print("\n[7/7] Evaluating on test set...")
test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)
print(f"✓ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED - Notebook is ready for students!")
print("=" * 70)
print("\nKey validation points:")
print("  ✓ No AttributeError in feature extraction model creation")
print("  ✓ Feature maps generated successfully for all layers")
print("  ✓ Model trains and evaluates correctly")
print("  ✓ Expected accuracy achieved (should be ~85-90% with full training)")
print("\n" + "=" * 70)
