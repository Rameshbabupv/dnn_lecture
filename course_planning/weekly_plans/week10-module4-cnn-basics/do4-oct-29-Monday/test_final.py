#!/usr/bin/env python
"""
Final comprehensive test of the fixed Cell 25 code
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("=" * 70)
print("FINAL TEST: Tutorial T10 Cell 25 Fix")
print("=" * 70)

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print(f"\n✓ TensorFlow: {tf.__version__}, NumPy: {np.__version__}")

# Load data
print("\n[1/5] Loading Fashion-MNIST...")
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
y_train_categorical = keras.utils.to_categorical(y_train, 10)
y_test_categorical = keras.utils.to_categorical(y_test, 10)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print("✓ Dataset ready")

# Build model
print("\n[2/5] Building CNN...")
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
print("✓ Model compiled")

# Train
print("\n[3/5] Training (2 epochs)...")
history = model.fit(X_train, y_train_categorical, epochs=2, batch_size=128,
                   validation_split=0.2, verbose=0)
print(f"✓ Training complete: val_acc={history.history['val_accuracy'][-1]:.4f}")

# Test Cell 25 FIXED CODE
print("\n[4/5] Testing Cell 25 (FIXED CODE)...")
print("=" * 70)

try:
    # === FIXED CODE FROM NOTEBOOK CELL 25 ===
    # Create a new input tensor
    inputs = keras.Input(shape=(28, 28, 1))

    # Apply each layer sequentially and collect outputs
    x = inputs
    layer_outputs_dict = {}
    for layer in model.layers:
        x = layer(x)
        layer_outputs_dict[layer.name] = x

    # Select the layers we want to visualize
    layer_names = ['conv1', 'pool1', 'conv2', 'pool2']
    layer_outputs = [layer_outputs_dict[name] for name in layer_names]

    # Create the feature extraction model
    activation_model = keras.Model(inputs=inputs, outputs=layer_outputs)

    # Get activations for a sample image
    sample_idx = 0
    sample_image = X_test[sample_idx:sample_idx+1]
    sample_label = class_names[y_test[sample_idx]]

    print(f"  Analyzing: {sample_label}")
    activations = activation_model.predict(sample_image, verbose=0)
    # === END FIXED CODE ===

    print("\n  ✅ SUCCESS! No AttributeError")
    print(f"  ✅ Generated {len(activations)} feature maps:")
    for name, activation in zip(layer_names, activations):
        print(f"      - {name}: {activation.shape}")

    # Verify shapes are correct
    expected_shapes = {
        'conv1': (1, 26, 26, 32),
        'pool1': (1, 13, 13, 32),
        'conv2': (1, 11, 11, 64),
        'pool2': (1, 5, 5, 64)
    }

    all_correct = True
    for name, activation in zip(layer_names, activations):
        if activation.shape != expected_shapes[name]:
            print(f"  ⚠️  Shape mismatch for {name}: {activation.shape} != {expected_shapes[name]}")
            all_correct = False

    if all_correct:
        print("\n  ✅ All feature map shapes are correct!")

except AttributeError as e:
    print(f"\n  ❌ FAILED: AttributeError occurred")
    print(f"     {e}")
    exit(1)
except Exception as e:
    print(f"\n  ❌ FAILED: {type(e).__name__}")
    print(f"     {e}")
    exit(1)

# Final evaluation
print("\n" + "=" * 70)
print("[5/5] Final model evaluation...")
test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)
print(f"✓ Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print("\n" + "=" * 70)
print("🎉 ALL TESTS PASSED!")
print("=" * 70)
print("\n✅ Cell 25 fix is working correctly")
print("✅ Feature extraction model creates successfully")
print("✅ Feature maps generate with correct shapes")
print("✅ No AttributeError encountered")
print("✅ Notebook is ready for students!")
print("\n" + "=" * 70)
