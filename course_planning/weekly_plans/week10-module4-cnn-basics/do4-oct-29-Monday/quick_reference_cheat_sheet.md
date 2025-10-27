ve# Tutorial T10: Quick Reference Cheat Sheet

**Week 10, Day 4 - Fashion-MNIST CNN Classification**
**October 29, 2025**

---

## 📥 Dataset Loading

```python
# Load Fashion-MNIST
from tensorflow import keras
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

**Dataset Info:**
- Training: 60,000 images (28×28 grayscale)
- Test: 10,000 images
- 10 classes
- Pixel values: 0-255

---

## 🔧 Data Preprocessing

```python
# Normalize to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape for CNN (add channel dimension)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
```

---

## 🏗️ Build CNN

```python
from tensorflow.keras import layers

model = keras.Sequential([
    # First conv block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    # Second conv block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Fully connected
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

---

## ⚙️ Compile & Train

```python
# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)
```

---

## 📊 Evaluate & Predict

```python
# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predict
predictions = model.predict(X_test[:10])
predicted_classes = np.argmax(predictions, axis=1)
```

---

## 📏 Key Formulas

### Convolution Output Size
```
output_size = (input_size - kernel_size + 2 × padding) / stride + 1
```

**Example:** 28×28 input, 3×3 kernel, stride=1, padding=0
- Output = (28 - 3 + 0) / 1 + 1 = **26×26**

### Pooling Output Size
```
output_size = input_size / pool_size
```

**Example:** 26×26 input, 2×2 pool, stride=2
- Output = 26 / 2 = **13×13**

### Parameter Count (Conv Layer)
```
params = (kernel_h × kernel_w × input_channels + 1) × output_filters
```

**Example:** 3×3 kernel, 1 input channel, 32 output filters
- Params = (3 × 3 × 1 + 1) × 32 = **320**

### Parameter Count (Dense Layer)
```
params = (input_units + 1) × output_units
```

**Example:** 1600 inputs, 64 outputs
- Params = (1600 + 1) × 64 = **102,464**

---

## 🎨 Visualization Code

### Training History
```python
import matplotlib.pyplot as plt

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

### Learned Filters
```python
# Extract first conv layer weights
filters, biases = model.layers[0].get_weights()

# Normalize
f_min, f_max = filters.min(), filters.max()
filters_norm = (filters - f_min) / (f_max - f_min)

# Visualize
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(filters_norm[:, :, 0, i], cmap='viridis')
    plt.axis('off')
plt.show()
```

### Feature Maps
```python
# Create feature extraction model
layer_outputs = [model.get_layer('conv1').output]
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)

# Get activations
sample_image = X_test[0:1]
activations = activation_model.predict(sample_image)

# Visualize first 8 feature maps
activation = activations[0]
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(activation[0, :, :, i], cmap='viridis')
    plt.axis('off')
plt.show()
```

---

## 🔍 Common Layer Types

| Layer | Purpose | Parameters |
|-------|---------|------------|
| `Conv2D(filters, (h, w))` | Learn spatial features | `(h×w×in_ch + 1) × filters` |
| `MaxPooling2D((h, w))` | Downsample, translation invariance | 0 |
| `Flatten()` | Convert 2D → 1D | 0 |
| `Dense(units)` | Fully connected, classification | `(inputs + 1) × units` |
| `Dropout(rate)` | Regularization | 0 |

---

## 🎯 When to Use What?

### Padding
- **`'valid'`** (default): No padding, output shrinks
  - Use when: You want dimension reduction
- **`'same'`**: Pad to maintain dimensions
  - Use when: Deep networks, need consistent sizes

### Activation Functions
- **ReLU** (`'relu'`): Hidden layers (fast, effective)
- **Softmax** (`'softmax'`): Multi-class output (probabilities sum to 1)
- **Sigmoid** (`'sigmoid'`): Binary classification

### Optimizers
- **Adam** (`'adam'`): Best default choice (adaptive learning rate)
- **SGD** (`'sgd'`): Simple, sometimes better for large datasets
- **RMSprop** (`'rmsprop'`): Good for RNNs

### Loss Functions
- **`categorical_crossentropy`**: Multi-class (one-hot encoded labels)
- **`sparse_categorical_crossentropy`**: Multi-class (integer labels)
- **`binary_crossentropy`**: Binary classification

---

## ⚠️ Common Errors & Fixes

### Error: Shape Mismatch
```
ValueError: Input 0 of layer "sequential" is incompatible with the layer
```
**Fix:** Reshape data to add channel dimension
```python
X_train = X_train.reshape(-1, 28, 28, 1)
```

### Error: Wrong Loss for Labels
```
ValueError: Shapes (None, 1) and (None, 10) are incompatible
```
**Fix:** Use `sparse_categorical_crossentropy` OR one-hot encode labels
```python
# Option 1: Change loss
model.compile(loss='sparse_categorical_crossentropy', ...)

# Option 2: One-hot encode
y_train = keras.utils.to_categorical(y_train, 10)
```

### Error: Model Not Learning (Accuracy ~10%)
**Possible causes:**
1. Forgot to normalize data (divide by 255)
2. Wrong loss function
3. Learning rate too high/low
4. Data not shuffled

**Fix:**
```python
# Normalize
X_train = X_train / 255.0

# Check loss function matches labels
model.compile(loss='categorical_crossentropy', ...)  # for one-hot
# OR
model.compile(loss='sparse_categorical_crossentropy', ...)  # for integers
```

### Error: Out of Memory
```
ResourceExhaustedError: OOM when allocating tensor
```
**Fix:** Reduce batch size
```python
model.fit(X_train, y_train, batch_size=32, ...)  # instead of 128
```

---

## 💡 Quick Tips

1. **Always normalize images** to [0, 1] range
2. **Use validation split** (0.2) to monitor overfitting
3. **Start with small models**, then add complexity
4. **Conv filters typically double** after each pooling (32 → 64 → 128)
5. **Kernel size 3×3** is most common (good balance)
6. **Use `model.summary()`** to verify architecture
7. **Save your model** after training: `model.save('model.h5')`
8. **GPU accelerates training** 5-10× (use Google Colab if no local GPU)

---

## 📐 Architecture Design Template

```python
model = keras.Sequential([
    # Block 1: Learn low-level features
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    # Block 2: Learn mid-level features
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Optional: Block 3 for deeper networks
    # layers.Conv2D(128, (3, 3), activation='relu'),
    # layers.MaxPooling2D((2, 2)),

    # Classification head
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    # layers.Dropout(0.5),  # Optional: reduce overfitting
    layers.Dense(num_classes, activation='softmax')
])
```

---

## 🏃 Complete Workflow (Copy-Paste Ready)

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# 1. Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# 2. Preprocess
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# 3. Build model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 4. Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2,
                    verbose=1)

# 6. Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# 7. Plot
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

---

## 🎓 Expected Results

- **Training Accuracy**: 95-97% (after 10 epochs)
- **Validation Accuracy**: 90-92%
- **Test Accuracy**: 90-92%
- **Training Time**: 2-3 minutes (CPU), 30 seconds (GPU)

---

## 🏠 Homework Reminder

**Task 1:** Manual convolution calculation (6×6, 3×3 kernel)
**Task 2:** Design CNN for MNIST with justification
**Task 3:** Modify tutorial code, experiment, document results

**Due:** Before Week 11 lecture

---

## 📖 Next Week Preview

- Famous architectures: LeNet, AlexNet, VGG, ResNet
- Advanced techniques: Dropout, Batch Normalization
- Transfer learning fundamentals

---

**Keep this cheat sheet handy during the tutorial!** 📋
