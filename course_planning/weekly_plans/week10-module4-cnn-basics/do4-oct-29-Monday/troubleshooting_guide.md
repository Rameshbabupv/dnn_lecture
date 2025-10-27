# Tutorial T10: Troubleshooting Guide

**Common Errors and Solutions for CNN Training**
**Week 10, Day 4 - October 29, 2025**

---

## Table of Contents

1. [Installation & Environment Issues](#installation--environment-issues)
2. [Data Loading Errors](#data-loading-errors)
3. [Model Architecture Errors](#model-architecture-errors)
4. [Training Problems](#training-problems)
5. [Memory Errors](#memory-errors)
6. [Visualization Errors](#visualization-errors)
7. [Google Colab Specific Issues](#google-colab-specific-issues)

---

## Installation & Environment Issues

### Error 1: ModuleNotFoundError: No module named 'tensorflow'

**Error Message:**
```
ModuleNotFoundError: No module named 'tensorflow'
```

**Cause:** TensorFlow is not installed in your environment.

**Solutions:**

**Local Environment:**
```bash
# Install TensorFlow
pip install tensorflow

# Or with specific version
pip install tensorflow==2.15.0

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

**Google Colab:**
```python
# TensorFlow is pre-installed in Colab
# If needed, upgrade:
!pip install --upgrade tensorflow
```

---

### Error 2: ImportError: cannot import name 'keras'

**Error Message:**
```
ImportError: cannot import name 'keras' from 'tensorflow'
```

**Cause:** Using outdated TensorFlow version.

**Solution:**
```bash
# Upgrade TensorFlow
pip install --upgrade tensorflow

# Or use standalone Keras (not recommended)
pip install keras
```

**In code, use:**
```python
# Recommended (TensorFlow 2.x)
from tensorflow import keras

# NOT this (deprecated)
import keras
```

---

### Error 3: GPU Not Found / CUDA Errors

**Error Message:**
```
Could not load dynamic library 'cudart64_110.dll'
```

**Cause:** CUDA libraries not installed or incompatible version.

**Solutions:**

**Option 1: Use CPU (simpler)**
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
```

**Option 2: Use Google Colab GPU**
- Runtime → Change runtime type → Hardware accelerator → GPU

**Option 3: Install CUDA locally**
- Install CUDA Toolkit matching your TensorFlow version
- See: https://www.tensorflow.org/install/gpu

**Training time comparison:**
- CPU: 2-3 minutes
- GPU: 30-45 seconds
- For tutorial purposes, CPU is acceptable!

---

## Data Loading Errors

### Error 4: Failed to Download Dataset

**Error Message:**
```
Exception: URL fetch failure on https://storage.googleapis.com/...
```

**Cause:** Network issues or firewall blocking download.

**Solutions:**

**Option 1: Retry with timeout**
```python
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Retry loading
try:
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
except:
    print("Retrying download...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
```

**Option 2: Manual download**
```python
# Download from alternative source
!wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
!wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
!wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
!wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

# Then load manually (see Fashion-MNIST documentation)
```

**Option 3: Use MNIST instead (for practice)**
```python
# Similar dataset, smaller size
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
```

---

### Error 5: IndexError: list index out of range

**Error Message:**
```
IndexError: list index out of range (accessing y_train[70000])
```

**Cause:** Trying to access index beyond dataset size.

**Solution:**
```python
# Check dataset size first
print(f"Training set size: {len(X_train)}")  # 60,000
print(f"Test set size: {len(X_test)}")       # 10,000

# Access valid indices
sample_image = X_train[0]  # First image (valid)
# NOT: X_train[70000]  # Out of bounds!
```

---

## Model Architecture Errors

### Error 6: Shape Mismatch - Input

**Error Message:**
```
ValueError: Input 0 of layer "sequential" is incompatible with the layer:
expected shape=(None, 28, 28, 1), found shape=(None, 28, 28)
```

**Cause:** Forgot to add channel dimension to input data.

**Solution:**
```python
# WRONG: X_train shape is (60000, 28, 28)
X_train = X_train / 255.0

# CORRECT: Add channel dimension
X_train = X_train.reshape(-1, 28, 28, 1)  # Now (60000, 28, 28, 1)
print(f"Shape after reshape: {X_train.shape}")

# Verify before training
assert X_train.shape == (60000, 28, 28, 1), "Wrong shape!"
```

---

### Error 7: Shape Mismatch - Output

**Error Message:**
```
ValueError: Shapes (None, 1) and (None, 10) are incompatible
```

**Cause:** Labels and loss function mismatch.

**Problem & Solutions:**

**Scenario 1: Using integer labels with wrong loss**
```python
# WRONG
y_train = y_train  # Shape: (60000,) - integers 0-9
model.compile(loss='categorical_crossentropy', ...)  # Expects one-hot

# FIX Option A: One-hot encode labels
y_train = keras.utils.to_categorical(y_train, 10)  # Shape: (60000, 10)

# FIX Option B: Change loss function
model.compile(loss='sparse_categorical_crossentropy', ...)  # Accepts integers
```

**Scenario 2: Using one-hot labels with wrong loss**
```python
# WRONG
y_train = keras.utils.to_categorical(y_train, 10)  # Shape: (60000, 10)
model.compile(loss='sparse_categorical_crossentropy', ...)  # Expects integers

# FIX: Change to categorical_crossentropy
model.compile(loss='categorical_crossentropy', ...)
```

**Quick Decision Table:**

| Label Format | Loss Function |
|--------------|---------------|
| Integers (0-9) | `sparse_categorical_crossentropy` |
| One-hot encoded | `categorical_crossentropy` |

---

### Error 8: Negative Dimension Size

**Error Message:**
```
ValueError: Negative dimension size caused by subtracting 3 from 1
```

**Cause:** Too many pooling layers, input too small, or wrong architecture.

**Solution:**

Check output dimensions:
```python
model.summary()

# If you see negative or zero dimensions, you've pooled too much!
# Example problem:
# 28×28 → pool → 14×14 → pool → 7×7 → pool → 3×3 → pool → 1×1 → pool → ERROR!

# Fix: Remove one or more pooling layers
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),  # 28×28 → 26×26 → 13×13
    layers.Conv2D(64, (3, 3), activation='relu'),
    # layers.MaxPooling2D((2, 2)),  # REMOVE this if input too small
    layers.Flatten(),
    ...
])
```

**Rule of thumb:** For 28×28 images, maximum 2-3 pooling layers.

---

### Error 9: AttributeError: 'Sequential' object has no attribute 'predict'

**Error Message:**
```
AttributeError: 'Sequential' object has no attribute 'predict'
```

**Cause:** Trying to use model before compiling.

**Solution:**
```python
# WRONG order
model = keras.Sequential([...])
predictions = model.predict(X_test)  # ERROR! Model not compiled

# CORRECT order
model = keras.Sequential([...])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5)  # Train first
predictions = model.predict(X_test)  # Now OK
```

---

## Training Problems

### Error 10: Model Not Learning (Accuracy ~10%)

**Symptoms:**
- Accuracy stuck at ~10% (random guessing for 10 classes)
- Loss not decreasing
- Validation accuracy same as training

**Possible Causes & Fixes:**

**Cause 1: Forgot to normalize data**
```python
# WRONG
X_train = X_train.reshape(-1, 28, 28, 1)  # Still 0-255 range

# CORRECT
X_train = X_train.astype('float32') / 255.0  # Now 0-1 range
X_train = X_train.reshape(-1, 28, 28, 1)
```

**Cause 2: Wrong loss function**
```python
# If using integer labels (0-9):
model.compile(loss='sparse_categorical_crossentropy', ...)

# If using one-hot encoded labels:
model.compile(loss='categorical_crossentropy', ...)

# Check your labels:
print(f"Label shape: {y_train.shape}")
print(f"First label: {y_train[0]}")
```

**Cause 3: Learning rate too high**
```python
# Try lower learning rate
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.0001), ...)  # Default is 0.001
```

**Cause 4: Data not shuffled**
```python
# Shuffle data before training
indices = np.random.permutation(len(X_train))
X_train = X_train[indices]
y_train = y_train[indices]
```

---

### Error 11: Overfitting (Training >> Validation)

**Symptoms:**
- Training accuracy: 98%
- Validation accuracy: 85%
- Large gap between training and validation

**Solutions:**

**Solution 1: Add Dropout**
```python
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  # Add dropout

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  # Add dropout

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),   # Higher dropout before output
    layers.Dense(10, activation='softmax')
])
```

**Solution 2: Data Augmentation**
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

model.fit(datagen.flow(X_train, y_train, batch_size=128),
          epochs=10,
          validation_data=(X_test, y_test))
```

**Solution 3: Early Stopping**
```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(X_train, y_train,
          epochs=20,
          validation_split=0.2,
          callbacks=[early_stop])
```

**Solution 4: Reduce Model Complexity**
```python
# Use fewer filters or layers
model = keras.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # 32→16
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),  # 64→32
    layers.Dense(10, activation='softmax')
])
```

---

### Error 12: Training Too Slow

**Symptoms:**
- Each epoch takes > 5 minutes
- CPU at 100%, no GPU usage

**Solutions:**

**Solution 1: Use GPU (Google Colab)**
```
Runtime → Change runtime type → Hardware accelerator → GPU
```

**Solution 2: Reduce Batch Size (if memory limited)**
```python
# Smaller batch = more steps but less memory
model.fit(X_train, y_train, batch_size=32, ...)  # Instead of 128
```

**Solution 3: Use Smaller Dataset (for testing)**
```python
# Use subset for quick iteration
X_train_small = X_train[:10000]
y_train_small = y_train[:10000]

model.fit(X_train_small, y_train_small, epochs=5, ...)
```

**Solution 4: Reduce Epochs (initial testing)**
```python
# Start with fewer epochs while debugging
model.fit(X_train, y_train, epochs=2, ...)  # Test with 2 epochs first
```

---

## Memory Errors

### Error 13: ResourceExhaustedError: OOM

**Error Message:**
```
ResourceExhaustedError: OOM when allocating tensor with shape [128,64,26,26]
```

**Cause:** Out of GPU/CPU memory.

**Solutions:**

**Solution 1: Reduce Batch Size**
```python
# Try progressively smaller batch sizes
model.fit(X_train, y_train, batch_size=32, ...)   # Instead of 128
model.fit(X_train, y_train, batch_size=16, ...)   # If still failing
```

**Solution 2: Reduce Model Size**
```python
# Use fewer filters
model = keras.Sequential([
    layers.Conv2D(16, (3, 3), ...),  # 32→16
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), ...),  # 64→32
    ...
])
```

**Solution 3: Clear Memory**
```python
# In Jupyter/Colab
import gc
from tensorflow.keras import backend as K

K.clear_session()
gc.collect()
```

**Solution 4: Use Google Colab with GPU**
- More memory available on Colab
- Runtime → Change runtime type → GPU

---

## Visualization Errors

### Error 14: UserWarning: Matplotlib is currently using agg

**Error Message:**
```
UserWarning: Matplotlib is currently using agg, which is a non-GUI backend
```

**Cause:** Running in environment without display (SSH, some IDEs).

**Solution:**

**In Jupyter/Colab:**
```python
%matplotlib inline
import matplotlib.pyplot as plt
```

**In Python script:**
```python
# Save plots instead of showing
plt.plot(history.history['accuracy'])
plt.savefig('accuracy_plot.png')
# Don't call plt.show()
```

---

### Error 15: Feature Maps Not Displaying

**Problem:** Feature map visualization shows black or blank images.

**Cause:** Feature maps not normalized for display.

**Solution:**
```python
# When visualizing feature maps, normalize each one
activation = activations[0]  # Shape: (1, H, W, channels)

for i in range(8):
    channel_image = activation[0, :, :, i]

    # Normalize to [0, 1] range
    channel_image -= channel_image.mean()
    if channel_image.std() > 0:
        channel_image /= channel_image.std()
    channel_image = np.clip(channel_image, 0, 1)

    plt.subplot(2, 4, i+1)
    plt.imshow(channel_image, cmap='viridis')
    plt.axis('off')
plt.show()
```

---

## Google Colab Specific Issues

### Error 16: Session Timeout

**Problem:** Colab session disconnects after 90 minutes of inactivity.

**Solutions:**

**Solution 1: Keep Session Alive (JavaScript)**
```python
# Run this in a Colab cell
from IPython.display import display, HTML

display(HTML("""
<script>
function ClickConnect(){
    console.log("Clicked on connect button");
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect, 60000)
</script>
"""))
```

**Solution 2: Save Model Frequently**
```python
# Save after training
model.save('/content/drive/MyDrive/fashion_mnist_model.h5')

# Load if session disconnects
model = keras.models.load_model('/content/drive/MyDrive/fashion_mnist_model.h5')
```

**Solution 3: Connect Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')

# Save all outputs to Drive
output_dir = '/content/drive/MyDrive/tutorial_t10_outputs/'
os.makedirs(output_dir, exist_ok=True)
```

---

### Error 17: Cannot Find File 'output/...'

**Problem:** FileNotFoundError when saving visualizations.

**Cause:** Output directory doesn't exist.

**Solution:**
```python
import os

# Create output directory
os.makedirs('output', exist_ok=True)

# Then save
plt.savefig('output/accuracy_plot.png')
```

---

## Quick Debugging Checklist

When your code isn't working, check these in order:

1. **Data Shape**
   ```python
   print(f"X_train shape: {X_train.shape}")  # Should be (60000, 28, 28, 1)
   print(f"y_train shape: {y_train.shape}")  # Should be (60000, 10) or (60000,)
   ```

2. **Data Range**
   ```python
   print(f"X_train range: [{X_train.min()}, {X_train.max()}]")  # Should be [0, 1]
   ```

3. **Model Summary**
   ```python
   model.summary()  # Check for negative dimensions or wrong shapes
   ```

4. **Loss Function Matches Labels**
   ```python
   # If y_train.shape = (60000,): use sparse_categorical_crossentropy
   # If y_train.shape = (60000, 10): use categorical_crossentropy
   ```

5. **Try Minimal Model First**
   ```python
   # Simplest possible model to verify setup
   model = keras.Sequential([
       layers.Flatten(input_shape=(28, 28, 1)),
       layers.Dense(10, activation='softmax')
   ])
   ```

---

## Still Having Issues?

### Step-by-Step Diagnostic

1. **Test imports:**
   ```python
   import tensorflow as tf
   print(f"TensorFlow: {tf.__version__}")
   ```

2. **Test data loading:**
   ```python
   (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
   print("✓ Data loaded")
   ```

3. **Test minimal model:**
   ```python
   model = keras.Sequential([layers.Flatten(input_shape=(28, 28, 1)),
                             layers.Dense(10, activation='softmax')])
   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
   print("✓ Model built")
   ```

4. **Test single epoch:**
   ```python
   X_small = X_train[:1000].reshape(-1, 28, 28, 1) / 255.0
   y_small = y_train[:1000]
   model.fit(X_small, y_small, epochs=1, batch_size=32)
   print("✓ Training works")
   ```

If all these pass, your environment is set up correctly!

---

## Contact for Help

If you're stuck after trying these solutions:

1. **During Tutorial:** Raise your hand, instructor will help
2. **Office Hours:** [Insert office hours here]
3. **Course Forum:** [Insert forum link]
4. **Email:** [Insert instructor email]

**When asking for help, include:**
- Full error message (copy-paste)
- Code snippet that causes error
- What you've already tried
- Environment (Colab/local, TensorFlow version)

---

**Remember:** Debugging is part of learning! Every error is an opportunity to understand how things work. 🐛➡️🦋
