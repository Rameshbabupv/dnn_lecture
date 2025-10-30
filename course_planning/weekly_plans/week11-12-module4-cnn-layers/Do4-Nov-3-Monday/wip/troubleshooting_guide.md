# Tutorial T11 - Troubleshooting Guide

**Course:** 21CSE558T - Deep Neural Network Architectures
**Tutorial:** T11 - CIFAR-10 with Modern CNN
**Date:** November 3, 2025

---

## 🔧 Common Issues and Solutions

This guide covers the most common problems students encounter during Tutorial T11 and how to fix them.

---

## 1. Dataset Loading Issues

### Problem 1.1: CIFAR-10 Download Fails

**Error:**
```
Exception: URL fetch failure on https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

**Causes:**
- Network connection issues
- Firewall blocking download
- Corrupted cached file

**Solutions:**
```python
# Solution 1: Delete cache and retry
import os
cache_dir = os.path.expanduser('~/.keras/datasets')
cifar_file = os.path.join(cache_dir, 'cifar-10-batches-py.tar.gz')
if os.path.exists(cifar_file):
    os.remove(cifar_file)
    print("Cache deleted. Retry loading...")

# Solution 2: Manual download
# Download from: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# Place in ~/.keras/datasets/
# Then reload
```

---

### Problem 1.2: "tuple" object has no attribute 'shape'

**Error:**
```
AttributeError: 'tuple' object has no attribute 'shape'
```

**Cause:**
```python
# Wrong - assigning tuple to single variable
(x_train, y_train), (x_test, y_test) = None
```

**Solution:**
```python
# Correct - unpacking properly
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

---

## 2. Preprocessing Issues

### Problem 2.1: Images Not Normalized

**Symptom:**
- Model loss is NaN
- Training doesn't converge
- Accuracy stays at 10% (random)

**Cause:**
```python
# Wrong - forgot to normalize
x_train_norm = x_train  # Still in range [0, 255]
```

**Solution:**
```python
# Correct - normalize to [0, 1]
x_train_norm = x_train.astype('float32') / 255.0
x_test_norm = x_test.astype('float32') / 255.0
```

**Verification:**
```python
print(f"Min: {x_train_norm.min()}, Max: {x_train_norm.max()}")
# Should print: Min: 0.0, Max: 1.0
```

---

### Problem 2.2: Label Encoding Mismatch

**Error:**
```
ValueError: Shapes (None, 10) and (None, 1) are incompatible
```

**Cause:**
```python
# Wrong - labels still in integer format
y_train_cat = y_train  # Shape: (50000, 1)
```

**Solution:**
```python
# Correct - one-hot encode
y_train_cat = to_categorical(y_train, 10)  # Shape: (50000, 10)
```

**Verification:**
```python
print(f"Label shape: {y_train_cat.shape}")  # Should be (50000, 10)
print(f"Sample label: {y_train_cat[0]}")    # Should be one-hot vector
```

---

## 3. Architecture Issues

### Problem 3.1: BatchNorm Placement Wrong

**Wrong Pattern:**
```python
# ❌ INCORRECT - Activation before BatchNorm
Conv2D(32, (3,3), activation='relu')
BatchNormalization()
```

**Correct Pattern:**
```python
# ✅ CORRECT - BatchNorm before activation
Conv2D(32, (3,3))  # No activation!
BatchNormalization()
Activation('relu')
```

**Why it matters:**
- BatchNorm normalizes pre-activation values
- Placing after activation reduces effectiveness
- Standard modern practice: Conv → BN → Activation

---

### Problem 3.2: Dropout After Output Layer

**Error Pattern:**
```python
# ❌ INCORRECT - Dropout after output
Dense(10, activation='softmax')
Dropout(0.5)  # This makes predictions random!
```

**Correct Pattern:**
```python
# ✅ CORRECT - Dropout before output
Dropout(0.5)
Dense(10, activation='softmax')  # No dropout after!
```

**Why it's wrong:**
- Dropout after softmax makes final predictions random
- Test-time predictions become unreliable
- NEVER use dropout after the output layer

---

### Problem 3.3: Forgot input_shape

**Error:**
```
ValueError: Input 0 of layer sequential is incompatible with the layer
```

**Cause:**
```python
# Wrong - forgot input_shape on first layer
Conv2D(32, (3,3), padding='same')
```

**Solution:**
```python
# Correct - specify input_shape
Conv2D(32, (3,3), padding='same', input_shape=(32, 32, 3))
```

---

### Problem 3.4: GlobalAveragePooling2D Misunderstanding

**Common Confusion:**
"My model has fewer parameters but worse accuracy!"

**Understanding:**
```python
# Old way (more parameters)
Flatten()  # 8×8×128 = 8,192 values
Dense(128)  # 8,192 × 128 = 1,048,576 parameters!

# Modern way (fewer parameters)
GlobalAveragePooling2D()  # 128 values (average each channel)
Dense(128)  # 128 × 128 = 16,384 parameters (64× fewer!)
```

**Why it works:**
- Forces each filter to learn one concept
- Acts as structural regularization
- May need more filters before GAP (use 256-512 instead of 128)

**If accuracy drops with GAP:**
```python
# Increase filters in last conv layer
Conv2D(256, (3,3), padding='same')  # Instead of 128
BatchNormalization()
Activation('relu')
GlobalAveragePooling2D()
```

---

## 4. Training Issues

### Problem 4.1: Model Not Learning (Loss = NaN)

**Symptoms:**
- Loss becomes NaN after a few epochs
- Accuracy stays at ~10% (random guessing)

**Possible Causes & Solutions:**

**Cause 1: Learning rate too high**
```python
# Solution: Reduce learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**Cause 2: Data not normalized**
```python
# Solution: Verify normalization
print(f"Data range: [{x_train.min()}, {x_train.max()}]")
# Should be [0.0, 1.0], not [0, 255]
```

**Cause 3: Exploding gradients**
```python
# Solution: Add gradient clipping
model.compile(
    optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

### Problem 4.2: Training Extremely Slow

**Symptom:**
- Training takes 5+ minutes per epoch on CPU
- Progress bar barely moves

**Solutions:**

**Solution 1: Reduce batch size**
```python
# If running out of memory
BATCH_SIZE = 32  # Instead of 128
```

**Solution 2: Use fewer augmentation steps**
```python
# Reduce steps_per_epoch for faster iteration
history = model.fit(
    train_datagen.flow(x_train, y_train, batch_size=64),
    steps_per_epoch=len(x_train) // 64 // 2,  # Half steps for speed
    ...
)
```

**Solution 3: Disable augmentation for testing**
```python
# For quick testing, train without augmentation
history = model.fit(
    x_train, y_train,  # Direct, no datagen
    validation_data=(x_val, y_val),
    epochs=5,  # Fewer epochs for testing
    batch_size=64
)
```

**Solution 4: Use GPU (if available)**
```python
# Check GPU availability
print(tf.config.list_physical_devices('GPU'))

# If available, TensorFlow uses it automatically
# If not, consider Google Colab with GPU runtime
```

---

### Problem 4.3: Out of Memory (OOM) Error

**Error:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**

**Solution 1: Reduce batch size**
```python
BATCH_SIZE = 16  # Instead of 64 or 128
```

**Solution 2: Use mixed precision (if GPU)**
```python
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

**Solution 3: Clear session between runs**
```python
import tensorflow as tf
tf.keras.backend.clear_session()
```

**Solution 4: Reduce model size**
```python
# Use fewer filters
Conv2D(16, (3,3), ...)  # Instead of 32
Conv2D(32, (3,3), ...)  # Instead of 64
```

---

## 5. Data Augmentation Issues

### Problem 5.1: Augmentation Not Applied

**Symptom:**
- Training with augmentation same speed as without
- No improvement in generalization

**Cause:**
```python
# Wrong - forgot to use datagen.flow()
history = model.fit(
    x_train, y_train,  # Direct data, no augmentation!
    ...
)
```

**Solution:**
```python
# Correct - use datagen.flow()
history = model.fit(
    train_datagen.flow(x_train, y_train, batch_size=64),
    steps_per_epoch=len(x_train) // 64,
    validation_data=(x_val, y_val),  # No augmentation for validation!
    epochs=30
)
```

---

### Problem 5.2: Augmenting Validation/Test Data

**Wrong:**
```python
# ❌ DON'T augment validation/test data!
val_datagen = ImageDataGenerator(rotation_range=15, ...)
validation_data = val_datagen.flow(x_val, y_val, batch_size=64)
```

**Correct:**
```python
# ✅ Only augment training data
train_datagen = ImageDataGenerator(rotation_range=15, ...)
# Validation uses original images
validation_data = (x_val, y_val)
```

**Why:**
- Validation/test should reflect real-world (no augmentation)
- Augmentation is for training robustness only
- Augmenting test data gives misleading metrics

---

### Problem 5.3: Inappropriate Augmentation for CIFAR-10

**Bad Augmentation:**
```python
# ❌ Inappropriate for CIFAR-10
ImageDataGenerator(
    rotation_range=180,    # ❌ Planes don't fly upside down!
    vertical_flip=True,    # ❌ Inverts meaning (plane→boat?)
    ...
)
```

**Good Augmentation:**
```python
# ✅ Appropriate for CIFAR-10
ImageDataGenerator(
    rotation_range=15,       # ✅ Slight angle variations
    width_shift_range=0.1,   # ✅ Position variations
    height_shift_range=0.1,  # ✅ Position variations
    horizontal_flip=True,    # ✅ Left/right mirror OK
    zoom_range=0.1           # ✅ Distance variations
)
```

---

## 6. Evaluation Issues

### Problem 6.1: Test Accuracy Much Lower Than Expected

**Possible Causes:**

**Cause 1: Forgot to normalize test data**
```python
# Wrong
test_loss, test_acc = model.evaluate(x_test, y_test_cat)  # Not normalized!

# Correct
test_loss, test_acc = model.evaluate(x_test_norm, y_test_cat)
```

**Cause 2: Used wrong labels**
```python
# Wrong - labels not one-hot encoded
test_loss, test_acc = model.evaluate(x_test_norm, y_test)  # Shape mismatch!

# Correct
test_loss, test_acc = model.evaluate(x_test_norm, y_test_cat)
```

---

### Problem 6.2: Confusion Matrix Error

**Error:**
```
ValueError: Found input variables with inconsistent numbers of samples
```

**Cause:**
```python
# Wrong - y_test is one-hot, need class indices
y_pred_classes = model.predict(x_test).argmax(axis=1)
cm = confusion_matrix(y_test, y_pred_classes)  # Shape mismatch!
```

**Solution:**
```python
# Correct - convert one-hot to class indices
y_pred_classes = model.predict(x_test).argmax(axis=1)
y_true_classes = y_test.argmax(axis=1)  # Or y_test.flatten() if not one-hot
cm = confusion_matrix(y_true_classes, y_pred_classes)
```

---

## 7. Performance Issues

### Problem 7.1: Still Overfitting Despite Regularization

**Symptoms:**
- Train accuracy: 90%+
- Val accuracy: 60-70%
- Gap > 15%

**Solutions (in order of priority):**

**1. Increase dropout rates**
```python
Dropout(0.3)  # Instead of 0.2
Dropout(0.4)  # Instead of 0.3
Dropout(0.6)  # Instead of 0.5
```

**2. Add more augmentation**
```python
train_datagen = ImageDataGenerator(
    rotation_range=20,  # Increase from 15
    width_shift_range=0.15,  # Increase from 0.1
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.15,
    shear_range=0.1  # Add shear
)
```

**3. Add L2 regularization**
```python
from tensorflow.keras.regularizers import l2

Conv2D(32, (3,3), padding='same', kernel_regularizer=l2(0.001))
```

**4. Reduce model capacity**
```python
# Use fewer filters
Conv2D(16, ...)  # Instead of 32
Conv2D(32, ...)  # Instead of 64
Conv2D(64, ...)  # Instead of 128
```

**5. Early stopping with lower patience**
```python
early_stop = EarlyStopping(monitor='val_loss', patience=3)  # Instead of 5
```

---

### Problem 7.2: Underfitting (Both Train and Val Accuracy Low)

**Symptoms:**
- Train accuracy: 60-70%
- Val accuracy: 55-65%
- Both plateaued, not improving

**Solutions:**

**1. Reduce regularization**
```python
Dropout(0.1)  # Instead of 0.2
Dropout(0.2)  # Instead of 0.3
Dropout(0.3)  # Instead of 0.5
```

**2. Increase model capacity**
```python
# Use more filters
Conv2D(64, ...)  # Instead of 32
Conv2D(128, ...)  # Instead of 64
Conv2D(256, ...)  # Instead of 128
```

**3. Add more conv layers**
```python
# Add another block
Conv2D(256, (3,3), padding='same')
BatchNormalization()
Activation('relu')
Conv2D(256, (3,3), padding='same')
BatchNormalization()
Activation('relu')
MaxPooling2D((2,2))
Dropout(0.4)
```

**4. Train longer**
```python
epochs=50  # Instead of 30
```

**5. Increase learning rate**
```python
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)  # Instead of default
```

---

## 8. Code Debugging Checklist

When your code doesn't work, check these in order:

### ✅ Data Loading
- [ ] CIFAR-10 loaded successfully
- [ ] Data shapes correct: (50000, 32, 32, 3) for training
- [ ] Labels shape: (50000, 10) after one-hot encoding

### ✅ Preprocessing
- [ ] Images normalized to [0, 1]
- [ ] Labels one-hot encoded
- [ ] Validation split created correctly

### ✅ Architecture
- [ ] First layer has input_shape=(32, 32, 3)
- [ ] BatchNorm BEFORE activation
- [ ] Dropout rates: 0.2 → 0.3 → 0.5
- [ ] GlobalAveragePooling2D (not Flatten)
- [ ] NO dropout after output layer
- [ ] Output layer: Dense(10, activation='softmax')

### ✅ Compilation
- [ ] optimizer='adam' (or configured optimizer)
- [ ] loss='categorical_crossentropy'
- [ ] metrics=['accuracy']

### ✅ Training
- [ ] Using train_datagen.flow() for augmentation
- [ ] Validation data NOT augmented
- [ ] Early stopping callback added
- [ ] Batch size reasonable (32-128)

### ✅ Evaluation
- [ ] Test data normalized same as training
- [ ] Test labels one-hot encoded
- [ ] model.evaluate() called correctly

---

## 9. Quick Fixes

### 🔴 Model not learning at all (accuracy ~10%)
```python
# Most likely: data not normalized
x_train_norm = x_train.astype('float32') / 255.0
```

### 🔴 Loss becomes NaN
```python
# Most likely: learning rate too high
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), ...)
```

### 🔴 Out of memory
```python
# Reduce batch size
BATCH_SIZE = 16
```

### 🔴 Training too slow
```python
# Disable augmentation for testing
model.fit(x_train, y_train, ...)  # No datagen
```

### 🔴 Severe overfitting
```python
# Increase dropout
Dropout(0.6)  # Instead of 0.5
```

### 🔴 Underfitting
```python
# Reduce dropout or increase model size
Dropout(0.2)  # Instead of 0.5
# OR
Conv2D(128, ...)  # Instead of 64
```

---

## 10. Getting Help

### Before Asking for Help:
1. Read error message carefully
2. Check this troubleshooting guide
3. Verify data shapes with print statements
4. Try running solution code to compare

### When Asking for Help, Provide:
- Complete error message
- Relevant code snippet
- Data shapes (print statements)
- What you've tried already

### Resources:
- TensorFlow documentation: https://www.tensorflow.org/api_docs
- Keras documentation: https://keras.io/
- Stack Overflow: https://stackoverflow.com/questions/tagged/tensorflow
- Course discussion forum

---

**Last Updated:** November 3, 2025
**Tutorial:** T11 - CIFAR-10 with Modern CNN

**Good luck with your tutorial! 🚀**
