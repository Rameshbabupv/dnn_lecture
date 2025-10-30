# CNN Layers & Regularization - Quick Reference Cheat Sheet

**Course:** 21CSE558T - Deep Neural Network Architectures
**Module 4:** CNNs (Week 2 of 3)
**Date:** October 31, 2025

---

## 🎯 Pooling Layers

| Type | Operation | Use Case | Parameters |
|------|-----------|----------|------------|
| **Max Pooling** | Keep maximum value | Classification (most common) | 0 |
| **Average Pooling** | Take average value | Smooth features | 0 |
| **Global Avg Pooling** | Average entire map | Replace Flatten+Dense | 0 |

### Pooling Output Formula
```
output_size = (input_size - pool_size) / stride + 1
```

**Example:** 28×28 → MaxPool(2×2, stride=2) → 14×14

### Code Examples
```python
# Max Pooling (most common)
MaxPooling2D(pool_size=(2, 2), strides=2)

# Global Average Pooling (modern approach)
GlobalAveragePooling2D()  # 7×7×512 → 512 values
```

---

## 🔧 Batch Normalization

### The Problem
**Internal Covariate Shift:** Layer inputs shift during training → Slow, unstable learning

### The Solution
**Batch Normalization:** Normalize inputs to mean=0, std=1

### Formula
1. Normalize: `x_hat = (x - μ) / √(σ² + ε)`
2. Scale & Shift: `y = γ * x_hat + β` (learnable γ, β)

### Correct Placement ✅
```python
Conv2D(32, (3,3))           # No activation
BatchNormalization()         # BEFORE activation
Activation('relu')           # Apply activation last
```

### Benefits
- ✅ 2-3× faster training
- ✅ Acts as regularization
- ✅ Less sensitive to initialization
- ✅ Allows higher learning rates

---

## 💧 Dropout

### How It Works
Randomly deactivate neurons during training (probability p = dropout rate)

### Placement Rules

| Location | Dropout Rate | Rule |
|----------|--------------|------|
| After Conv layers | 0.2 - 0.3 | Optional, light dropout |
| After FC layers | 0.4 - 0.5 | Almost always! |
| Before output | 0.0 | ❌ NEVER! |

### Code Example
```python
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # 50% dropout

model.add(Dense(10, activation='softmax'))
# NO dropout here!
```

---

## 🖼️ Data Augmentation

### Common Techniques

**Geometric:**
- Rotation: ±15° (be careful with 6 vs 9!)
- Horizontal flip: Safe for most objects
- Vertical flip: Rare (satellites, abstracts only)
- Shift: ±10% width/height
- Zoom: 90-110%

**Photometric:**
- Brightness: 80-120%
- Contrast adjustment
- Saturation adjustment

### When to Use
- ✅ Small datasets (<10K images per class)
- ✅ Class imbalance
- ✅ Real-world variations expected
- ❌ Medical imaging (be very careful!)
- ❌ Text/digits (no flips/rotations!)

### Keras Implementation
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

# IMPORTANT: Don't augment validation/test data!
val_datagen = ImageDataGenerator(rescale=1./255)
```

---

## 🏗️ Modern CNN Architecture Template

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation,
    MaxPooling2D, GlobalAveragePooling2D,
    Dropout, Dense
)

model = Sequential([
    # Block 1: 32 filters
    Conv2D(32, (3,3), padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(32, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),
    Dropout(0.2),

    # Block 2: 64 filters (double)
    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    # Block 3: 128 filters (double again)
    Conv2D(128, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    GlobalAveragePooling2D(),  # Modern: Replace Flatten!

    # Output
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

---

## ✅ Regularization Checklist

When building CNNs, include:

- [ ] **Batch Normalization** (after Conv, before activation)
- [ ] **Data Augmentation** (if dataset < 50K)
- [ ] **Dropout** (0.2-0.3 after pooling, 0.5 before output)
- [ ] **Global Average Pooling** (replace Flatten+Dense)
- [ ] **Early Stopping** (monitor val_loss, patience=5)

---

## 📊 Parameter Comparison

| Approach | Parameters | Overfitting Risk |
|----------|------------|------------------|
| **Old:** Flatten + Dense(1024) | ~25M | Very High |
| **Modern:** GlobalAvgPool | ~5K | Low |
| **Reduction:** | **5000×** | ✅ |

---

## 🎓 Filter Progression Rule

**Double filters, halve dimensions:**
```
32×32×32  (32 filters, large spatial)
    ↓ Pool(2×2)
16×16×64  (64 filters, medium spatial)
    ↓ Pool(2×2)
8×8×128   (128 filters, small spatial)
    ↓ Pool(2×2)
4×4×256   (256 filters, tiny spatial)
```

---

## 🚫 Common Mistakes to Avoid

❌ **Dropout after output layer**
```python
Dense(10, activation='softmax')
Dropout(0.5)  # ❌ NO! Makes predictions random!
```

❌ **Augmenting test data**
```python
test_datagen = ImageDataGenerator(
    rotation_range=15,  # ❌ NO! Test should be real images!
)
```

❌ **BatchNorm after activation**
```python
Conv2D(32, activation='relu')  # ❌ Activation too early
BatchNormalization()
```

❌ **Using ALL regularization blindly**
- Start with BatchNorm only
- Add augmentation if still overfitting
- Add dropout if still overfitting
- Don't over-regularize!

---

## 💡 Quick Decision Guide

### Should I use Max or Average Pooling?
→ **Max Pooling** (99% of the time)

### Should I use Global Average Pooling?
→ **Yes**, if you have 256+ channels before it

### Where do I put BatchNormalization?
→ After Conv/Dense, **BEFORE** activation

### What dropout rate should I use?
→ 0.5 for FC layers, 0.2-0.3 for Conv layers

### Should I augment my data?
→ Yes if dataset < 50K images, check domain appropriateness

### What if my model is still overfitting?
→ Check list:
1. BatchNorm ✅?
2. Data augmentation ✅?
3. Dropout ✅?
4. Early stopping ✅?
5. Reduce model size?

---

## 📈 Expected Performance

**Baseline (Week 10):**
- Train: 95%, Test: 60% → Gap: 35% 😱

**Modern (Week 11 with all techniques):**
- Train: 82%, Test: 80% → Gap: 2% 🎉

**Improvement:** +20% test accuracy, 17× less overfitting!

---

## 🔗 Remember the Characters

- **Character: Meera (Photographer):** Max pooling = keep sharpest photos
- **Character: Sneha (Factory Manager):** BatchNorm = quality control checkpoints
- **Character: Ravi (Cricket Coach):** Dropout = random player absence training
- **Character: Priya (Student):** Understanding > Memorization

---

**Print this cheat sheet and bring to Tutorial T11 (Monday, Nov 3)!**

**Next:** Implement all techniques on CIFAR-10 dataset 🚀
