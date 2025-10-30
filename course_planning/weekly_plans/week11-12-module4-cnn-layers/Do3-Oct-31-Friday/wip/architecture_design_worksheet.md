# CNN Architecture Design Worksheet

**Course:** 21CSE558T - Deep Neural Network Architectures
**Module 4:** CNNs (Week 2 of 3)
**Date:** October 31, 2025
**Name:** _________________________
**Roll No:** _________________________

---

## Instructions

This worksheet helps you practice designing CNN architectures with proper regularization techniques. Work through each problem, showing calculations and design decisions.

---

## Problem 1: Calculate Pooling Output Dimensions (10 marks)

Calculate the output dimensions after pooling operations.

**Formula:** `output_size = (input_size - pool_size) / stride + 1`

### Q1.1 (2 marks)
Input: 64×64 image
Pooling: MaxPooling2D(pool_size=2, stride=2)

**Calculate:**
- Output Height: ________________
- Output Width: ________________
- Show calculation:

```




```

### Q1.2 (2 marks)
Input: 224×224 image
Pooling: MaxPooling2D(pool_size=3, stride=2)

**Calculate:**
- Output Height: ________________
- Output Width: ________________
- Show calculation:

```




```

### Q1.3 (3 marks)
A CNN has the following sequence:
1. Input: 32×32×3
2. Conv2D(32 filters, 3×3, padding='same')
3. MaxPooling2D(2×2, stride=2)
4. Conv2D(64 filters, 3×3, padding='same')
5. MaxPooling2D(2×2, stride=2)

**Calculate dimensions after each step:**
- After Conv2D (step 2): ________________
- After MaxPool (step 3): ________________
- After Conv2D (step 4): ________________
- After MaxPool (step 5): ________________

### Q1.4 (3 marks)
Compare parameter reduction:

**Scenario A (Old approach):**
- Last conv output: 7×7×512
- Flatten → Dense(4096) → Dense(1000)

**Scenario B (Modern approach):**
- Last conv output: 7×7×512
- GlobalAveragePooling2D() → Dense(1000)

**Calculate:**
- Scenario A parameters: ________________
- Scenario B parameters: ________________
- Parameter reduction: ______× fewer

---

## Problem 2: Batch Normalization Placement (10 marks)

### Q2.1 (5 marks)
**Fix the incorrect architecture:**

```python
# ❌ INCORRECT CODE - Fix the BatchNorm placement!
model = Sequential([
    Conv2D(32, (3,3), activation='relu'),  # Line 1
    BatchNormalization(),                   # Line 2
    MaxPooling2D((2,2)),                    # Line 3

    Conv2D(64, (3,3), activation='relu'),  # Line 4
    BatchNormalization(),                   # Line 5
    MaxPooling2D((2,2)),                    # Line 6
])
```

**Write the CORRECTED code below:**

```python
model = Sequential([
    # YOUR CORRECTED CODE HERE








])
```

**Explain what was wrong (2 sentences):**

```




```

### Q2.2 (5 marks)
**Multiple Choice:** Circle ALL correct statements about Batch Normalization:

a) BatchNormalization should be placed AFTER the activation function
b) BatchNormalization normalizes inputs to mean=0, std=1
c) BatchNormalization has 2 learnable parameters per feature: γ (scale) and β (shift)
d) BatchNormalization slows down training
e) BatchNormalization acts as a regularization technique
f) BatchNormalization requires batch size ≥ 16 for reliable statistics
g) BatchNormalization should be used after the output layer

**Number of correct answers:** ________

---

## Problem 3: Dropout Strategy Design (10 marks)

### Q3.1 (6 marks)
Design dropout strategy for this architecture:

```python
model = Sequential([
    Conv2D(32, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(32, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),
    # Dropout here? Rate: ________

    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),
    # Dropout here? Rate: ________

    Conv2D(128, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    GlobalAveragePooling2D(),
    # Dropout here? Rate: ________

    Dense(10, activation='softmax')
    # Dropout here? Rate: ________
])
```

**Fill in dropout rates (or write "NO" if dropout not needed):**

**Explain your reasoning for each decision:**

```
After first MaxPool:


After second MaxPool:


After GlobalAveragePooling:


After Dense output:


```

### Q3.2 (4 marks)
**Scenario:** You trained a model with the following results:
- Training accuracy: 98%
- Validation accuracy: 65%
- You have Dropout(0.3) after FC layers

**What should you do?** (Circle one and explain)

a) Increase dropout rate to 0.5-0.6
b) Decrease dropout rate to 0.1-0.2
c) Remove dropout completely
d) Add more dropout layers after Conv layers

**Explanation:**

```





```

---

## Problem 4: Data Augmentation Selection (15 marks)

### Q4.1 (9 marks)
For each dataset, mark which augmentations are **appropriate** (✓) or **inappropriate** (✗):

| Augmentation | Cat/Dog Classification | Medical X-rays | Handwritten Digits (0-9) |
|--------------|------------------------|----------------|--------------------------|
| Rotation ±15° | ______ | ______ | ______ |
| Rotation ±180° | ______ | ______ | ______ |
| Horizontal Flip | ______ | ______ | ______ |
| Vertical Flip | ______ | ______ | ______ |
| Zoom 90-110% | ______ | ______ | ______ |
| Brightness ±20% | ______ | ______ | ______ |
| Color Jitter | ______ | ______ | ______ |

### Q4.2 (6 marks)
**Write Keras ImageDataGenerator code for each scenario:**

**Scenario A: Natural images (cats, dogs, birds)**
```python
train_datagen = ImageDataGenerator(
    # YOUR CODE HERE




)
```

**Scenario B: Handwritten digits (MNIST)**
```python
train_datagen = ImageDataGenerator(
    # YOUR CODE HERE



)
```

**Explain the difference in your choices:**

```




```

---

## Problem 5: Complete Architecture Design (25 marks)

### Q5.1 (15 marks)
**Design a complete modern CNN for CIFAR-10 classification:**

**Requirements:**
- Input: 32×32×3 images
- Output: 10 classes
- Use: Batch Normalization, Dropout, Global Average Pooling
- Follow filter progression: 32 → 64 → 128
- Include 3 conv blocks with pooling

**Write complete Keras code:**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation,
    MaxPooling2D, GlobalAveragePooling2D,
    Dropout, Dense
)

model = Sequential([
    # YOUR COMPLETE ARCHITECTURE HERE





















])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Q5.2 (5 marks)
**Justify your architectural choices:**

**Why did you place BatchNorm where you did?**
```


```

**Why did you choose those dropout rates?**
```


```

**Why use Global Average Pooling instead of Flatten + Dense?**
```


```

### Q5.3 (5 marks)
**Calculate total parameters in your architecture:**

Use `model.summary()` or calculate manually:

**Total parameters:** ________________

**Show calculation for at least 2 layers:**

```





```

---

## Problem 6: Debugging Exercise (15 marks)

### Q6.1 (7 marks)
**This CNN has overfitting issues. Identify problems and fix them:**

```python
# Training results: Train Acc: 99%, Val Acc: 55% (SEVERE OVERFITTING!)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(1024, activation='relu'),  # Very large!
    Dense(512, activation='relu'),   # Another large one!
    Dense(10, activation='softmax')
])
```

**List 5 problems:**
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________

**Write CORRECTED architecture:**

```python
model = Sequential([
    # YOUR CORRECTED CODE HERE











])
```

### Q6.2 (8 marks)
**Scenario:** After applying your fixes, you get:
- Training accuracy: 75%
- Validation accuracy: 73%
- Training is very slow (10 min/epoch)

**Diagnose and propose solutions:**

**Is this overfitting or underfitting?** _______________

**Why is training so slow?**
```


```

**Propose 3 solutions to speed up training:**
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________

---

## Problem 7: Architecture Comparison (15 marks)

### Q7.1 (10 marks)
Compare these two architectures for CIFAR-10 classification:

**Architecture A (Old Style):**
```
Conv(32) → ReLU → Pool
Conv(64) → ReLU → Pool
Flatten → Dense(512) → ReLU → Dense(10)
```

**Architecture B (Modern Style):**
```
Conv(32) → BN → ReLU → Conv(32) → BN → ReLU → Pool → Dropout(0.2)
Conv(64) → BN → ReLU → Conv(64) → BN → ReLU → Pool → Dropout(0.3)
Conv(128) → BN → ReLU → GlobalAvgPool → Dropout(0.5) → Dense(10)
```

**Fill in the comparison table:**

| Aspect | Architecture A | Architecture B | Winner? |
|--------|----------------|----------------|---------|
| Total Parameters | _________ | _________ | _______ |
| Overfitting Risk | (High/Low) | (High/Low) | _______ |
| Training Speed | (Slow/Fast) | (Slow/Fast) | _______ |
| Expected Test Acc | ~___% | ~___% | _______ |
| Regularization | (Yes/No) | (Yes/No) | _______ |

**Which architecture would you choose and why?**
```




```

### Q7.2 (5 marks)
**Predict training curves:**

Sketch what you expect for training/validation accuracy over 50 epochs:

**Architecture A:**
```
Epoch 1:  Train: ___%, Val: ___%
Epoch 10: Train: ___%, Val: ___%
Epoch 50: Train: ___%, Val: ___%
```

**Architecture B:**
```
Epoch 1:  Train: ___%, Val: ___%
Epoch 10: Train: ___%, Val: ___%
Epoch 50: Train: ___%, Val: ___%
```

**Explain the difference:**
```



```

---

## Bonus Problem (5 bonus marks)

### Challenge: Design Ultra-Efficient CNN

**Constraint:** Maximum 100K parameters
**Task:** Classify CIFAR-10 (10 classes)
**Goal:** Highest possible accuracy with parameter budget

**Your architecture:**

```python
model = Sequential([
    # YOUR ULTRA-EFFICIENT DESIGN HERE









])
```

**Estimated parameters:** ________________

**Techniques used to reduce parameters:**
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________

---

## Submission Checklist

Before submitting, ensure:
- [ ] All calculations shown with work
- [ ] Code is properly formatted and commented
- [ ] All questions answered
- [ ] Explanations are clear and concise
- [ ] Name and roll number at top

---

## Answer Key (For Instructor)

**Problem 1:**
- Q1.1: 32×32
- Q1.2: 111×111
- Q1.3: 32×32×32 → 16×16×32 → 16×16×64 → 8×8×64
- Q1.4: Scenario A: 102M+ params, Scenario B: 512K params, ~200× reduction

**Problem 2:**
- Q2.1: Move BatchNorm before activation
- Q2.2: Correct: b, c, e, f (4 answers)

**Problem 3:**
- Q3.1: 0.2, 0.3, 0.5, NO
- Q3.2: (a) Increase dropout - severe overfitting

**Problem 4:**
- Q4.1: Refer to lecture notes for domain-specific appropriateness
- Q4.2: Natural images: all augmentations; Digits: minimal (rotation ±10°, shift, zoom only)

**Problem 5-7:** Subjective, grade based on understanding and proper technique application

---

**Total: 100 marks (+5 bonus)**

**Due Date:** Before Tutorial T11 (Monday, November 3, 2025)
