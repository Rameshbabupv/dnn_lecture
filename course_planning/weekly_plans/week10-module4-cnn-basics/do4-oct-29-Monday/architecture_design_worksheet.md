# CNN Architecture Design Worksheet

**Tutorial T10 - Pre-Coding Exercise**
**Week 10, Day 4 - October 29, 2025**

---

## Purpose

This worksheet helps you **plan before coding**. Designing your CNN architecture on paper first helps you:
- Understand output dimensions at each layer
- Count parameters before building
- Make informed architectural decisions
- Catch errors before they become code bugs

**Complete this worksheet BEFORE writing code in Tutorial T10.**

---

## Problem Statement

**Dataset:** Fashion-MNIST
- **Input:** 28×28 grayscale images (28×28×1)
- **Output:** 10 classes (clothing categories)
- **Goal:** Classify images with >88% accuracy

---

## Part 1: Architecture Planning

Fill in the table below to design your CNN architecture:

### Architecture Table

| Layer # | Layer Type | Parameters | Input Shape | Output Shape | # Params | Calculation |
|---------|------------|------------|-------------|--------------|----------|-------------|
| 0 | Input | - | - | 28×28×1 | 0 | - |
| 1 | Conv2D | filters=32, kernel=(3,3), activation='relu' | 28×28×1 | ___×___×___ | ______ | |
| 2 | MaxPooling2D | pool_size=(2,2) | ___×___×___ | ___×___×___ | 0 | |
| 3 | Conv2D | filters=64, kernel=(3,3), activation='relu' | ___×___×___ | ___×___×___ | ______ | |
| 4 | MaxPooling2D | pool_size=(2,2) | ___×___×___ | ___×___×___ | 0 | |
| 5 | Flatten | - | ___×___×___ | _______ | 0 | |
| 6 | Dense | units=64, activation='relu' | _______ | 64 | ______ | |
| 7 | Dense | units=10, activation='softmax' | 64 | 10 | ______ | |

**Total Parameters:** ______________

---

## Part 2: Output Dimension Calculations

Show your work for calculating output dimensions:

### Layer 1: Conv2D (32 filters, 3×3 kernel)

**Formula:** `output_size = (input_size - kernel_size + 2×padding) / stride + 1`

**Given:**
- Input: 28×28×1
- Kernel: 3×3
- Stride: 1 (default)
- Padding: 0 ('valid' padding)

**Calculation:**
```
Height: (28 - 3 + 0) / 1 + 1 = _____
Width:  (28 - 3 + 0) / 1 + 1 = _____
Channels: 32 (number of filters)

Output Shape: _____×_____×_____
```

---

### Layer 2: MaxPooling2D (2×2 pool)

**Formula:** `output_size = input_size / pool_size`

**Given:**
- Input: ___×___×32 (from Layer 1)
- Pool size: 2×2

**Calculation:**
```
Height: _____ / 2 = _____
Width:  _____ / 2 = _____
Channels: 32 (unchanged)

Output Shape: _____×_____×_____
```

---

### Layer 3: Conv2D (64 filters, 3×3 kernel)

**Given:**
- Input: ___×___×32 (from Layer 2)
- Kernel: 3×3
- Stride: 1
- Padding: 0

**Calculation:**
```
Height: (_____ - 3 + 0) / 1 + 1 = _____
Width:  (_____ - 3 + 0) / 1 + 1 = _____
Channels: 64

Output Shape: _____×_____×_____
```

---

### Layer 4: MaxPooling2D (2×2 pool)

**Given:**
- Input: ___×___×64 (from Layer 3)
- Pool size: 2×2

**Calculation:**
```
Height: _____ / 2 = _____
Width:  _____ / 2 = _____
Channels: 64

Output Shape: _____×_____×_____
```

---

### Layer 5: Flatten

**Given:**
- Input: ___×___×64 (from Layer 4)

**Calculation:**
```
Total features = height × width × channels
              = _____ × _____ × 64
              = _______

Output Shape: (_______)
```

---

## Part 3: Parameter Counting

Show your work for calculating parameters in each layer:

### Layer 1: Conv2D (32 filters, 3×3, input_channels=1)

**Formula:** `params = (kernel_h × kernel_w × input_channels + 1) × num_filters`

**Calculation:**
```
Parameters = (3 × 3 × 1 + 1) × 32
           = (9 + 1) × 32
           = 10 × 32
           = ______

Breakdown:
  - Weights: 3 × 3 × 1 × 32 = _______
  - Biases: 32 (one per filter) = _______
  - Total: _______
```

---

### Layer 3: Conv2D (64 filters, 3×3, input_channels=32)

**Calculation:**
```
Parameters = (3 × 3 × 32 + 1) × 64
           = (288 + 1) × 64
           = 289 × 64
           = ________

Breakdown:
  - Weights: 3 × 3 × 32 × 64 = _______
  - Biases: 64 = _______
  - Total: _______
```

---

### Layer 6: Dense (64 units, input_features=______)

**Formula:** `params = (input_features + 1) × output_units`

**Calculation:**
```
Parameters = (______ + 1) × 64
           = ______ × 64
           = ________

Breakdown:
  - Weights: ______ × 64 = _______
  - Biases: 64 = _______
  - Total: _______
```

---

### Layer 7: Dense (10 units, input_features=64)

**Calculation:**
```
Parameters = (64 + 1) × 10
           = 65 × 10
           = ______

Breakdown:
  - Weights: 64 × 10 = _______
  - Biases: 10 = _______
  - Total: _______
```

---

### Total Parameter Count

```
Layer 1 (Conv2D):      ______
Layer 2 (MaxPool):     0
Layer 3 (Conv2D):      ______
Layer 4 (MaxPool):     0
Layer 5 (Flatten):     0
Layer 6 (Dense):       ______
Layer 7 (Dense):       ______

TOTAL PARAMETERS:      ________
```

---

## Part 4: Design Justification

Answer the following questions about your architectural choices:

### Question 1: Why Two Convolutional Blocks?
```
Your answer (2-3 sentences):
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________
```

### Question 2: Why 32 and 64 Filters?
```
Why not 16 and 32? Why not 64 and 128?
Your answer (2-3 sentences):
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________
```

### Question 3: Why 3×3 Kernels?
```
Why not 5×5 or 7×7?
Your answer (2-3 sentences):
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________
```

### Question 4: Why MaxPooling After Each Conv Layer?
```
What would happen without pooling?
Your answer (2-3 sentences):
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________
```

### Question 5: Why Dense Layer with 64 Units?
```
Why not 128 or 256?
Your answer (2-3 sentences):
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________
```

---

## Part 5: Comparison with MLP

**Hypothetical MLP Architecture:**
```
Input: 28×28×1 → Flatten: 784
Dense(128, relu)
Dense(10, softmax)
```

**Calculate MLP Parameters:**
```
Layer 1: (784 + 1) × 128 = _______
Layer 2: (128 + 1) × 10  = _______
Total:                    = _______
```

**Comparison:**
```
CNN Parameters:     _______ (from Part 3)
MLP Parameters:     _______ (from above)

Difference:         _______ parameters

CNN is _____ times more/less parameter-efficient than MLP.
```

**Why is CNN more efficient?**
```
Your answer (3-4 sentences):
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________
```

---

## Part 6: Expected Performance

Based on your design, estimate:

### Training Time
```
Estimated time per epoch (CPU): _______ seconds
Estimated total training time (10 epochs): _______ minutes
```

### Expected Accuracy
```
Expected training accuracy after 10 epochs: _____%
Expected validation accuracy: _____%
Expected test accuracy: _____%

Reasoning for your estimates:
_______________________________________________________________
_______________________________________________________________
```

### Potential Issues
```
List 2-3 potential problems with this architecture:
1. _____________________________________________________________
2. _____________________________________________________________
3. _____________________________________________________________
```

---

## Part 7: Alternative Design (Optional Challenge)

Design an **alternative architecture** that might perform better:

| Layer # | Layer Type | Parameters | Output Shape |
|---------|------------|------------|--------------|
| 0 | Input | - | 28×28×1 |
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |

**What improvements did you make?**
```
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________
```

**Why might this perform better?**
```
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________
```

---

## Part 8: Verification Checklist

Before coding, verify:

- [ ] All output shapes calculated
- [ ] All parameter counts calculated
- [ ] Total parameters computed
- [ ] Output dimensions decrease progressively (28→26→13→11→5)
- [ ] Number of channels increases (1→32→64)
- [ ] Final flatten layer matches dense input
- [ ] Output layer has 10 units (for 10 classes)
- [ ] Activation functions specified
- [ ] Design choices justified

---

## Part 9: Translate to Code

Now that your design is complete, translate to Keras code:

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # Layer 1: Conv2D
    layers.Conv2D(
        filters=______,
        kernel_size=(_____, _____),
        activation='_____',
        input_shape=(_____, _____, _____)
    ),

    # Layer 2: MaxPooling2D
    layers.MaxPooling2D(
        pool_size=(_____, _____)
    ),

    # Layer 3: Conv2D
    layers.Conv2D(
        filters=______,
        kernel_size=(_____, _____),
        activation='_____'
    ),

    # Layer 4: MaxPooling2D
    layers.MaxPooling2D(
        pool_size=(_____, _____)
    ),

    # Layer 5: Flatten
    layers.Flatten(),

    # Layer 6: Dense
    layers.Dense(
        units=______,
        activation='_____'
    ),

    # Layer 7: Output Dense
    layers.Dense(
        units=______,
        activation='_____'
    )
])

# Verify your calculations
model.summary()
```

**Check:** Does `model.summary()` match your calculations?
- [ ] Yes, all shapes match
- [ ] No, found discrepancies (explain): ___________________________

---

## Part 10: Reflection

### What did you learn from this exercise?
```
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________
```

### What was most challenging?
```
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________
```

### How will this planning help during coding?
```
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________
```

---

## Answer Key (For Self-Check After Completing)

<details>
<summary>Click to reveal answers (only after attempting!)</summary>

### Output Shapes:
- Layer 1: 26×26×32
- Layer 2: 13×13×32
- Layer 3: 11×11×64
- Layer 4: 5×5×64
- Layer 5: 1600
- Layer 6: 64
- Layer 7: 10

### Parameters:
- Layer 1: 320
- Layer 3: 18,496
- Layer 6: 102,464
- Layer 7: 650
- **Total: 121,930**

### MLP Comparison:
- MLP parameters: 100,480 + 1,290 = 101,770
- CNN has ~20,000 MORE parameters but learns spatial features efficiently
- CNN would have FEWER parameters if image was larger (weight sharing advantage scales with image size)

</details>

---

**Complete this worksheet before starting Tutorial T10 coding!**

**Planning prevents problems. A few minutes of design saves hours of debugging.** 📐➡️💻

---

**Instructor Check (Optional):**

After completing, show this worksheet to your instructor for feedback before coding.

Instructor Signature: __________________  Date: __________
