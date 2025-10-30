# CNN Architecture Design Worksheet - ANSWER KEY

**Course:** 21CSE558T - Deep Neural Network Architectures
**Module 4:** CNNs (Week 2 of 3)
**Date:** October 31, 2025
**Instructor Use Only**

---

## Instructions for Using This Answer Key

This answer key is designed for **active learning and assessment**. Answers are provided in **randomized order** (labeled A through Z) and do NOT match the question order.

**How to use:**
1. Students must find the correct answer from the shuffled list below
2. Students write the answer letter (A, B, C, etc.) next to each question
3. Use the **Answer Matching Table** at the end to verify correctness
4. This format prevents simple copying and ensures understanding

---

## SHUFFLED ANSWER BANK

Students must match these answers to the correct questions. Answers are intentionally out of order!

---

### Answer A
**Problem 2.1 - Corrected BatchNorm Placement**

```python
model = Sequential([
    # Block 1 - CORRECTED
    Conv2D(32, (3,3), padding='same'),
    BatchNormalization(),           # BEFORE activation
    Activation('relu'),             # Activation last
    MaxPooling2D((2,2)),

    # Block 2 - CORRECTED
    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(),           # BEFORE activation
    Activation('relu'),             # Activation last
    MaxPooling2D((2,2)),
])
```

**Explanation:** BatchNormalization should be placed AFTER Conv2D but BEFORE the activation function. The original code had the activation built into Conv2D, which applies it before BatchNorm. The correct pattern is: Conv → BN → Activation.

---

### Answer B
**Problem 6.1 - Five Problems Identified**

1. **No Batch Normalization** - Missing quality control checkpoints for stable training
2. **No Dropout** - No regularization to prevent co-adaptation
3. **Huge FC layers** - Dense(1024) and Dense(512) cause parameter explosion
4. **Using Flatten instead of GlobalAveragePooling2D** - Old-style approach with too many parameters
5. **Too few Conv layers** - Only 3 conv layers, modern CNNs stack more (2 per block)

**Corrected Architecture:**

```python
model = Sequential([
    # Block 1
    Conv2D(32, (3,3), padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(32, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),
    Dropout(0.2),

    # Block 2
    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    # Block 3
    Conv2D(128, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    GlobalAveragePooling2D(),  # Modern approach!

    # Output
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

---

### Answer C
**Problem 4.1 - Data Augmentation Appropriateness Table**

| Augmentation | Cat/Dog Classification | Medical X-rays | Handwritten Digits (0-9) |
|--------------|------------------------|----------------|--------------------------|
| Rotation ±15° | ✓ | ✗ | ✓ |
| Rotation ±180° | ✗ | ✗ | ✗ |
| Horizontal Flip | ✓ | ✗ | ✗ |
| Vertical Flip | ✗ | ✗ | ✗ |
| Zoom 90-110% | ✓ | ✓ | ✓ |
| Brightness ±20% | ✓ | ✓ | ✗ |
| Color Jitter | ✓ | ✗ | ✗ |

**Reasoning:**
- **Cat/Dog:** Most augmentations OK except extreme rotations and vertical flips (animals don't walk upside down)
- **Medical X-rays:** NO flips (left lung ≠ right lung, anatomical asymmetry matters). Minimal rotation. Brightness OK for different machines.
- **Handwritten Digits:** NO flips (creates invalid digits: 6→9, b→q). Minimal rotation. No color (grayscale). No brightness (digits are binary).

---

### Answer D
**Problem 3.2 - Overfitting Diagnosis**

**Correct Answer: (a) Increase dropout rate to 0.5-0.6**

**Explanation:**
The model shows **severe overfitting** (33% gap between train 98% and val 65%). Current Dropout(0.3) is insufficient. We need stronger regularization:
- Increase dropout to 0.5-0.6 after FC layers
- Consider adding dropout after conv layers (0.2-0.3)
- May also need data augmentation
- The large gap indicates the model is memorizing training data, so we need to force it to generalize more aggressively

---

### Answer E
**Problem 7.1 - Architecture Comparison Table**

| Aspect | Architecture A | Architecture B | Winner? |
|--------|----------------|----------------|---------|
| Total Parameters | ~2.1M | ~650K | B |
| Overfitting Risk | High | Low | B |
| Training Speed | Fast | Slow | A |
| Expected Test Acc | ~65% | ~82% | B |
| Regularization | No | Yes | B |

**Which architecture to choose:** **Architecture B (Modern Style)**

**Reasoning:**
- Architecture B has 3× fewer parameters (less overfitting)
- Architecture B uses BatchNorm (faster convergence, acts as regularization)
- Architecture B uses Dropout (prevents co-adaptation)
- Architecture B uses GlobalAvgPool (massive parameter reduction)
- Architecture B uses double-conv blocks (better feature learning)
- Training is slower per epoch but achieves much better generalization
- Architecture B is production-ready, Architecture A is outdated

---

### Answer F
**Problem 1.1 - Pooling Output: 64×64 → MaxPool(2×2, stride=2)**

**Calculation:**
```
output_size = (input_size - pool_size) / stride + 1
output_height = (64 - 2) / 2 + 1 = 62 / 2 + 1 = 31 + 1 = 32
output_width = (64 - 2) / 2 + 1 = 62 / 2 + 1 = 31 + 1 = 32
```

**Answer:**
- Output Height: **32**
- Output Width: **32**

---

### Answer G
**Problem 4.2 - Scenario A: Natural Images (Cats, Dogs, Birds)**

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,          # Animals at different angles
    width_shift_range=0.1,      # Position variations
    height_shift_range=0.1,     # Position variations
    horizontal_flip=True,       # Left/right views
    zoom_range=0.1,             # Distance variations
    brightness_range=[0.8, 1.2], # Lighting conditions
    fill_mode='nearest'
)
```

---

### Answer H
**Problem 6.2 - Diagnosis and Solutions**

**Is this overfitting or underfitting?** **Underfitting**

**Why is training so slow?**
- Too much regularization applied after fixing overfitting
- Batch size too small (needs more samples per batch)
- Model capacity may be too small now
- BatchNorm statistics unstable with small batches
- Dropout may be too aggressive

**Propose 3 solutions to speed up training:**
1. **Increase batch size to 64-128** - Better GPU utilization, more stable BatchNorm statistics
2. **Use learning rate scheduling** - Start with higher learning rate (0.001), decay after plateau
3. **Reduce dropout slightly** - Current regularization may be too strong, preventing learning
4. **Add BatchNormalization** - Speeds up convergence 2-3×
5. **Use data augmentation instead of heavy dropout** - Better regularization without slowing training

---

### Answer I
**Problem 1.2 - Pooling Output: 224×224 → MaxPool(3×3, stride=2)**

**Calculation:**
```
output_size = (input_size - pool_size) / stride + 1
output_height = (224 - 3) / 2 + 1 = 221 / 2 + 1 = 110.5 + 1 = 111.5 → 111 (floor)
output_width = (224 - 3) / 2 + 1 = 221 / 2 + 1 = 110.5 + 1 = 111.5 → 111 (floor)
```

**Answer:**
- Output Height: **111**
- Output Width: **111**

---

### Answer J
**Problem 3.1 - Dropout Strategy Design**

**Dropout Rates:**
- After first MaxPool: **0.2** (light dropout for conv layers)
- After second MaxPool: **0.3** (moderate dropout, deeper in network)
- After GlobalAveragePooling: **0.5** (heavy dropout before output)
- After Dense output: **NO** (NEVER dropout after output layer!)

**Reasoning for each decision:**

**After first MaxPool (0.2):**
- Early conv layers learn basic features (edges, colors)
- Light dropout (0.2) prevents overfitting without losing important features
- Too much dropout here hurts feature learning

**After second MaxPool (0.3):**
- Mid-level features (textures, parts)
- Slightly higher dropout (0.3) as network goes deeper
- More abstract features, more prone to overfitting

**After GlobalAveragePooling (0.5):**
- High-level features before classification
- Heavy dropout (0.5) is standard before output
- Most critical regularization point
- Prevents co-adaptation of high-level features

**After Dense output (NO):**
- NEVER use dropout after output layer!
- Would make predictions random
- Softmax needs stable inputs for proper probability distribution

---

### Answer K
**Problem 1.3 - CNN Sequence Dimensions**

**Starting:** 32×32×3

**After Conv2D (step 2):** 32×32×32
- Conv with padding='same' keeps spatial dimensions
- 32 filters → depth becomes 32

**After MaxPool (step 3):** 16×16×32
- Pool(2×2, stride=2) halves dimensions
- (32-2)/2 + 1 = 16
- Depth unchanged (32 channels)

**After Conv2D (step 4):** 16×16×64
- Conv with padding='same' keeps spatial dimensions
- 64 filters → depth becomes 64

**After MaxPool (step 5):** 8×8×64
- Pool(2×2, stride=2) halves dimensions again
- (16-2)/2 + 1 = 8
- Depth unchanged (64 channels)

---

### Answer L
**Problem 5.1 - Complete Modern CNN Architecture**

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

    # Block 2: 64 filters
    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    # Block 3: 128 filters
    Conv2D(128, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(128, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    # Classification head
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

### Answer M
**Problem 2.2 - Batch Normalization Multiple Choice**

**Correct statements:**
- ✓ **b) BatchNormalization normalizes inputs to mean=0, std=1**
- ✓ **c) BatchNormalization has 2 learnable parameters per feature: γ (scale) and β (shift)**
- ✓ **e) BatchNormalization acts as a regularization technique**
- ✓ **f) BatchNormalization requires batch size ≥ 16 for reliable statistics**

**Incorrect statements:**
- ✗ a) BatchNormalization should be placed AFTER the activation function (WRONG - should be BEFORE)
- ✗ d) BatchNormalization slows down training (WRONG - speeds up 2-3×)
- ✗ g) BatchNormalization should be used after the output layer (WRONG - never after output!)

**Number of correct answers: 4**

---

### Answer N
**Problem 5.2 - Architectural Justifications**

**Why did you place BatchNorm where you did?**
- Placed after Conv2D but BEFORE activation in every block
- This is the correct modern pattern: Conv → BN → Activation
- BatchNorm normalizes the pre-activation values to mean=0, std=1
- Then learnable γ and β allow the network to scale/shift as needed
- This prevents internal covariate shift and speeds up training 2-3×
- Placing after activation (old approach) is less effective

**Why did you choose those dropout rates?**
- 0.2 after first pooling: Light dropout for early conv layers (basic features)
- 0.3 after second/third pooling: Moderate dropout for mid/deep conv layers
- 0.5 before output: Heavy dropout for final FC layer (standard practice)
- Progressive increase in dropout rate as network goes deeper
- Conv layers less prone to overfitting than FC layers, so lower rates
- NO dropout after output (would make predictions random)

**Why use Global Average Pooling instead of Flatten + Dense?**
- **Parameter reduction:** 7×7×128 = 6272 → Flatten+Dense(1024) = 6.4M parameters
- **With GAP:** 7×7×128 → GAP → 128 values → Dense(10) = only 1,290 parameters
- **5000× fewer parameters!** Dramatically reduces overfitting
- GAP acts as structural regularization (spatial dimensions → single value per channel)
- Forces each filter to learn one concept (more interpretable)
- Modern best practice for image classification CNNs

---

### Answer O
**Problem 1.4 - Parameter Comparison**

**Scenario A (Old approach):**
- Last conv output: 7×7×512 = 25,088 values
- Flatten → Dense(4096): 25,088 × 4,096 = **102,760,448 parameters**
- Dense(4096) → Dense(1000): 4,096 × 1,000 = **4,096,000 parameters**
- **Total: ~106.8M parameters** just in FC layers!

**Scenario B (Modern approach):**
- Last conv output: 7×7×512 = 25,088 values
- GlobalAveragePooling2D(): 7×7→1 per channel = 512 values (0 parameters)
- Dense(1000): 512 × 1,000 = **512,000 parameters**
- **Total: 512K parameters**

**Parameter reduction: 106.8M / 0.512M = 208× fewer parameters!**

This is why modern CNNs use Global Average Pooling - it virtually eliminates the parameter explosion problem that plagued early architectures like AlexNet.

---

### Answer P
**Problem 7.2 - Training Curve Predictions**

**Architecture A (Old Style - No Regularization):**
```
Epoch 1:  Train: 35%, Val: 32%  (Random initialization)
Epoch 10: Train: 85%, Val: 62%  (Learning but starting to overfit)
Epoch 50: Train: 99%, Val: 65%  (Severe overfitting - memorized training data)
```

**Architecture B (Modern Style - Full Regularization):**
```
Epoch 1:  Train: 25%, Val: 24%  (Slower start due to dropout/regularization)
Epoch 10: Train: 70%, Val: 68%  (Steady progress, small gap)
Epoch 50: Train: 85%, Val: 82%  (Excellent generalization, small gap maintained)
```

**Explain the difference:**

**Architecture A** shows classic overfitting behavior:
- Quickly memorizes training data (high train accuracy)
- Fails to generalize (low validation accuracy)
- Gap grows larger over time (99% - 65% = 34% gap!)
- No regularization means network just memorizes

**Architecture B** shows healthy learning:
- Slower training due to dropout/regularization (intentional!)
- Train and validation track closely (small gap)
- Better final validation accuracy despite lower train accuracy
- Regularization forces network to learn general patterns, not memorize
- This is exactly what we want in production models!

**Key Insight:** Lower training accuracy with regularization is not a bug, it's a feature! We trade some training accuracy for much better generalization.

---

### Answer Q
**Problem 5.3 - Total Parameter Calculation**

**Total parameters: Approximately 216,458 parameters**

**Calculation for 2 example layers:**

**Layer: Conv2D(32, (3,3), padding='same') - First layer**
- Input shape: 32×32×3
- Kernel: 3×3
- Filters: 32
- Parameters per filter: 3×3×3 (kernel) + 1 (bias) = 28
- Total: 32 filters × 28 = **896 parameters**

**Layer: Conv2D(64, (3,3), padding='same') - Second block**
- Input channels: 32 (from previous layer)
- Kernel: 3×3
- Filters: 64
- Parameters per filter: 3×3×32 (kernel) + 1 (bias) = 289
- Total: 64 filters × 289 = **18,496 parameters**

**Layer: GlobalAveragePooling2D()**
- **0 parameters** (just takes average, no learnable weights)

**Layer: Dense(10, activation='softmax')**
- Input: 128 (from GAP of 128 channels)
- Output: 10 classes
- Parameters: 128 × 10 + 10 (bias) = **1,290 parameters**

**Why so few parameters?**
- Global Average Pooling eliminates the massive FC layers
- Old approach: 7×7×128 → Flatten → Dense(512) = 1.6M+ parameters
- Modern approach: 128 → Dense(10) = 1,290 parameters
- **1,200× reduction** in final layer parameters!

---

### Answer R
**Problem 4.2 - Scenario B: Handwritten Digits (MNIST)**

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,          # Slight rotation only (careful with 6 vs 9!)
    width_shift_range=0.1,      # Writing position varies
    height_shift_range=0.1,     # Writing position varies
    zoom_range=0.1,             # Size variations
    fill_mode='constant',       # Black background for digits
    cval=0                      # Black padding
    # NO horizontal_flip (creates invalid digits)
    # NO vertical_flip (6→9, b→q problems)
    # NO brightness (digits are binary black/white)
    # NO color jitter (grayscale only)
)
```

**Explanation of difference:**

**Natural images (Scenario A):**
- Horizontal flips OK (cats look same from left/right)
- Brightness varies (day/night/indoor/outdoor)
- Color matters (brown dog vs white dog)
- More aggressive augmentation safe

**Handwritten digits (Scenario B):**
- NO flips (would create invalid/wrong digits)
- NO brightness (digits are binary, contrast defines them)
- NO color (grayscale images only)
- Minimal rotation (±10° max, careful with 6/9 confusion)
- Shift and zoom OK (people write in different sizes/positions)
- Much more conservative augmentation due to domain constraints

**Key Lesson:** Always consider domain semantics before augmentation!

---

### Answer S
**Bonus Problem - Ultra-Efficient CNN (<100K parameters)**

```python
model = Sequential([
    # Block 1: Keep filters small (16 instead of 32)
    Conv2D(16, (3,3), padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),
    Dropout(0.2),

    # Block 2: 32 filters (modest increase)
    Conv2D(32, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    # Block 3: 64 filters (stay below 128)
    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    GlobalAveragePooling2D(),  # CRITICAL for parameter reduction!

    # Output: Direct connection (no intermediate FC layer)
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

**Estimated parameters: ~48,000 parameters** (well under 100K budget!)

**Techniques used to reduce parameters:**

1. **Reduced filter counts (16→32→64 instead of 32→64→128):**
   - Fewer filters = fewer parameters in each conv layer
   - Still maintains progressive deepening pattern
   - Cuts parameters by ~75%

2. **Global Average Pooling instead of Flatten+Dense:**
   - Eliminates massive FC layer parameters
   - 64 channels → GAP → 64 values → Dense(10) = only 650 params
   - Old approach would be: 4×4×64 → Dense(128) = 131K params (exceeds budget!)

3. **Single conv layer per block instead of double:**
   - One 3×3 conv per block instead of two
   - Reduces total conv layers from 6 to 3
   - Halves conv layer parameters

4. **Removed intermediate FC layers entirely:**
   - No Dense(128) or Dense(256) before output
   - Direct: GAP → Dense(10)
   - Massive parameter savings

5. **Used BatchNormalization for regularization:**
   - Minimal parameters (4 per channel: γ, β, μ, σ)
   - Strong regularization effect
   - Allows smaller model to work effectively

**Expected accuracy:** 75-80% on CIFAR-10 (impressive for <100K params!)

---

## 📋 ANSWER MATCHING TABLE

Use this table to check student answers. Students should have written the correct answer letter next to each question.

| Question | Correct Answer | Topic |
|----------|----------------|-------|
| **Problem 1.1** | F | Pooling output dimensions (64×64→32×32) |
| **Problem 1.2** | I | Pooling output dimensions (224×224→111×111) |
| **Problem 1.3** | K | CNN sequence dimension tracking |
| **Problem 1.4** | O | Parameter explosion comparison (208× reduction) |
| **Problem 2.1** | A | Correct BatchNorm placement code |
| **Problem 2.2** | M | BatchNorm multiple choice (4 correct) |
| **Problem 3.1** | J | Dropout strategy design with rates |
| **Problem 3.2** | D | Overfitting diagnosis (increase dropout) |
| **Problem 4.1** | C | Data augmentation appropriateness table |
| **Problem 4.2 (A)** | G | Natural images augmentation code |
| **Problem 4.2 (B)** | R | Handwritten digits augmentation code |
| **Problem 5.1** | L | Complete modern CNN architecture code |
| **Problem 5.2** | N | Architectural justifications (BatchNorm, Dropout, GAP) |
| **Problem 5.3** | Q | Total parameter calculation (~216K) |
| **Problem 6.1** | B | Debugging: 5 problems + corrected code |
| **Problem 6.2** | H | Underfitting diagnosis + speed solutions |
| **Problem 7.1** | E | Architecture comparison table |
| **Problem 7.2** | P | Training curve predictions + explanation |
| **Bonus** | S | Ultra-efficient CNN design (<100K params) |

---

## 🎯 Grading Rubric

### Problem-wise Point Distribution

| Problem | Max Points | Answer Letter | Key Concepts Tested |
|---------|-----------|---------------|---------------------|
| 1.1 | 2 | F | Pooling formula application |
| 1.2 | 2 | I | Pooling with non-standard kernel |
| 1.3 | 3 | K | Dimension tracking through CNN |
| 1.4 | 3 | O | Parameter explosion understanding |
| 2.1 | 5 | A | BatchNorm placement (code + explanation) |
| 2.2 | 5 | M | BatchNorm conceptual understanding |
| 3.1 | 6 | J | Dropout placement strategy |
| 3.2 | 4 | D | Overfitting diagnosis |
| 4.1 | 9 | C | Domain-appropriate augmentation |
| 4.2 | 6 | G + R | Augmentation implementation |
| 5.1 | 15 | L | Complete architecture design |
| 5.2 | 5 | N | Architectural reasoning |
| 5.3 | 5 | Q | Parameter calculation |
| 6.1 | 7 | B | Debugging overfitting |
| 6.2 | 8 | H | Underfitting diagnosis |
| 7.1 | 10 | E | Architecture comparison |
| 7.2 | 5 | P | Training dynamics understanding |
| **Subtotal** | **100** | | |
| Bonus | 5 | S | Efficient architecture design |
| **Total** | **105** | | |

---

## 📊 Common Student Mistakes to Watch For

### High-Risk Areas (Where Students Often Lose Points)

1. **Problem 1.4 (Parameter Calculation):**
   - Forgetting to multiply Flatten size by Dense size
   - Not calculating both scenarios fully
   - Math errors in large multiplications

2. **Problem 2.1 (BatchNorm Placement):**
   - Still putting activation in Conv2D
   - Placing BatchNorm after activation
   - Missing the separation of concerns

3. **Problem 3.1 (Dropout Rates):**
   - Using dropout after output layer (critical mistake!)
   - Using same rate everywhere (not progressive)
   - Too high dropout in conv layers (>0.3)

4. **Problem 4.1 (Augmentation Appropriateness):**
   - Allowing horizontal flip for medical X-rays (left≠right organs)
   - Allowing flips for digits (6→9 problem)
   - Not considering domain semantics

5. **Problem 5.1 (Complete Architecture):**
   - Using Flatten instead of GlobalAveragePooling2D
   - Placing BatchNorm after activation
   - Missing dropout layers
   - Wrong filter progression

6. **Problem 6.1 (Debugging):**
   - Identifying <5 problems
   - Not providing corrected code
   - Corrected code still has issues

---

## 💡 Teaching Notes

### Concepts Successfully Demonstrated by This Assessment

Students who score well demonstrate understanding of:
- ✅ Pooling mathematics and dimension tracking
- ✅ Parameter explosion problem and Global Average Pooling solution
- ✅ Correct BatchNorm placement (Conv → BN → Activation)
- ✅ Strategic dropout placement and rate selection
- ✅ Domain-appropriate data augmentation
- ✅ Complete modern CNN architecture patterns
- ✅ Debugging overfitting vs underfitting
- ✅ Architecture comparison and trade-offs

### Red Flags for Remediation

Students who struggle with these areas need additional help:
- ❌ Dropout after output layer (fundamental misunderstanding)
- ❌ BatchNorm after activation (outdated knowledge)
- ❌ Using Flatten+Dense instead of Global Average Pooling
- ❌ Allowing domain-inappropriate augmentation (e.g., flips for digits)
- ❌ Cannot diagnose overfitting vs underfitting

---

**END OF ANSWER KEY**

**Last Updated:** October 30, 2025
**Status:** Ready for Assessment

