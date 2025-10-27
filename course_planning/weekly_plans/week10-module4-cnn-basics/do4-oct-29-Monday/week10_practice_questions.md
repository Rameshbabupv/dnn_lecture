# Week 10 Practice Questions

**Unit Test 2 Preparation (October 31, 2025)**
**Module 4: CNN Basics - Week 10 Content**

---

## Overview

These practice questions cover Week 10 material on CNN fundamentals. Unit Test 2 will include similar questions covering Modules 3-4.

**Question Types:**
- Multiple Choice Questions (MCQ): 1 mark each
- Short Answer Questions (SAQ): 5 marks each
- Long Answer Questions (LAQ): 10 marks each

---

## Part A: Multiple Choice Questions (1 mark each)

### MCQ 1
What is the output dimension of a 28×28 input image after applying Conv2D with 32 filters, kernel size (3,3), stride=1, and padding='valid'?

A) 28×28×32
B) 26×26×32 ✓
C) 14×14×32
D) 24×24×32

**Explanation:** Using formula (W - F + 2P) / S + 1 = (28 - 3 + 0) / 1 + 1 = 26

---

### MCQ 2
Which layer introduces translation invariance in CNNs?

A) Conv2D
B) MaxPooling2D ✓
C) Flatten
D) Dense

**Explanation:** Pooling layers downsample and provide translation invariance by selecting the strongest activation in a region, regardless of exact position.

---

### MCQ 3
How many parameters does a Conv2D layer have with 64 filters, kernel size (5,5), and 3 input channels?

A) 64
B) 4,800
C) 4,864 ✓
D) 75

**Explanation:** Parameters = (kernel_h × kernel_w × input_channels + 1) × filters = (5 × 5 × 3 + 1) × 64 = 76 × 64 = 4,864

---

### MCQ 4
What is the primary advantage of CNNs over fully connected networks for image processing?

A) Faster training speed
B) Parameter efficiency through weight sharing ✓
C) Better accuracy on all tasks
D) Simpler architecture

**Explanation:** CNNs use weight sharing (same filters across the image), drastically reducing parameters compared to fully connected layers.

---

### MCQ 5
Which activation function is typically used in the output layer for multi-class classification?

A) ReLU
B) Sigmoid
C) Softmax ✓
D) Tanh

**Explanation:** Softmax converts outputs to probabilities that sum to 1, ideal for multi-class classification.

---

### MCQ 6
What does 'valid' padding mean in convolution operations?

A) Add zeros around the input
B) No padding, output size shrinks ✓
C) Same as input size
D) Invalid operation

**Explanation:** 'valid' padding means no padding is added, so output dimensions are smaller than input (shrinking by kernel_size - 1).

---

### MCQ 7
In a CNN, which layers learn hierarchical features?

A) Only the first convolutional layer
B) Only the fully connected layers
C) All convolutional layers progressively ✓
D) Only pooling layers

**Explanation:** Early conv layers learn simple features (edges), deeper layers learn complex features (shapes, objects) - hierarchical learning.

---

### MCQ 8
What is the output dimension after MaxPooling2D with pool_size=(2,2) applied to a 13×13×32 feature map?

A) 13×13×32
B) 11×11×32
C) 6×6×32 ✓
D) 7×7×32

**Explanation:** Pooling reduces dimensions: 13/2 = 6.5 → 6 (floor division). Output: 6×6×32

---

### MCQ 9
Which loss function should be used with one-hot encoded labels in multi-class classification?

A) binary_crossentropy
B) mean_squared_error
C) categorical_crossentropy ✓
D) sparse_categorical_crossentropy

**Explanation:** categorical_crossentropy works with one-hot encoded labels. sparse_categorical_crossentropy works with integer labels.

---

### MCQ 10
What is the role of the Flatten layer in a CNN?

A) Reduce overfitting
B) Convert 2D feature maps to 1D vector ✓
C) Add non-linearity
D) Downsample the input

**Explanation:** Flatten converts multi-dimensional feature maps to a 1D vector, required before fully connected (Dense) layers.

---

### MCQ 11
Which statement about CNN filters is TRUE?

A) Filters are manually designed for each task
B) Filters are learned automatically during training ✓
C) All filters in a layer are identical
D) Filters are fixed and cannot change

**Explanation:** Unlike traditional hand-crafted filters (Sobel, Gabor), CNN filters are learned from data through backpropagation.

---

### MCQ 12
What happens if you apply too many pooling layers to a small image?

A) Training becomes faster
B) Accuracy improves
C) Feature maps become too small or negative dimensions ✓
D) No effect

**Explanation:** Excessive pooling reduces dimensions too much, potentially causing errors or losing spatial information.

---

## Part B: Short Answer Questions (5 marks each)

### SAQ 1: Output Dimension Calculation

Calculate the output dimensions after each layer for the following architecture:

**Input:** 32×32×3 (RGB image)

**Layers:**
1. Conv2D(64 filters, kernel=(5,5), stride=1, padding='valid')
2. MaxPooling2D(pool_size=(2,2))
3. Conv2D(128 filters, kernel=(3,3), stride=1, padding='valid')

Show all calculations using the formulas:
- Convolution: `(W - F + 2P) / S + 1`
- Pooling: `W / pool_size`

**Expected Answer:**

```
Input: 32×32×3

Layer 1 - Conv2D(64, (5,5), stride=1, padding='valid'):
  Formula: (W - F + 2P) / S + 1
  Calculation: (32 - 5 + 0) / 1 + 1 = 28
  Output: 28×28×64

Layer 2 - MaxPooling2D((2,2)):
  Formula: W / pool_size
  Calculation: 28 / 2 = 14
  Output: 14×14×64

Layer 3 - Conv2D(128, (3,3), stride=1, padding='valid'):
  Formula: (W - F + 2P) / S + 1
  Calculation: (14 - 3 + 0) / 1 + 1 = 12
  Output: 12×12×128

Final Output: 12×12×128
```

**Marking Scheme:**
- Input/output dimensions for each layer: 3 marks (1 per layer)
- Correct formula usage: 1 mark
- Final output dimension: 1 mark

---

### SAQ 2: Parameter Counting

Calculate the total number of trainable parameters in this CNN:

```
Input: 28×28×1
Conv2D(32 filters, 3×3)
MaxPooling2D(2×2)
Conv2D(64 filters, 3×3)
Flatten
Dense(128 units)
Dense(10 units)
```

Show calculations for each layer.

**Expected Answer:**

```
Layer 1 - Conv2D(32, (3,3), input_channels=1):
  Parameters = (kernel_h × kernel_w × input_ch + 1) × filters
  = (3 × 3 × 1 + 1) × 32
  = 10 × 32 = 320

Layer 2 - MaxPooling2D: 0 parameters (no trainable weights)

Layer 3 - Conv2D(64, (3,3), input_channels=32):
  Parameters = (3 × 3 × 32 + 1) × 64
  = 289 × 64 = 18,496

Layer 4 - Flatten: 0 parameters

After Conv2D + Pooling, dimensions: 5×5×64 = 1,600 features

Layer 5 - Dense(128, input_features=1600):
  Parameters = (input + 1) × output
  = (1,600 + 1) × 128 = 204,928

Layer 6 - Dense(10, input_features=128):
  Parameters = (128 + 1) × 10 = 1,290

Total Parameters: 320 + 0 + 18,496 + 0 + 204,928 + 1,290 = 225,034
```

**Marking Scheme:**
- Correct parameter formulas: 2 marks
- Conv layer calculations: 1.5 marks
- Dense layer calculations: 1 mark
- Total sum: 0.5 marks

---

### SAQ 3: CNN vs MLP Comparison

Compare Convolutional Neural Networks (CNNs) and Multi-Layer Perceptrons (MLPs) for image classification. Discuss three key differences with examples.

**Expected Answer:**

**1. Parameter Efficiency (Weight Sharing)**
- **CNN:** Uses same filters across entire image (weight sharing). A 3×3 filter with 32 channels has only ~300 parameters regardless of image size.
- **MLP:** Every pixel connected to every neuron. For 28×28 image with 128 hidden units: 784 × 128 = 100,352 parameters just for first layer.
- **Advantage:** CNNs require far fewer parameters, reducing overfitting and memory usage.

**2. Spatial Structure Preservation**
- **CNN:** Maintains 2D spatial relationships through convolution. Nearby pixels processed together, preserving local patterns.
- **MLP:** Flattens image to 1D vector, losing spatial information. Pixel at position (5,10) treated independently of neighbors.
- **Advantage:** CNNs leverage spatial locality in images (edges, textures are local patterns).

**3. Translation Equivariance/Invariance**
- **CNN:** Same features detected anywhere in image (translation equivariance). Pooling adds translation invariance.
- **MLP:** Must learn to recognize features separately at each position.
- **Advantage:** CNNs generalize better - if trained on centered objects, still recognize off-center objects.

**Conclusion:** CNNs are specialized for image data due to parameter efficiency, spatial awareness, and translation properties.

**Marking Scheme:**
- Three differences clearly explained: 3 marks (1 per difference)
- Examples provided: 1 mark
- Clear comparison structure: 1 mark

---

### SAQ 4: Pooling Layer Purpose

Explain the purpose of pooling layers in CNNs. Discuss two types of pooling and when to use each.

**Expected Answer:**

**Purpose of Pooling Layers:**

1. **Dimensionality Reduction:** Reduces spatial dimensions (width × height), lowering computational cost and memory usage.

2. **Translation Invariance:** Makes network robust to small translations. Feature detected whether object is shifted slightly.

3. **Overfitting Prevention:** Reduces parameters and provides regularization effect.

4. **Receptive Field Expansion:** Each neuron in deeper layers "sees" larger area of input image.

**Max Pooling:**
- **Operation:** Selects maximum value in each pool region
- **Use case:** When you want to detect if feature is present anywhere in region
- **Example:** 2×2 max pooling on [1,3,2,4] → 4 (keeps strongest activation)
- **Best for:** Object detection, classification (presence of feature matters)

**Average Pooling:**
- **Operation:** Calculates average of values in pool region
- **Use case:** When you want smooth downsampling preserving all information
- **Example:** 2×2 average pooling on [1,3,2,4] → 2.5
- **Best for:** Texture analysis, global average pooling before output

**When to use:**
- **Max pooling:** Default choice for most CNNs (VGG, ResNet)
- **Average pooling:** Smoother downsampling, global pooling before classification
- **No pooling:** Very deep networks sometimes use stride=2 convolutions instead

**Marking Scheme:**
- Purpose of pooling (3 points): 1.5 marks
- Max pooling explanation: 1 mark
- Average pooling explanation: 1 mark
- Use cases/comparison: 1.5 marks

---

### SAQ 5: Architecture Design Question

Design a CNN architecture for CIFAR-10 classification (32×32 RGB images, 10 classes). Your design should include:
- At least 3 convolutional layers
- Appropriate pooling
- Dense layers for classification
- Justification for choices (50-75 words)

**Example Answer:**

**Architecture:**
```
Input: 32×32×3

Block 1:
  Conv2D(32, (3,3), relu)  → 30×30×32
  Conv2D(32, (3,3), relu)  → 28×28×32
  MaxPooling2D((2,2))      → 14×14×32

Block 2:
  Conv2D(64, (3,3), relu)  → 12×12×64
  Conv2D(64, (3,3), relu)  → 10×10×64
  MaxPooling2D((2,2))      → 5×5×64

Block 3:
  Conv2D(128, (3,3), relu) → 3×3×128
  MaxPooling2D((2,2))      → 1×1×128

Classification:
  Flatten                   → 128
  Dense(256, relu)
  Dropout(0.5)
  Dense(10, softmax)        → 10 classes
```

**Justification:**
Used VGG-style architecture with increasing filter counts (32→64→128) to learn progressively complex features. Two conv layers before pooling extract features at each scale. 3×3 kernels balance receptive field and parameters. Dropout prevents overfitting. Total ~500K parameters - manageable for CIFAR-10 dataset size.

**Marking Scheme:**
- Complete architecture specified: 2 marks
- Correct output dimensions: 1 mark
- Appropriate layer choices: 1 mark
- Justification quality: 1 mark

---

## Part C: Long Answer Questions (10 marks each)

### LAQ 1: Complete Convolution Calculation

Given this 5×5 input image and 3×3 kernel:

**Image:**
```
┌─────────────┐
│ 2  1  3  0  1 │
│ 1  4  2  1  0 │
│ 3  1  5  2  1 │
│ 0  2  1  4  3 │
│ 1  0  3  1  2 │
└─────────────┘
```

**Kernel (Vertical Edge Detector):**
```
┌──────────┐
│ -1  0  1 │
│ -2  0  2 │
│ -1  0  1 │
└──────────┘
```

**Parameters:** Stride=1, Padding=0

**Tasks:**
1. Calculate output dimensions (1 mark)
2. Perform complete convolution, showing detailed calculations for ALL positions (6 marks)
3. Implement in NumPy code (2 marks)
4. Explain what this kernel detects and interpret results (1 mark)

**Expected Answer:**

**1. Output Dimensions:**
```
Formula: (W - F + 2P) / S + 1
Output = (5 - 3 + 0) / 1 + 1 = 3×3
```

**2. Convolution Calculations:**

**Position (0,0):**
```
Region:        Kernel:         Element-wise mult:
┌───────┐     ┌──────────┐   ┌────────────────┐
│ 2 1 3 │  *  │ -1  0  1 │ = │ -2   0   3     │
│ 1 4 2 │     │ -2  0  2 │   │ -2   0   4     │
│ 3 1 5 │     │ -1  0  1 │   │ -3   0   5     │
└───────┘     └──────────┘   └────────────────┘

Sum = -2 + 0 + 3 + (-2) + 0 + 4 + (-3) + 0 + 5 = 5
Output[0,0] = 5
```

**Position (0,1):**
```
Region:        Kernel:         Element-wise mult:
┌───────┐     ┌──────────┐   ┌────────────────┐
│ 1 3 0 │  *  │ -1  0  1 │ = │ -1   0   0     │
│ 4 2 1 │     │ -2  0  2 │   │ -8   0   2     │
│ 1 5 2 │     │ -1  0  1 │   │ -1   0   2     │
└───────┘     └──────────┘   └────────────────┘

Sum = -1 + 0 + 0 + (-8) + 0 + 2 + (-1) + 0 + 2 = -6
Output[0,1] = -6
```

(Continue for all 9 positions... student should show all)

**Final Output Matrix:**
```
┌─────────┐
│  5  -6   2 │
│  8   0  -4 │
│ -2   6   1 │
└─────────┘
```

**3. NumPy Implementation:**
```python
import numpy as np

# Define input and kernel
image = np.array([
    [2, 1, 3, 0, 1],
    [1, 4, 2, 1, 0],
    [3, 1, 5, 2, 1],
    [0, 2, 1, 4, 3],
    [1, 0, 3, 1, 2]
])

kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Perform convolution
output = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        region = image[i:i+3, j:j+3]
        output[i, j] = np.sum(region * kernel)

print(output)
```

**4. Interpretation:**
This is a **vertical edge detector** (Sobel-like kernel). Positive values indicate right-to-left intensity changes (vertical edges), negative values indicate left-to-right changes. The kernel emphasizes vertical gradients while ignoring horizontal information (middle column is zeros).

In the result:
- Position (0,0) = 5: Strong vertical edge (2→3 transition)
- Position (0,1) = -6: Strong vertical edge in opposite direction
- Position (1,1) = 0: No vertical edge (uniform region)

**Marking Scheme:**
- Output dimensions: 1 mark
- All 9 convolution calculations shown: 6 marks (0.66 per position)
- NumPy code correct: 2 marks
- Interpretation: 1 mark

---

### LAQ 2: CNN Implementation and Analysis

Implement a CNN in Keras for Fashion-MNIST classification with the following specifications:

**Requirements:**
- Input: 28×28 grayscale images
- Architecture: 2 convolutional blocks + fully connected layers
- Include model.summary() output
- Calculate total parameters manually and verify
- Explain role of each layer type

**Tasks:**
1. Write complete Keras code (4 marks)
2. Show model.summary() output (1 mark)
3. Calculate total parameters manually (3 marks)
4. Explain role of each layer type (2 marks)

**Expected Answer:**

**1. Keras Code:**
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalize and reshape
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# One-hot encode labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Build model
model = keras.Sequential([
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu',
                  input_shape=(28, 28, 1), name='conv1'),
    layers.MaxPooling2D((2, 2), name='pool1'),

    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
    layers.MaxPooling2D((2, 2), name='pool2'),

    # Fully connected layers
    layers.Flatten(name='flatten'),
    layers.Dense(64, activation='relu', name='dense1'),
    layers.Dense(10, activation='softmax', name='output')
], name='Fashion_MNIST_CNN')

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display architecture
model.summary()

# Train
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

**2. Model Summary Output:**
```
Model: "Fashion_MNIST_CNN"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv1 (Conv2D)              (None, 26, 26, 32)        320
 pool1 (MaxPooling2D)        (None, 13, 13, 32)        0
 conv2 (Conv2D)              (None, 11, 11, 64)        18496
 pool2 (MaxPooling2D)        (None, 5, 5, 64)          0
 flatten (Flatten)           (None, 1600)              0
 dense1 (Dense)              (None, 64)                102464
 output (Dense)              (None, 10)                650
=================================================================
Total params: 121,930
Trainable params: 121,930
Non-trainable params: 0
```

**3. Manual Parameter Calculation:**
```
Conv1: (3×3×1 + 1) × 32 = 10 × 32 = 320
Pool1: 0 (no trainable parameters)
Conv2: (3×3×32 + 1) × 64 = 289 × 64 = 18,496
Pool2: 0
Flatten: 0 (5×5×64 = 1,600 features)
Dense1: (1,600 + 1) × 64 = 102,464
Output: (64 + 1) × 10 = 650

Total: 320 + 18,496 + 102,464 + 650 = 121,930 ✓
```

**4. Layer Roles:**

- **Conv2D:** Learns spatial features using sliding filters. First layer learns edges/textures, second learns shapes/patterns. Parameters shared across image (weight sharing).

- **MaxPooling2D:** Downsamples by keeping maximum activation in each region. Provides translation invariance and reduces dimensions.

- **Flatten:** Converts 2D feature maps (5×5×64) to 1D vector (1,600), required for dense layers.

- **Dense (hidden):** Fully connected layer for high-level reasoning, combining features learned by conv layers.

- **Dense (output):** Classification layer with softmax activation, outputs probability distribution over 10 classes.

**Marking Scheme:**
- Complete working code: 4 marks
- model.summary() shown: 1 mark
- Manual parameter calculation: 3 marks
- Layer role explanations: 2 marks

---

## Answer Key Summary

### MCQ Answers:
1. B (26×26×32)
2. B (MaxPooling2D)
3. C (4,864)
4. B (Parameter efficiency)
5. C (Softmax)
6. B (No padding, shrinks)
7. C (All conv layers)
8. C (6×6×32)
9. C (categorical_crossentropy)
10. B (Convert 2D to 1D)
11. B (Learned automatically)
12. C (Too small dimensions)

---

## Study Tips for Unit Test 2

1. **Master formulas:**
   - Convolution output: `(W - F + 2P) / S + 1`
   - Pooling output: `W / pool_size`
   - Conv parameters: `(K_h × K_w × in_ch + 1) × filters`
   - Dense parameters: `(input + 1) × output`

2. **Practice calculations by hand:**
   - Work through multiple convolution examples
   - Calculate output dimensions for various architectures
   - Count parameters for different layer configurations

3. **Understand concepts deeply:**
   - Why CNNs work for images (spatial structure, weight sharing)
   - Role of each layer type
   - Tradeoffs in architecture design

4. **Code familiarity:**
   - Be able to write basic CNN in Keras from scratch
   - Understand model.compile() and model.fit() parameters
   - Know how to preprocess image data

5. **Review materials:**
   - Week 10 DO3 Jupyter notebooks (00-09)
   - Tutorial T10 solution code
   - Quick reference cheat sheet
   - Homework assignment calculations

---

**Good luck on Unit Test 2! Focus on understanding, not memorization. 🎓**
