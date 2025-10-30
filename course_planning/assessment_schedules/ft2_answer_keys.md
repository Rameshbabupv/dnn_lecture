# Formative Test 2 (FT2) - Answer Keys

**Course:** 21CSE558T - Deep Neural Network Architectures
**Coverage:** Modules 3-4 (Image Processing & CNNs)
**Date:** November 14, 2025
**Total Marks:** 25 (per set)

---

## SET A - ANSWER KEY

### PART A: Multiple Choice Questions (10 marks)

| Q.No | Correct Answer | Explanation |
|------|----------------|-------------|
| 1 | **c) 3** | RGB images have three channels: Red, Green, and Blue. |
| 2 | **c) Median filter** | Median filter replaces each pixel with the median of neighboring pixels, effectively removing salt-and-pepper noise while preserving edges. |
| 3 | **b) Performs non-maximum suppression and hysteresis thresholding** | Canny includes multiple stages for robust edge detection: gradient calculation, non-maximum suppression, double thresholding, and edge tracking. |
| 4 | **b) Automatic threshold selection for binarization** | Otsu's method automatically determines the optimal threshold to separate foreground and background by maximizing inter-class variance. |
| 5 | **b) Texture features** | LBP captures texture information by comparing each pixel with its neighbors to create binary patterns. |
| 6 | **b) Extract local features from input** | Convolution applies filters/kernels to extract local patterns like edges, textures, and shapes. |
| 7 | **a) Preserves the strongest features/activations** | Max pooling selects the maximum value in each region, preserving the most prominent features. |
| 8 | **b) After pooling layers and before dense layers** | Dropout is typically placed after pooling (e.g., 0.2, 0.3) and before dense layers (e.g., 0.5). NEVER after output layer. |
| 9 | **b) Handwritten digit recognition** | LeNet-5 (1998) by Yann LeCun was designed for recognizing handwritten digits (MNIST), used in check reading systems. |
| 10 | **c) You have limited labeled data and domains are similar** | Transfer learning works best when you have few labeled examples and the source dataset (e.g., ImageNet) is similar to your target task. |

**Total Part A: 10 marks**

---

### PART B: Short Answer Questions (Answer any 3 out of 5 = 15 marks)

**Note:** Students must answer ANY THREE questions from Q11-Q15. Below are complete answers for all 5 questions.

---

#### **Question 11 (5 marks):** Feature Extraction Comparison

**Expected Answer:**

**Traditional Feature Extraction (LBP, GLCM):**
- Manually designed features based on domain knowledge
- Computationally efficient and interpretable
- Works well with small datasets
- Limited to predefined patterns (texture, edges, shapes)
- Requires domain expertise to select appropriate features

**Deep Learning Automatic Feature Extraction:**
- Learns features automatically from data through training
- Can discover complex, hierarchical patterns
- Requires large datasets and computational resources
- End-to-end learning without manual feature engineering
- Better generalization for complex tasks

**When to choose Traditional:**
- Small datasets (few hundred images)
- Limited computational resources
- Need interpretable features
- Well-defined texture/shape problems

**When to choose Deep Learning:**
- Large datasets (thousands+ images)
- Complex patterns beyond manual feature design
- GPU resources available
- State-of-the-art accuracy required

**Marking Scheme:**
- Explanation of traditional features (1.5 marks)
- Explanation of deep learning features (1.5 marks)
- When to choose traditional (1 mark)
- When to choose deep learning (1 mark)

---

#### **Question 12 (5 marks):** Batch Normalization Placement

**Expected Answer:**

**Implementation A (Conv → ReLU → BatchNorm):**
- Old practice from early BatchNorm papers
- Normalizes after activation (post-activation values)
- ReLU zeros out negative values before normalization
- Less effective because normalization acts on already transformed values

**Implementation B (Conv → BatchNorm → ReLU) - CORRECT:**
- Modern best practice
- Normalizes pre-activation values (before ReLU)
- BatchNorm operates on full range of values (including negatives)
- More effective normalization of the linear transformation output

**Why Implementation B is Better:**

1. **Better Gradient Flow:**
   - BatchNorm normalizes the full distribution before activation
   - Prevents activation saturation issues
   - Gradients flow more uniformly through the network

2. **Theoretical Justification:**
   - BatchNorm was designed to normalize covariate shift in layer inputs
   - Pre-activation values are the actual inputs to the non-linearity
   - Normalizing before activation addresses the root cause

3. **Empirical Performance:**
   - Modern architectures (ResNet, DenseNet, EfficientNet) use Conv → BN → ReLU
   - Faster convergence and better final accuracy
   - More stable training with higher learning rates

**Correct Pattern:**
```
Conv2D(32, (3,3), padding='same')  # No activation!
BatchNormalization()
Activation('relu')
```

**Marking Scheme:**
- Implementation A explanation (1 mark)
- Implementation B explanation (1 mark)
- Why B is better (2 marks)
- Practical impact (1 mark)

---

#### **Question 13 (5 marks):** Batch Normalization Placement

**Expected Answer:**

**Comparing Approaches:**

**Approach A: Train from Scratch**
- Requires large dataset (typically 10,000+ images per class)
- 500 images (125 per class) is too small!
- Risk of severe overfitting
- Needs extensive training time
- May not learn meaningful features with limited data

**Approach B: Transfer Learning with VGG16 - CORRECT CHOICE**
- Leverages features learned from 1.2M ImageNet images
- Works well with small datasets (hundreds to thousands)
- Pre-trained features (edges, textures, shapes) are universal
- Faster training (only fine-tune top layers)
- Better generalization

**Transfer Learning Strategy:**

**Step 1: Load Pre-trained VGG16**
```python
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
```

**Step 2: Freeze Base Layers**
```python
base_model.trainable = False  # Preserve pre-trained features
```

**Step 3: Add Custom Classification Head**
```python
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')  # 4 classes
])
```

**Step 4: Initial Training (Frozen Base)**
- Train only new dense layers
- Use moderate learning rate (0.001)
- Train for 10-20 epochs

**Step 5: Fine-tuning (Optional)**
- Unfreeze top few layers of VGG16
- Use very low learning rate (0.0001)
- Train for additional 5-10 epochs

**Why This Works:**
- Early VGG16 layers: Generic features (edges, textures) - keep frozen
- Later VGG16 layers: Specific to natural images - fine-tune
- Custom head: Learns X-ray disease patterns
- Small dataset handled well with pre-trained features

**Marking Scheme:**
- Approach comparison (1 mark)
- Choosing transfer learning with justification (1 mark)
- Transfer learning strategy steps (2 marks)
- Fine-tuning explanation (1 mark)

---

#### **Question 14 (5 marks):** Dropout Placement Strategy

**Expected Answer:**

**What is Wrong:**

1. **Dropout After Output Layer is WRONG:**
   - Dropout after Dense(10, softmax) makes predictions random!
   - During inference, output probabilities become unreliable
   - Final layer should NEVER have dropout after it
   - Defeats the purpose of having a stable classification output

2. **Same Dropout Rate (0.5) Everywhere is WRONG:**
   - Early layers learn low-level features (edges, textures) - need less regularization
   - Later layers learn high-level features - need more regularization
   - Using 0.5 everywhere over-regularizes early layers

**Correct Dropout Placement Strategy:**

**Progressive Dropout Rates:**
- After first pooling: Dropout(0.2) - light regularization
- After second pooling: Dropout(0.3) - moderate regularization
- Before final dense layer: Dropout(0.5) - heavy regularization
- **NEVER** after output layer

**Example Correct Architecture:**
```
Block 1:
  Conv2D → BatchNorm → ReLU
  Conv2D → BatchNorm → ReLU
  MaxPooling2D
  Dropout(0.2)  ← Light

Block 2:
  Conv2D → BatchNorm → ReLU
  Conv2D → BatchNorm → ReLU
  MaxPooling2D
  Dropout(0.3)  ← Moderate

Block 3:
  Conv2D → BatchNorm → ReLU
  GlobalAveragePooling2D

Output:
  Dropout(0.5)  ← Heavy (before output, not after!)
  Dense(10, activation='softmax')
  (NO DROPOUT HERE!)
```

**Why Progressive Rates Work:**
- Early features are general (useful across tasks) - preserve them
- Late features are task-specific - regularize heavily to prevent overfitting
- Gradual increase prevents under-regularization and over-regularization

**Marking Scheme:**
- Identifying dropout after output error (1.5 marks)
- Identifying same rate everywhere error (1 mark)
- Correct progressive strategy (1.5 marks)
- Justification (1 mark)

---

#### **Question 15 (5 marks):** Transfer Learning Strategy

**Expected Answer:**

**Comparing Approaches:**

**Approach A: Train from Scratch**
- Requires large dataset (typically 10,000+ images per class)
- 500 images (125 per class) is too small!
- Risk of severe overfitting
- Needs extensive training time
- May not learn meaningful features with limited data

**Approach B: Transfer Learning with VGG16 - CORRECT CHOICE**
- Leverages features learned from 1.2M ImageNet images
- Works well with small datasets (hundreds to thousands)
- Pre-trained features (edges, textures, shapes) are universal
- Faster training (only fine-tune top layers)
- Better generalization

**Transfer Learning Strategy:**

**Step 1: Load Pre-trained VGG16**
```python
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
```

**Step 2: Freeze Base Layers**
```python
base_model.trainable = False  # Preserve pre-trained features
```

**Step 3: Add Custom Classification Head**
```python
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')  # 4 classes
])
```

**Step 4: Initial Training (Frozen Base)**
- Train only new dense layers
- Use moderate learning rate (0.001)
- Train for 10-20 epochs

**Step 5: Fine-tuning (Optional)**
- Unfreeze top few layers of VGG16
- Use very low learning rate (0.0001)
- Train for additional 5-10 epochs

**Why This Works:**
- Early VGG16 layers: Generic features (edges, textures) - keep frozen
- Later VGG16 layers: Specific to natural images - fine-tune
- Custom head: Learns X-ray disease patterns
- Small dataset handled well with pre-trained features

**Marking Scheme:**
- Approach comparison (1 mark)
- Choosing transfer learning with justification (1 mark)
- Transfer learning strategy steps (2 marks)
- Fine-tuning explanation (1 mark)

---

**Total Part B: 15 marks (any 3 out of 5 questions)**

**SET A TOTAL: 25 marks**

---

---

## SET B - ANSWER KEY

### PART A: Multiple Choice Questions (10 marks)

| Q.No | Correct Answer | Explanation |
|------|----------------|-------------|
| 1 | **b) 0 to 255** | 8-bit images use 2^8 = 256 values, ranging from 0 (black) to 255 (white). |
| 2 | **c) Improving image contrast** | Histogram equalization redistributes pixel intensities to enhance contrast in low-contrast images. |
| 3 | **c) Laplacian** | Laplacian operator uses the second derivative to detect edges, finding regions of rapid intensity change. |
| 4 | **b) A topographic surface with catchment basins** | Watershed algorithm treats the gradient magnitude as a topographic relief, where local minima are "flooded" to create segmentation boundaries. |
| 5 | **b) cv2.cvtColor()** | cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) converts color images to grayscale. |
| 6 | **b) Output size is the same as input (with stride=1)** | 'same' padding adds zeros around the input to ensure output spatial dimensions match input dimensions (when stride=1). |
| 7 | **b) Reducing entire feature map to a single value per channel** | GlobalAveragePooling2D averages all values in each feature map, converting (H, W, C) to (1, 1, C), then flattened to (C,). |
| 8 | **b) After convolution, before activation** | Modern best practice: Conv2D → BatchNormalization → Activation (e.g., ReLU). This normalizes pre-activation values for stable training. |
| 9 | **c) Use of ReLU activation and GPU training** | AlexNet used ReLU (faster training than sigmoid), GPU acceleration, and dropout, achieving breakthrough performance on ImageNet 2012. |
| 10 | **c) Making their weights non-trainable** | Freezing layers (base_model.trainable = False) prevents their weights from updating during training, preserving pre-trained features. |

**Total Part A: 10 marks**

---

### PART B: Short Answer Questions (Answer any 3 out of 5 = 15 marks)

**Note:** Students must answer ANY THREE questions from Q11-Q15. Below are complete answers for all 5 questions.

---

#### **Question 11 (5 marks):** Edge Detection Method Selection

**Expected Answer:**

**Sobel Edge Detector:**
- Uses gradient-based approach with two 3×3 kernels (Gx, Gy)
- Simple, fast, computationally efficient
- Detects edges by calculating gradient magnitude
- Produces thick edges (multiple pixels wide)
- Sensitive to noise (no built-in noise reduction)
- Good for real-time applications

**Canny Edge Detector:**
- Multi-stage algorithm: Gaussian smoothing → gradient → non-maximum suppression → hysteresis thresholding
- Built-in Gaussian blur for noise reduction
- Produces thin, well-localized edges (single pixel wide)
- Uses double thresholding to reduce false edges
- More robust to noise but computationally expensive

**Scenario 1: Low-noise indoor photographs**
- **Choice: Sobel**
- **Justification:** Since images are low-noise, Sobel's simplicity and speed are advantageous. Canny's noise reduction overhead is unnecessary. Sobel's faster processing suitable for batch processing of clean images.

**Scenario 2: High-noise outdoor surveillance**
- **Choice: Canny**
- **Justification:** Surveillance footage has noise from weather, lighting variations, compression artifacts. Canny's Gaussian smoothing reduces noise before edge detection. Hysteresis thresholding eliminates weak false edges from noise. Better edge localization despite computational cost.

**Marking Scheme:**
- Sobel explanation (1 mark)
- Canny explanation (1.5 marks)
- Scenario 1 choice and justification (1 mark)
- Scenario 2 choice and justification (1.5 marks)

---

#### **Question 12 (5 marks):** Segmentation Techniques Comparison

**Expected Answer:**

**Thresholding-based Segmentation:**
- Separates pixels based on intensity values
- Simple and fast (Otsu's method for automatic threshold)
- Works well when tumor has distinct intensity from background
- Limitation: Struggles with varying illumination or overlapping intensity ranges
- Example: Binary thresholding creates two classes (tumor/not tumor)

**Region-based Segmentation:**
- Groups pixels based on similarity criteria (intensity, color, texture)
- Methods: Region growing, region splitting/merging, watershed
- Considers spatial coherence and connectivity
- Better handles gradual transitions and complex boundaries
- More robust to noise than simple thresholding

**For Medical Tumor Segmentation:**

**Region-based is more suitable because:**
- Tumors have complex, irregular boundaries
- Intensity may vary within tumor region
- Background tissue has varying intensity (not uniform)
- Region growing can start from seed point in tumor
- Spatial coherence ensures connected tumor region
- Can incorporate texture and boundary information

**However, thresholding may work if:**
- Tumor has very distinct intensity (high contrast)
- Image quality is consistent
- Used as pre-processing before region-based refinement

**Marking Scheme:**
- Thresholding explanation (1 mark)
- Region-based explanation (1.5 marks)
- Suitability for medical imaging (1.5 marks)
- Justification with reasoning (1 mark)

---

#### **Question 12 (5 marks):** Dropout Placement Strategy

**Expected Answer:**

**What is Wrong:**

1. **Dropout After Output Layer is WRONG:**
   - Dropout after Dense(10, softmax) makes predictions random!
   - During inference, output probabilities become unreliable
   - Final layer should NEVER have dropout after it
   - Defeats the purpose of having a stable classification output

2. **Same Dropout Rate (0.5) Everywhere is WRONG:**
   - Early layers learn low-level features (edges, textures) - need less regularization
   - Later layers learn high-level features - need more regularization
   - Using 0.5 everywhere over-regularizes early layers

**Correct Dropout Placement Strategy:**

**Progressive Dropout Rates:**
- After first pooling: Dropout(0.2) - light regularization
- After second pooling: Dropout(0.3) - moderate regularization
- Before final dense layer: Dropout(0.5) - heavy regularization
- **NEVER** after output layer

**Example Correct Architecture:**
```
Block 1:
  Conv2D → BatchNorm → ReLU
  Conv2D → BatchNorm → ReLU
  MaxPooling2D
  Dropout(0.2)  ← Light

Block 2:
  Conv2D → BatchNorm → ReLU
  Conv2D → BatchNorm → ReLU
  MaxPooling2D
  Dropout(0.3)  ← Moderate

Block 3:
  Conv2D → BatchNorm → ReLU
  GlobalAveragePooling2D

Output:
  Dropout(0.5)  ← Heavy (before output, not after!)
  Dense(10, activation='softmax')
  (NO DROPOUT HERE!)
```

**Why Progressive Rates Work:**
- Early features are general (useful across tasks) - preserve them
- Late features are task-specific - regularize heavily to prevent overfitting
- Gradual increase prevents under-regularization and over-regularization

**Marking Scheme:**
- Identifying dropout after output error (1.5 marks)
- Identifying same rate everywhere error (1 mark)
- Correct progressive strategy (1.5 marks)
- Justification (1 mark)

---

#### **Question 13 (5 marks):** Data Augmentation Strategy

**Expected Answer:**

**What is Wrong:**

1. **rotation_range=180 (±180°) is WRONG:**
   - Airplanes upside-down don't look like airplanes
   - Cars rotated 180° are unrealistic
   - Model learns unrealistic variations
   - Confuses the model during training

2. **vertical_flip=True is WRONG for CIFAR-10:**
   - Airplanes don't fly upside-down in normal scenarios
   - Flipped animals/vehicles change semantic meaning
   - Introduces label noise (flipped airplane ≠ airplane in real world)
   - Only horizontal flip makes sense (left/right mirror is realistic)

**Why Validation Accuracy is Poor:**
- Training data includes unrealistic augmentations
- Model learns to recognize upside-down objects
- Real validation/test images are right-side up
- Mismatch between augmented training and real validation data
- Model wastes capacity learning impossible variations

**Appropriate Augmentation for CIFAR-10:**

```python
ImageDataGenerator(
    rotation_range=15,        # ±15° slight angle variations (realistic)
    width_shift_range=0.1,    # 10% horizontal position shift
    height_shift_range=0.1,   # 10% vertical position shift
    horizontal_flip=True,     # ✓ Mirror is realistic
    vertical_flip=False,      # ✗ Upside-down not realistic
    zoom_range=0.1            # 90-110% zoom (distance variation)
)
```

**Justification:**
- **rotation_range=15:** Small rotations mimic camera angle variations
- **width/height_shift:** Objects can appear at different positions
- **horizontal_flip:** Left-facing vs right-facing airplane is valid
- **zoom_range:** Objects at different distances
- **NO vertical_flip:** Preserves semantic meaning

**General Principles:**
- Augmentation should reflect real-world variations
- Don't create label-changing transformations
- Consider domain semantics
- Apply ONLY to training data

**Marking Scheme:**
- Identifying rotation error (1 mark)
- Identifying vertical flip error (1 mark)
- Appropriate augmentation strategy (2 marks)
- Justification and principles (1 mark)

---

#### **Question 15 (5 marks):** Batch Normalization Placement

**Expected Answer:**

**Implementation A (Conv → ReLU → BatchNorm):**
- Old practice from early BatchNorm papers
- Normalizes after activation (post-activation values)
- ReLU zeros out negative values before normalization
- Less effective because normalization acts on already transformed values

**Implementation B (Conv → BatchNorm → ReLU) - CORRECT:**
- Modern best practice
- Normalizes pre-activation values (before ReLU)
- BatchNorm operates on full range of values (including negatives)
- More effective normalization of the linear transformation output

**Why Implementation B is Better:**

1. **Better Gradient Flow:**
   - BatchNorm normalizes the full distribution before activation
   - Prevents activation saturation issues
   - Gradients flow more uniformly through the network

2. **Theoretical Justification:**
   - BatchNorm was designed to normalize covariate shift in layer inputs
   - Pre-activation values are the actual inputs to the non-linearity
   - Normalizing before activation addresses the root cause

3. **Empirical Performance:**
   - Modern architectures (ResNet, DenseNet, EfficientNet) use Conv → BN → ReLU
   - Faster convergence and better final accuracy
   - More stable training with higher learning rates

**Correct Pattern:**
```
Conv2D(32, (3,3), padding='same')  # No activation!
BatchNormalization()
Activation('relu')
```

**Marking Scheme:**
- Implementation A explanation (1 mark)
- Implementation B explanation (1 mark)
- Why B is better (2 marks)
- Practical impact (1 mark)

---

**Total Part B: 15 marks (any 3 out of 5 questions)**

**SET B TOTAL: 25 marks**

---

---

## Grading Rubric for Part B (5-mark questions)

### Excellent (5 marks):
- Complete explanation with correct reasoning
- All key points covered
- Practical examples or scenarios included
- Clear, well-structured answer
- Demonstrates deep understanding

### Good (4 marks):
- Mostly correct with minor gaps
- Most key points covered
- Some reasoning provided
- Generally clear explanation
- Minor omissions in depth

### Satisfactory (3 marks):
- Basic understanding shown
- Some key points missing
- Limited reasoning
- Answer lacks depth
- Partial explanations

### Poor (1-2 marks):
- Incomplete explanation
- Major gaps in understanding
- Incorrect reasoning
- Fragmented answer
- Missing critical concepts

### Fail (0 marks):
- Completely incorrect
- No relevant answer
- No understanding demonstrated

---

## Common Mistakes to Avoid (for graders)

### Part A (MCQs):
- Ensure students select only ONE answer per question
- No partial marks for MCQs (0 or 1 only)
- Check for erasures or ambiguous markings

### Part B (SAQs):
- Don't penalize for minor terminology variations if concept is correct
- Award partial marks based on rubric
- Look for understanding, not just memorization
- Reward clear reasoning and justification
- Deduct for completely missing key concepts

---

**Last Updated:** October 30, 2025
**Status:** Complete - Ready for FT2 grading
