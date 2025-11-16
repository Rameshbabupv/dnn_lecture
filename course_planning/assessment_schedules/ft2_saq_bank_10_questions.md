# 10 SAQ Question Bank for Formative Test 2 (FT2)

**Course:** 21CSE558T - Deep Neural Network Architectures
**Coverage:** Modules 3-4 (Image Processing & CNNs)
**Format:** 10 marks each
**Total Questions:** 10 (5 from Module 3, 5 from Module 4)
**Date:** November 14, 2025

---

## Module 3: Image Processing & Deep Neural Networks (5 questions)

### **Q1.** Feature Extraction Comparison

You are building an image classification system and need to choose between traditional feature extraction methods (like LBP or GLCM) and deep learning automatic feature extraction. Explain the key differences and when you would choose each approach.

**Module:** 3 | **Week:** 7-8 | **Difficulty:** Moderate | **CO:** CO-3

**Expected Answer (10 marks):**

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
- Explanation of traditional features (3 marks)
- Explanation of deep learning features (3 marks)
- When to choose traditional (2 marks)
- When to choose deep learning (2 marks)

---

### **Q2.** Image Segmentation Techniques

You have a medical image dataset where you need to separate tumor regions from healthy tissue. Compare thresholding-based segmentation with region-based segmentation approaches. Explain which would be more suitable for this task and why.

**Module:** 3 | **Week:** 7-8 | **Difficulty:** Moderate | **CO:** CO-3

**Expected Answer (10 marks):**

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
- Thresholding explanation (2 marks)
- Region-based explanation (3 marks)
- Suitability for medical imaging (3 marks)
- Justification with reasoning (2 marks)

---

### **Q3.** Edge Detection Method Selection

You are processing images from two different scenarios: (1) Low-noise indoor photographs, (2) High-noise outdoor surveillance footage. Explain which edge detection method (Sobel vs. Canny) you would choose for each scenario and justify your choices.

**Module:** 3 | **Week:** 7 | **Difficulty:** Easy | **CO:** CO-3

**Expected Answer (10 marks):**

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
- Sobel explanation (2 marks)
- Canny explanation (3 marks)
- Scenario 1 choice and justification (2 marks)
- Scenario 2 choice and justification (3 marks)

---

### **Q8.** Image Noise and Filtering

You are developing an image processing pipeline for two applications: (1) Medical imaging where images have Gaussian noise from sensor electronics, (2) Old scanned documents with salt-and-pepper noise from scanning artifacts. Explain which filter (Gaussian blur vs Median filter) you would use for each application and justify your choices.

**Module:** 3 | **Week:** 7 | **Difficulty:** Easy-Moderate | **CO:** CO-3

**Expected Answer (10 marks):**

**Gaussian Filter:**
- Smoothing filter based on Gaussian distribution (bell curve)
- Weighted average of neighboring pixels (center weighted more)
- Applied through convolution with Gaussian kernel
- Effective for continuous noise (Gaussian noise from sensors)
- Smooths images by reducing high-frequency components
- Preserves overall structure but blurs edges

**Median Filter:**
- Non-linear filter that replaces pixel with median of neighborhood
- Sorts neighborhood pixel values and picks middle value
- Excellent for impulsive noise (salt-and-pepper, isolated pixels)
- Preserves edges better than Gaussian for sparse noise
- Does not use averaging, so outliers don't affect result
- Computationally more expensive than Gaussian

**Application 1: Medical Imaging with Gaussian Noise**
- **Choice: Gaussian Filter**
- **Justification:**
  - Sensor electronics produce continuous, normally-distributed noise
  - Gaussian filter optimal for Gaussian noise (matched filter theory)
  - Medical images need smooth noise reduction without artifacts
  - Weighted averaging reduces random pixel variations effectively
  - Fast computation important for real-time medical imaging

**Application 2: Scanned Documents with Salt-and-Pepper Noise**
- **Choice: Median Filter**
- **Justification:**
  - Salt-and-pepper is impulsive noise (random black/white pixels)
  - Median filter removes outliers without blurring text edges
  - Preserves sharp text boundaries critical for readability
  - Gaussian would blur text and make edges fuzzy
  - Median completely eliminates isolated noise pixels
  - Better document quality for OCR applications

**General Principles:**
- **Gaussian Filter:** Best for continuous, additive noise
- **Median Filter:** Best for impulsive, sparse noise
- **Trade-off:** Gaussian is faster, Median preserves edges better
- **Rule of Thumb:** Match filter type to noise characteristics

**Marking Scheme:**
- Gaussian filter explanation (2 marks)
- Median filter explanation (2 marks)
- Application 1 choice and justification (3 marks)
- Application 2 choice and justification (3 marks)

---

### **Q9.** Histogram Equalization and Image Enhancement

You are working with two datasets: (1) Low-contrast medical CT scans where important details are hard to see, (2) Outdoor photographs taken in bright sunlight with good contrast. Your colleague suggests applying histogram equalization to both datasets. Explain when histogram equalization helps image quality and when it can hurt, using these two scenarios.

**Module:** 3 | **Week:** 7 | **Difficulty:** Moderate | **CO:** CO-3

**Expected Answer (10 marks):**

**Histogram Equalization:**
- Technique that redistributes pixel intensity values across full range
- Transforms image so histogram is approximately uniform
- Spreads out concentrated intensity values
- Increases global contrast by using full 0-255 range
- Non-linear transformation based on cumulative distribution function (CDF)
- Applied automatically without manual parameter tuning

**How Histogram Equalization Works:**
1. Compute histogram of original image
2. Calculate cumulative distribution function (CDF)
3. Map old intensities to new intensities using CDF
4. Result: Flat histogram with better spread

**Scenario 1: Low-Contrast Medical CT Scans**
- **Effect: HELPS image quality**
- **Why:**
  - CT scans often have narrow intensity range (histogram concentrated)
  - Important tissue differences hidden in small intensity variations
  - Histogram equalization spreads these values across full range
  - Subtle details (tumors, lesions) become visible
  - Radiologists can see fine tissue distinctions
  - No information loss, just better visualization

**Example:**
- Original CT: Most pixels between 100-150 (poor contrast)
- After equalization: Pixels spread 0-255 (good contrast)
- Tumor boundary that was barely visible becomes clear

**Scenario 2: Outdoor Photographs with Good Contrast**
- **Effect: HURTS image quality**
- **Why:**
  - Already has well-distributed histogram (good contrast)
  - Histogram equalization over-enhances, creating unnatural appearance
  - Background details become excessively prominent
  - May introduce posterization (banding artifacts)
  - Colors become oversaturated or washed out
  - Noise gets amplified along with signal

**Example:**
- Original photo: Natural sky gradient, balanced colors
- After equalization: Harsh sky, exaggerated clouds, unnatural tones
- Loss of natural appearance for minimal gain

**When Histogram Equalization Helps:**
- Low-contrast images (medical scans, foggy photos, underexposed)
- Narrow histogram (concentrated in small intensity range)
- Hidden details need to be revealed
- Scientific/diagnostic imaging (not aesthetic quality)

**When Histogram Equalization Hurts:**
- Already good contrast (well-distributed histogram)
- Natural appearance matters (portraits, landscapes)
- Color images (can distort color balance)
- Noisy images (amplifies noise)

**Better Alternatives for Scenario 2:**
- Adaptive Histogram Equalization (CLAHE) - local enhancement
- Gamma correction for targeted brightness adjustment
- Leave well-exposed images unchanged

**Marking Scheme:**
- Histogram equalization explanation and mechanism (2 marks)
- Scenario 1 analysis (helps quality) (3 marks)
- Scenario 2 analysis (hurts quality) (3 marks)
- General guidelines for when to use/avoid (2 marks)

---

---

## Module 4: Convolutional Neural Networks & Transfer Learning (5 questions)

### **Q4.** Batch Normalization Placement in CNNs

You are building a CNN and see two different implementations online. Implementation A places BatchNormalization AFTER the activation function (Conv2D → ReLU → BatchNorm), while Implementation B places it BEFORE activation (Conv2D → BatchNorm → ReLU). Explain which is the modern best practice and why it matters for training.

**Module:** 4 | **Week:** 11 | **Difficulty:** Moderate | **CO:** CO-4

**Expected Answer (10 marks):**

**Implementation A (Conv → ReLU → BatchNorm):**
- Old practice from early BatchNorm papers
- Normalizes after activation (post-activation values)
- ReLU zeros out negative values before normalization
- Less effective because normalization acts on already transformed values

**Implementation B (Conv → BatchNorm → ReLU):**
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

4. **Practical Impact:**
   - Implementation A: Slower convergence, may need lower learning rate
   - Implementation B: Faster training, better regularization effect

**Correct Pattern:**
```
Conv2D(32, (3,3), padding='same')  # No activation!
BatchNormalization()
Activation('relu')
```

**Marking Scheme:**
- Implementation A explanation (2 marks)
- Implementation B explanation (2 marks)
- Why B is better (4 marks)
- Practical impact (2 marks)

---

### **Q5.** Dropout Placement Strategy in CNNs

You build a CNN for CIFAR-10 classification with three convolutional blocks. Your colleague suggests placing Dropout(0.5) after every layer including after the final Dense(10) output layer. Explain what is wrong with this approach and describe the correct dropout placement strategy.

**Module:** 4 | **Week:** 11 | **Difficulty:** Easy | **CO:** CO-4

**Expected Answer (10 marks):**

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
- Identifying dropout after output error (3 marks)
- Identifying same rate everywhere error (2 marks)
- Correct progressive strategy (3 marks)
- Justification (2 marks)

---

### **Q6.** Data Augmentation for Image Classification

You train a CNN on CIFAR-10 (airplanes, cars, animals) with the following augmentation: rotation_range=180, vertical_flip=True, horizontal_flip=True. Your validation accuracy is poor despite good training accuracy. Explain what is wrong with this augmentation strategy and propose appropriate augmentation for CIFAR-10.

**Module:** 4 | **Week:** 11 | **Difficulty:** Moderate | **CO:** CO-4

**Expected Answer (10 marks):**

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

**General Augmentation Principles:**
- Augmentation should reflect real-world variations
- Don't create label-changing transformations
- Consider domain semantics (what makes sense for the objects)
- Apply ONLY to training data, NEVER to validation/test

**Marking Scheme:**
- Identifying rotation error (2 marks)
- Identifying vertical flip error (2 marks)
- Appropriate augmentation strategy (4 marks)
- Justification and principles (2 marks)

---

### **Q7.** Transfer Learning Strategy

You have a dataset of 500 medical X-ray images to classify lung diseases (4 classes). You consider two approaches: (A) Train a CNN from scratch, (B) Use transfer learning with VGG16 pre-trained on ImageNet. Explain which approach is more suitable and describe the transfer learning strategy you would use.

**Module:** 4 | **Week:** 12 (or 11 if covered) | **Difficulty:** Moderate | **CO:** CO-4

**Expected Answer (10 marks):**

**Comparing Approaches:**

**Approach A: Train from Scratch**
- Requires large dataset (typically 10,000+ images per class)
- You have only 500 images (125 per class) - too small!
- Risk of severe overfitting
- Needs extensive training time
- May not learn meaningful features with limited data

**Approach B: Transfer Learning with VGG16**
- Leverages features learned from 1.2M ImageNet images
- Works well with small datasets (hundreds to thousands)
- Pre-trained features (edges, textures, shapes) are universal
- Faster training (only fine-tune top layers)
- Better generalization

**Which is More Suitable:**
- **Transfer Learning (Approach B)** is clearly better
- 500 images is too small for training from scratch
- Pre-trained features from natural images transfer well to medical images
- Low-level features (edges, textures) are domain-agnostic

**Transfer Learning Strategy:**

**Step 1: Load Pre-trained VGG16**
```python
base_model = tf.keras.applications.VGG16(
    weights='imagenet',
    include_top=False,  # Remove ImageNet classifier
    input_shape=(224, 224, 3)
)
```

**Step 2: Freeze Base Layers Initially**
```python
base_model.trainable = False  # Freeze all VGG16 layers
```
- Preserves pre-trained features
- Only train new classification head

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
- Train only the new dense layers
- Use moderate learning rate (0.001)
- Train for 10-20 epochs until validation accuracy plateaus

**Step 5: Fine-tuning (Optional)**
- Unfreeze top few layers of VGG16
- Use very low learning rate (0.0001)
- Train for additional 5-10 epochs
- Adapts features to X-ray domain

**Why This Works:**
- Early VGG16 layers: Generic features (edges, textures) - keep frozen
- Later VGG16 layers: Specific to natural images - fine-tune
- Custom head: Learns X-ray disease patterns
- Small dataset handled well with pre-trained features

**Expected Outcome:**
- 70-85% accuracy with 500 images
- Much better than from-scratch (<60% likely)
- Reduced overfitting with frozen features

**Marking Scheme:**
- Approach comparison (2 marks)
- Choosing transfer learning with justification (2 marks)
- Transfer learning strategy steps (4 marks)
- Fine-tuning explanation (2 marks)

---

### **Q10.** CNN Parameter Calculation and Model Size

You are building a CNN for a mobile application with limited memory (max 50MB model size). You compare two architectures:

**Architecture A:**
- Conv2D(64, (7,7)) → MaxPooling → Flatten → Dense(2048) → Dense(1000)
- Input: 224×224×3

**Architecture B:**
- Conv2D(64, (3,3)) → MaxPooling → GlobalAveragePooling2D → Dense(1000)
- Input: 224×224×3

Calculate the number of parameters in the Dense layers for both architectures and explain which is more suitable for mobile deployment.

**Module:** 4 | **Week:** 10-11 | **Difficulty:** Moderate | **CO:** CO-4

**Expected Answer (10 marks):**

**Understanding CNN Parameter Calculation:**

**Convolutional Layer Parameters:**
- Formula: `(kernel_height × kernel_width × input_channels + 1) × num_filters`
- The "+1" is for bias term
- Example: Conv2D(64, (3,3)) with 3 input channels = (3×3×3 + 1) × 64 = 1,792 parameters

**Dense Layer Parameters:**
- Formula: `(input_size × output_size) + output_size`
- Input size depends on previous layer output
- Bias terms add one parameter per neuron

**Architecture A Calculation:**

**Step 1: Conv2D(64, (7,7)) parameters**
- (7 × 7 × 3 + 1) × 64 = 9,472 parameters

**Step 2: After MaxPooling (assume 2×2, stride 2)**
- Input: 224×224×3
- After Conv: 224×224×64 (with 'same' padding)
- After MaxPool: 112×112×64

**Step 3: After Flatten**
- 112 × 112 × 64 = 802,816 values

**Step 4: Dense(2048) parameters**
- (802,816 × 2,048) + 2,048 = **1,644,970,048 parameters** (1.64 billion!)
- This is the killer: massive parameter count

**Step 5: Dense(1000) parameters**
- (2,048 × 1,000) + 1,000 = 2,049,000 parameters

**Total Architecture A:** ~1.65 billion parameters
**Model Size:** 1.65B × 4 bytes (float32) = **6.6 GB** (way over 50MB limit!)

---

**Architecture B Calculation:**

**Step 1: Conv2D(64, (3,3)) parameters**
- (3 × 3 × 3 + 1) × 64 = 1,792 parameters

**Step 2: After MaxPooling**
- After Conv: 224×224×64
- After MaxPool: 112×112×64

**Step 3: GlobalAveragePooling2D**
- **Key difference:** Reduces 112×112×64 to just **64 values** (one per channel)
- No parameters, just averages each 112×112 feature map to single value
- Output: 64 values instead of 802,816!

**Step 4: Dense(1000) parameters**
- (64 × 1,000) + 1,000 = **65,000 parameters**
- Dramatically smaller than Architecture A's first Dense layer

**Total Architecture B:** ~67,000 parameters
**Model Size:** 67K × 4 bytes = **268 KB** (well under 50MB limit!)

---

**Comparison:**

| Aspect | Architecture A | Architecture B |
|--------|---------------|---------------|
| Dense layer input size | 802,816 | 64 |
| First Dense params | 1.64 billion | 65,000 |
| Total model size | 6.6 GB | 268 KB |
| Fits in 50MB? | ❌ NO | ✅ YES |
| Mobile deployment | Impossible | Excellent |

---

**Which is More Suitable for Mobile: Architecture B**

**Reasons:**

1. **Memory Efficiency:**
   - Architecture B is 25,000× smaller
   - Easily fits in mobile device memory
   - Can be downloaded over cellular network

2. **GlobalAveragePooling Advantage:**
   - Eliminates need for large Flatten → Dense connection
   - Reduces spatial dimensions (112×112) to single value per channel
   - No parameters, just computation
   - Modern CNN best practice (used in ResNet, MobileNet)

3. **Computational Efficiency:**
   - Fewer parameters = faster inference
   - Lower battery consumption on mobile
   - Faster app startup time

4. **Regularization Benefit:**
   - Fewer parameters reduce overfitting risk
   - GlobalAveragePooling acts as structural regularization
   - More generalizable model

**General Principle:**
- **Flatten → Dense** creates parameter explosion
- **GlobalAveragePooling → Dense** keeps models lightweight
- For mobile/embedded: Always prefer GlobalAveragePooling

**Marking Scheme:**
- Architecture A parameter calculation (3 marks)
- Architecture B parameter calculation (3 marks)
- Comparison and size analysis (2 marks)
- Mobile suitability justification (2 marks)

---

---

## Summary Statistics

### Distribution:
- **Module 3:** 5 questions
  - Feature extraction comparison (Q1)
  - Segmentation techniques (Q2)
  - Edge detection selection (Q3)
  - Image noise and filtering (Q8)
  - Histogram equalization (Q9)

- **Module 4:** 5 questions
  - BatchNorm placement (Q4)
  - Dropout placement strategy (Q5)
  - Data augmentation (Q6)
  - Transfer learning strategy (Q7)
  - CNN parameter calculation (Q10)

### Difficulty:
- Easy: 1 question (Q5)
- Easy-Moderate: 1 question (Q8)
- Moderate: 8 questions (Q1, Q2, Q3, Q4, Q6, Q7, Q9, Q10)
- Difficult: 0 questions

### Key Features:
- ✅ 10 marks per question (doubled from original 5 marks)
- ✅ Conceptual focus with some calculations (Q10 includes parameter calculations)
- ✅ NO code-based debugging questions
- ✅ CNN-specific applications (Q4, Q5, Q6, Q10)
- ✅ Image processing fundamentals (Q8, Q9)
- ✅ Includes Transfer Learning (Q7)
- ✅ Scenario-based explanations (similar to FT1 format)
- ✅ All questions expect detailed explanations with reasoning
- ✅ Complete marking schemes provided (totaling 10 marks each)
- ✅ Balanced coverage: 5 Module 3 + 5 Module 4

### Course Outcomes Coverage:
- CO-3 (Image Processing): 5 questions (Q1, Q2, Q3, Q8, Q9)
- CO-4 (CNNs & Transfer Learning): 5 questions (Q4, Q5, Q6, Q7, Q10)

---

## Marking Guidelines

**10-mark Question Rubric:**
- **Excellent (9-10 marks):** Complete, thorough explanation with correct reasoning, examples, and practical understanding. All components addressed comprehensively.
- **Very Good (7-8 marks):** Strong answer with most components covered well, minor gaps in depth or examples
- **Good (5-6 marks):** Satisfactory understanding with core concepts explained, but missing some details or depth
- **Satisfactory (3-4 marks):** Basic understanding shown but missing key points, incomplete explanations
- **Poor (1-2 marks):** Incomplete or partially incorrect explanation, significant gaps in understanding
- **Fail (0 marks):** Completely incorrect, irrelevant, or no answer provided

**Key Assessment Criteria:**
- Understanding of core concepts
- Ability to compare/contrast approaches
- Practical decision-making skills
- Justification with reasoning
- Clear, structured explanations

---

## Changelog

**Version 2.0 - November 11, 2025:**
- Changed marking scheme from 5 marks to 10 marks per question
- Added 3 new questions (Q8, Q9, Q10) to expand question bank
- Total questions increased from 7 to 10
- Distribution now balanced: 5 Module 3 + 5 Module 4
- Updated all marking schemes (doubled points for Q1-Q7)
- Updated marking rubric to 0-10 scale

**Version 1.0 - October 30, 2025:**
- Initial creation with 7 questions (5 marks each)
- 3 Module 3 + 4 Module 4 questions

---

**Last Updated:** November 11, 2025
**Status:** Complete - Ready for FT2 test paper generation (10-mark format)
**Total Question Bank:** 10 questions (5 from Module 3, 5 from Module 4)
