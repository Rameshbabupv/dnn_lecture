# Week 9 Plan - Module 3: Feature Extraction & Classification

**Duration:** Oct 13-17, 2025
**Focus:** Feature Extraction from Images
**Learning Outcome:** CO-3 - Apply DNNs in image processing (Module 3 Completion)

## Course Progress Overview

### Where We Are (Week 9/15 - 60% Complete)
- **Course Phase:** Completing Image Processing Foundation
- **Module:** 3 of 5 (Image Processing & DNNs - FINAL WEEK)
- **Tutorial:** T9 of T15 (60% of practical work)

### Journey So Far
- ✅ **Weeks 1-3:** Neural network fundamentals (Module 1)
- ✅ **Weeks 4-6:** Optimization & regularization (Module 2)
- ✅ **Week 7:** Image enhancement & edge detection (T7)
- ✅ **Week 8:** Image segmentation & ROI extraction (T8)
- 🎯 **Week 9:** **Feature extraction & classification (T9)** ← **YOU ARE HERE**
- ⏳ **Weeks 10-12:** CNNs & transfer learning (Module 4)
- ⏳ **Weeks 13-15:** Object detection (Module 5)

### Strategic Position
**Week 9 is the COMPLETION of Module 3** and serves as:
- **Final image processing week** → Manual feature extraction mastery
- **Bridge to CNNs** → Understanding what CNNs automate
- **Pre-Unit Test 2** → Critical revision period (Oct 31 exam)
- **Feature engineering** → Traditional ML vs Deep Learning comparison

### Assessment Context
- **Unit Test 1:** ✅ Completed (Sep 19) - Modules 1-2
- **Preparing for:** Unit Test 2 (Oct 31) - Modules 3-4
- **Tutorial T9:** Feature extraction practical assessment

## Week 9 Schedule

### Day Order 3 - October 15, 2025 (2 Hours) - LECTURE
**Topic: Feature Extraction from Images**

#### Session Breakdown

**Part 1: Shape Features (40 minutes)**
- Contour analysis and properties
- Geometric features:
  - Area and perimeter
  - Circularity and compactness
  - Aspect ratio
- Moments and centroids:
  - Spatial moments (m00, m10, m01)
  - Central moments
  - Hu moments (rotation invariant)
- Shape descriptors:
  - Bounding boxes (axis-aligned, rotated)
  - Convex hulls
  - Solidity and extent

**Part 2: Color Features (40 minutes)**
- Color representation and spaces:
  - RGB color model
  - HSV for perceptual features
  - LAB for uniform color space
- Color histogram features:
  - 1D histograms per channel
  - 2D and 3D color histograms
  - Normalized histograms
- Statistical color features:
  - Mean, standard deviation
  - Skewness and kurtosis
  - Color moments
- Dominant color extraction:
  - K-means clustering approach
  - Color quantization techniques

**Part 3: Texture Features (30 minutes)**
- Texture definition and importance
- Statistical texture descriptors:
  - GLCM (Gray Level Co-occurrence Matrix)
  - Contrast, correlation, energy, homogeneity
- Local texture patterns:
  - LBP (Local Binary Patterns)
  - Rotation invariant LBP
- Filter-based approaches:
  - Gabor filters for orientation
  - Haralick features
  - Wavelet transforms

**Part 4: Integration with Deep Learning (10 minutes)**
- Feature vectors for classification:
  - Combining shape, color, texture
  - Feature normalization
  - Dimensionality considerations
- Traditional ML pipeline:
  - Manual feature extraction
  - Classical classifiers (SVM, Random Forest)
- Comparison with CNNs:
  - Automatic feature learning
  - Why CNNs replace manual extraction
  - When to use each approach
- Preview: Module 4 transition

### Day Order 4 - October 16, 2025 (1 Hour) - TUTORIAL T9
**Tutorial: Feature Extraction from Images Using OpenCV**

#### Tutorial Structure

**Part 1: Shape Feature Extraction (25 minutes)**
- **Objective:** Extract geometric and moment features from objects
- **Activities:**
  - Load and preprocess sample images
  - Detect contours using `cv2.findContours()`
  - Calculate shape features:
    - Area: `cv2.contourArea()`
    - Perimeter: `cv2.arcLength()`
    - Circularity: 4π × area / perimeter²
    - Hu moments: `cv2.HuMoments()`
  - Extract bounding features:
    - Axis-aligned: `cv2.boundingRect()`
    - Rotated: `cv2.minAreaRect()`
    - Convex hull: `cv2.convexHull()`
- **Exercise 1:** Extract 10 shape features from object dataset

**Part 2: Color Feature Extraction (20 minutes)**
- **Objective:** Extract color-based descriptors
- **Activities:**
  - Compute color histograms:
    - RGB histograms: `cv2.calcHist()`
    - HSV conversion: `cv2.cvtColor()`
  - Calculate color moments:
    - Mean: `np.mean()`
    - Standard deviation: `np.std()`
    - Skewness: `scipy.stats.skew()`
  - Extract dominant colors:
    - K-means clustering: `cv2.kmeans()`
- **Exercise 2:** Build 18-dimensional color feature vector

**Part 3: Classification Pipeline (15 minutes)**
- **Objective:** Use extracted features for simple classification
- **Activities:**
  - Combine shape + color features
  - Normalize feature vectors
  - Train simple classifier:
    - Split data (train/test)
    - Use scikit-learn (SVM or Random Forest)
    - Evaluate accuracy
  - Visualize results
- **Integration Exercise:** Complete classification pipeline

#### Student Deliverables
1. **Code Files:**
   - `shape_features.py` - Shape extraction functions
   - `color_features.py` - Color extraction functions
   - `classifier_pipeline.py` - Complete classification system
2. **Output:**
   - Feature CSV file (all extracted features)
   - Classification report (accuracy, confusion matrix)
   - Visualization plots
3. **Documentation:**
   - Feature description document
   - Results analysis (1-2 paragraphs)

## Core Topics Summary

### Module 3 Coverage (Syllabus Alignment)

From Syllabus - Module 3 Topics:
- ✅ **Fundamentals of Image Processing:**
  - ✅ Image Enhancement (Week 7)
  - ✅ Noise Removal Techniques (Week 7)
  - ✅ Edge Detection Techniques (Week 7)
  - ✅ Image Segmentation (Week 8)
  - ✅ ROI Segmentation (Week 8)
  - ✅ Morphological Processing (Week 8)
- ✅ **Feature Extraction from Images:**
  - 🎯 Shape (Week 9 - do3)
  - 🎯 Colour (Week 9 - do3)
  - 🎯 Texture (Week 9 - do3)
- ✅ **Unstructured Image Structural Data** (Week 9)
- ✅ **Image Classification from Extracted Features** (Week 9 - do4)
- ✅ **Various Applications of Computer Vision** (Throughout Module 3)

### Module 3 Tutorials (Complete)
- ✅ **T7:** Building Programs on Image Processing Using OpenCV
- ✅ **T8:** Image Segmentation Using OpenCV
- 🎯 **T9:** Feature Extraction from Image Using OpenCV

## Key OpenCV & Python Functions (T9)

### Shape Features
```python
cv2.findContours()         # Contour detection
cv2.contourArea()          # Area calculation
cv2.arcLength()            # Perimeter calculation
cv2.moments()              # Image moments
cv2.HuMoments()            # Hu moment invariants
cv2.boundingRect()         # Axis-aligned bounding box
cv2.minAreaRect()          # Rotated bounding box
cv2.convexHull()           # Convex hull
cv2.contourArea()          # For solidity calculation
```

### Color Features
```python
cv2.cvtColor()             # Color space conversion
cv2.calcHist()             # Histogram calculation
np.mean(), np.std()        # Statistical moments
cv2.kmeans()               # K-means clustering
cv2.normalize()            # Histogram normalization
```

### Texture Features (Optional Advanced)
```python
cv2.calcHist()             # For GLCM computation
np.gradient()              # Gradient calculation
cv2.filter2D()             # Gabor filter application
```

### Classification (scikit-learn)
```python
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
```

## Learning Progression

### Pre-Week 9 Knowledge (Prerequisites)
- Image loading and preprocessing (Week 7)
- Segmentation and ROI extraction (Week 8)
- Basic OpenCV operations
- NumPy array manipulation

### Week 9 Learning Outcomes
By the end of Week 9, students will be able to:
1. **Extract shape features** from segmented objects
2. **Compute color descriptors** from images
3. **Calculate texture features** using statistical methods
4. **Build feature vectors** combining multiple descriptors
5. **Implement classification** using extracted features
6. **Compare manual vs automatic** feature learning (CNN preview)

### Post-Week 9 (Module 4 Preview)
- Understanding why CNNs automate feature extraction
- Appreciation for deep learning advantages
- Transition to convolutional architectures

## Assessment Focus

### Tutorial T9 Assessment Criteria (100 Points)
- **Shape Feature Extraction (30 points):**
  - Correct contour detection (10)
  - Accurate geometric calculations (10)
  - Proper moment computation (10)
- **Color Feature Extraction (30 points):**
  - Histogram computation (10)
  - Color moments calculation (10)
  - Dominant color extraction (10)
- **Classification Pipeline (30 points):**
  - Feature vector construction (10)
  - Classifier implementation (10)
  - Accuracy and evaluation (10)
- **Code Quality & Documentation (10 points):**
  - Clean, readable code (5)
  - Proper documentation (5)
- **Bonus (10 points):**
  - Texture features implementation (5)
  - Advanced classification techniques (5)

### Unit Test 2 Preparation (Oct 31)
Week 9 content is critical for:
- Feature extraction theory questions
- OpenCV function usage
- Classification pipeline understanding
- Comparison: Traditional ML vs Deep Learning

## Technical Requirements

### Software Environment
- **Primary:** Google Colab (recommended)
- **Local:** Python 3.8+ with virtual environment
- **Virtual Env Path:** `labs/srnenv/`

### Required Libraries
```python
import cv2                  # OpenCV 4.x
import numpy as np          # NumPy
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
```

### Dataset Requirements
- Sample images with labeled objects (5-10 classes)
- Preprocessed and segmented images from Week 8
- Minimum 50 images for classification exercise

## Connection to Course Objectives

### Course Learning Rationale (CLR-3)
✅ **CLR-3:** Explore the application of deep neural networks in image processing
- Manual feature extraction understanding
- Preparation for automatic feature learning (CNNs)
- Real-world computer vision applications

### Course Outcome (CO-3)
✅ **CO-3:** Apply deep neural networks in image processing problems
- **Week 7:** Preprocessing pipeline (enhancement, noise, edges)
- **Week 8:** Segmentation and region extraction
- **Week 9:** Feature extraction and classification
- **Complete:** Full traditional computer vision pipeline

### Bridge to CO-4 (Next Module)
Prepares for **CO-4:** Implement convolutional neural networks
- Understanding what CNNs automate
- Appreciation for learned features vs handcrafted
- Foundation for CNN architecture comprehension

## Next Week Preview

### Week 10 - Module 4 Begins: Convolutional Neural Networks
- **Topic:** Introduction to CNNs and Biological Motivation
- **Tutorial T10:** Perform Classification using CNN in Keras
- **Key Transition:** From manual to automatic feature learning
- **Learning Outcome:** CO-4 - Implement CNNs

### Module 3 → Module 4 Bridge
Week 9 completion provides:
- Deep understanding of image features
- Context for CNN convolutional layers (automatic feature extraction)
- Appreciation for pooling layers (spatial hierarchy)
- Foundation for understanding fully connected layers (classification)

## Key Takeaways for Week 9

### For Students
1. Master manual feature extraction techniques
2. Understand shape, color, and texture descriptors
3. Build complete classification pipeline
4. Prepare for automatic feature learning with CNNs

### For Instructor
1. Emphasize the "why" behind each feature type
2. Connect manual extraction to CNN automation
3. Use visual examples throughout
4. Ensure students complete T9 for practical assessment

### For Course Progression
1. **Completes Module 3** - All image processing fundamentals covered
2. **Prepares for Module 4** - CNN concepts will build on this foundation
3. **Assessment Ready** - Students prepared for Unit Test 2
4. **Practical Skills** - T9 submission for continuous assessment

---

## Files to Create for Week 9

### Do3 - October 15 (Lecture)
- [ ] Comprehensive lecture notes (shape, color, texture features)
- [ ] Lecture slides with visual examples
- [ ] Code demonstrations for each feature type
- [ ] Handout: Feature extraction formulas and OpenCV functions

### Do4 - October 16 (Tutorial T9)
- [ ] Tutorial T9 comprehensive guide
- [ ] Jupyter notebook with exercises
- [ ] Sample dataset with labeled images
- [ ] Starter code templates
- [ ] Quick reference cheat sheet
- [ ] Assessment rubric
- [ ] Solution code (for instructor)

### Supporting Materials
- [ ] Feature extraction pipeline diagram
- [ ] Comparison chart: Manual vs CNN features
- [ ] Unit Test 2 practice questions (Module 3 section)

---

**Status:** Planning Complete ✅
**Next Action:** Create detailed materials for do3 and do4
**Timeline:** Materials needed by Oct 14 (day before delivery)
