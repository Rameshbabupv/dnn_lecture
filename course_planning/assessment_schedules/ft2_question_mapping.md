# FT2 Question Mapping - Set A and Set B to Question Banks

## Set A Mapping

### Part A (MCQs):
| Set A Q# | Question Text | Maps to QB | Match? |
|----------|---------------|------------|--------|
| 1 | In a color image, how many channels are typically present in the RGB format? | MCQ-1 | ✅ EXACT |
| 2 | Which filter is most effective for removing salt-and-pepper noise? | MCQ-7 | ✅ EXACT |
| 3 | The Canny edge detector is superior to simple gradient-based methods because it: | MCQ-10 | ✅ EXACT |
| 4 | Otsu's method is used for: | MCQ-12 | ✅ EXACT |
| 5 | Local Binary Pattern (LBP) is primarily used for extracting: | MCQ-15 | ✅ EXACT |
| 6 | In a CNN, what is the primary purpose of the convolution operation? | MCQ-21 | ✅ EXACT |
| 7 | What is the main advantage of using Max Pooling over Average Pooling? | MCQ-24 | ✅ EXACT |
| 8 | In a CNN architecture, where should Dropout typically be applied? | MCQ-29 | ✅ EXACT |
| 9 | LeNet-5, the pioneering CNN architecture, was originally designed for: | MCQ-33 | ✅ EXACT |
| 10 | Transfer learning is most effective when: | MCQ-37 | ✅ EXACT |

### Part B (SAQs):
| Set A Q# | Question Summary | Maps to QB | Match? |
|----------|------------------|------------|--------|
| 11 | Feature extraction comparison (Traditional vs Deep Learning) | SAQ-1 | ✅ EXACT |
| 12 | Segmentation techniques (Thresholding vs Region-based) | SAQ-2 | ✅ EXACT |
| 13 | BatchNorm placement (Conv→BN→ReLU vs Conv→ReLU→BN) | SAQ-4 | ✅ EXACT |
| 14 | Dropout placement strategy (Progressive rates, never after output) | SAQ-5 | ✅ EXACT |
| 15 | Transfer learning strategy (VGG16 for 500 medical images) | SAQ-7 | ✅ EXACT |

---

## Set B Mapping

### Part A (MCQs):
| Set B Q# | Question Text | Maps to QB | Match? |
|----------|---------------|------------|--------|
| 1 | What is the pixel value range in a standard 8-bit grayscale image? | MCQ-2 | ✅ EXACT |
| 2 | Histogram equalization is primarily used for: | MCQ-4 | ✅ EXACT |
| 3 | Which operator detects edges based on the second derivative of the image? | MCQ-11 | ✅ EXACT |
| 4 | Watershed segmentation treats the image gradient as: | MCQ-14 | ✅ EXACT |
| 5 | In OpenCV, which function is used to convert a color image to grayscale? | MCQ-19 | ✅ EXACT |
| 6 | What happens when you use 'same' padding in a convolutional layer? | MCQ-22 | ✅ EXACT |
| 7 | GlobalAveragePooling2D differs from regular pooling by: | MCQ-25 | ✅ EXACT |
| 8 | In a CNN, Batch Normalization should be placed: | MCQ-27 | ✅ EXACT |
| 9 | What was the key innovation of AlexNet (2012) that led to its breakthrough performance? | MCQ-34 | ✅ EXACT |
| 10 | In transfer learning, "freezing" base layers means: | MCQ-38 | ✅ EXACT |

### Part B (SAQs):
| Set B Q# | Question Summary | Maps to QB | Match? |
|----------|------------------|------------|--------|
| 11 | Edge detection method selection (Sobel vs Canny for different scenarios) | SAQ-3 | ✅ EXACT |
| 12 | Segmentation techniques (Thresholding vs Region-based for medical) | SAQ-2 | ✅ EXACT |
| 13 | Dropout placement strategy (Progressive rates, never after output) | SAQ-5 | ✅ EXACT |
| 14 | Data augmentation (rotation_range=180, vertical_flip issues) | SAQ-6 | ✅ EXACT |
| 15 | BatchNorm placement (Conv→BN→ReLU vs Conv→ReLU→BN) | SAQ-4 | ✅ EXACT |

---

## Verification Summary

### Set A:
- **Part A MCQs:** 10/10 ✅ ALL MATCH
- **Part B SAQs:** 5/5 ✅ ALL MATCH
- **Total:** 15/15 ✅ PERFECT MATCH

### Set B:
- **Part A MCQs:** 10/10 ✅ ALL MATCH
- **Part B SAQs:** 5/5 ✅ ALL MATCH
- **Total:** 15/15 ✅ PERFECT MATCH

---

## QBQN Reference Format

### For MCQs:
- Format: `QBQN: MCQ-{number}`
- Example: `QBQN: MCQ-1` (refers to Q1 in ft2_mcq_bank_40_questions.md)

### For SAQs:
- Format: `QBQN: SAQ-{number}`
- Example: `QBQN: SAQ-1` (refers to Q1 in ft2_saq_bank_7_questions.md)

### Placement:
- **Position:** Right after question text, before answer options (for MCQ) or before marks row (for SAQ)

---

**Status:** ✅ All questions verified - Ready to add QBQN references
