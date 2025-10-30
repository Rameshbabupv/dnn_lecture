# Formative Test 2 (FT2) - Set B

**Course:** 21CSE558T - Deep Neural Network Architectures
**Coverage:** Modules 3-4 (Image Processing & CNNs)
**Date:** November 14, 2025
**Duration:** TBD
**Total Marks:** 25

---

## Instructions to Students

1. This test contains **TWO parts**: Part A (MCQs) and Part B (Short Answer Questions)
2. **Part A:** Answer ALL 10 multiple choice questions (10 marks)
3. **Part B:** Answer ALL 3 short answer questions (15 marks)
4. Write your answers clearly and legibly
5. For Part B, provide complete explanations with proper reasoning

---

## PART A: Multiple Choice Questions (10 × 1 = 10 marks)

**Answer ALL questions. Choose the most appropriate option.**

---

| Q.No | Question | Marks | BL | CO | PO | PI Code |
|------|----------|-------|----|----|----|----|
| 1 | What is the pixel value range in a standard 8-bit grayscale image?<br>**QBQN: MCQ-2**<br>a) 0 to 100<br>b) 0 to 255<br>c) -128 to 127<br>d) 0 to 1 | 1 | L1 | CO-3 | PO1 | 21CSE558T.CO3.PO1 |
| 2 | Histogram equalization is primarily used for:<br>**QBQN: MCQ-4**<br>a) Edge detection<br>b) Image compression<br>c) Improving image contrast<br>d) Noise removal | 1 | L1 | CO-3 | PO1 | 21CSE558T.CO3.PO1 |
| 3 | Which operator detects edges based on the second derivative of the image?<br>**QBQN: MCQ-11**<br>a) Sobel<br>b) Prewitt<br>c) Laplacian<br>d) Roberts | 1 | L2 | CO-3 | PO1 | 21CSE558T.CO3.PO1 |
| 4 | Watershed segmentation treats the image gradient as:<br>**QBQN: MCQ-14**<br>a) A flat surface<br>b) A topographic surface with catchment basins<br>c) A binary mask<br>d) A histogram | 1 | L2 | CO-3 | PO1 | 21CSE558T.CO3.PO1 |
| 5 | In OpenCV, which function is used to convert a color image to grayscale?<br>**QBQN: MCQ-19**<br>a) cv2.gray()<br>b) cv2.cvtColor()<br>c) cv2.convert()<br>d) cv2.grayscale() | 1 | L1 | CO-3 | PO1 | 21CSE558T.CO3.PO1 |
| 6 | What happens when you use 'same' padding in a convolutional layer?<br>**QBQN: MCQ-22**<br>a) Output size is smaller than input<br>b) Output size is the same as input (with stride=1)<br>c) No padding is added<br>d) Output size is always doubled | 1 | L2 | CO-4 | PO1 | 21CSE558T.CO4.PO1 |
| 7 | GlobalAveragePooling2D differs from regular pooling by:<br>**QBQN: MCQ-25**<br>a) Using larger pool size<br>b) Reducing entire feature map to a single value per channel<br>c) Only working with grayscale images<br>d) Requiring more parameters | 1 | L2 | CO-4 | PO1 | 21CSE558T.CO4.PO1 |
| 8 | In a CNN, Batch Normalization should be placed:<br>**QBQN: MCQ-27**<br>a) Before the convolutional layer<br>b) After convolution, before activation<br>c) After activation function<br>d) Only at the output layer | 1 | L2 | CO-4 | PO2 | 21CSE558T.CO4.PO2 |
| 9 | What was the key innovation of AlexNet (2012) that led to its breakthrough performance?<br>**QBQN: MCQ-34**<br>a) Use of sigmoid activation throughout<br>b) Very shallow architecture (3 layers)<br>c) Use of ReLU activation and GPU training<br>d) No pooling layers | 1 | L2 | CO-4 | PO1 | 21CSE558T.CO4.PO1 |
| 10 | In transfer learning, "freezing" base layers means:<br>**QBQN: MCQ-38**<br>a) Deleting those layers<br>b) Setting their weights to zero<br>c) Making their weights non-trainable<br>d) Doubling their learning rate | 1 | L2 | CO-4 | PO2 | 21CSE558T.CO4.PO2 |

---

## PART B: Short Answer Questions (Answer any 3 out of 5 × 5 = 15 marks)

**Answer ANY THREE questions. Each question carries 5 marks. Provide complete explanations with proper reasoning.**

---

| Q.No | Question | Marks | BL | CO | PO | PI Code |
|------|----------|-------|----|----|----|----|
| 11 | You are processing images from two different scenarios: (1) Low-noise indoor photographs, (2) High-noise outdoor surveillance footage. Explain which edge detection method (Sobel vs. Canny) you would choose for each scenario and justify your choices.<br>**QBQN: SAQ-3** | 5 | L2 | CO-3 | PO2 | 21CSE558T.CO3.PO2 |
| 12 | You have a medical image dataset where you need to separate tumor regions from healthy tissue. Compare thresholding-based segmentation with region-based segmentation approaches. Explain which would be more suitable for this task and why.<br>**QBQN: SAQ-2** | 5 | L3 | CO-3 | PO2 | 21CSE558T.CO3.PO2 |
| 13 | You build a CNN for CIFAR-10 classification with three convolutional blocks. Your colleague suggests placing Dropout(0.5) after every layer including after the final Dense(10) output layer. Explain what is wrong with this approach and describe the correct dropout placement strategy.<br>**QBQN: SAQ-5** | 5 | L3 | CO-4 | PO2 | 21CSE558T.CO4.PO2 |
| 14 | You train a CNN on CIFAR-10 (airplanes, cars, animals) with the following augmentation: rotation_range=180, vertical_flip=True, horizontal_flip=True. Your validation accuracy is poor despite good training accuracy. Explain what is wrong with this augmentation strategy and propose appropriate augmentation for CIFAR-10.<br>**QBQN: SAQ-6** | 5 | L4 | CO-4 | PO2 | 21CSE558T.CO4.PO2 |
| 15 | You are building a CNN and see two different implementations online. Implementation A places BatchNormalization AFTER the activation function (Conv2D → ReLU → BatchNorm), while Implementation B places it BEFORE activation (Conv2D → BatchNorm → ReLU). Explain which is the modern best practice and why it matters for training.<br>**QBQN: SAQ-4** | 5 | L3 | CO-4 | PO2 | 21CSE558T.CO4.PO2 |

---

## Bloom's Taxonomy (BL) Levels:
- **L1:** Remembering (Knowledge)
- **L2:** Understanding (Comprehension)
- **L3:** Applying (Application)
- **L4:** Analyzing (Analysis)

## Course Outcomes (CO):
- **CO-3:** Implement deep learning for image processing applications
- **CO-4:** Design and implement CNN architectures and transfer learning

## Program Outcomes (PO):
- **PO1:** Engineering Knowledge
- **PO2:** Problem Analysis

---

**END OF SET B**

---

## Summary - Set B

### Part A Distribution:
- Module 3 (Image Processing): 5 MCQs (Q1-Q5)
  - Image representation (Q1)
  - Image enhancement (Q2)
  - Edge detection (Q3)
  - Segmentation (Q4)
  - OpenCV operations (Q5)

- Module 4 (CNNs): 5 MCQs (Q6-Q10)
  - Convolution padding (Q6)
  - Pooling layers (Q7)
  - BatchNorm placement (Q8)
  - Famous architectures (Q9)
  - Transfer learning (Q10)

### Part B Distribution (Answer any 3 out of 5):
- Module 3: 2 SAQs
  - Q11: Edge detection method selection
  - Q12: Segmentation techniques
- Module 4: 3 SAQs
  - Q13: Dropout placement strategy
  - Q14: Data augmentation
  - Q15: BatchNorm placement

### Difficulty:
- Easy: 3 questions
- Moderate: 7 questions

### Total: 25 marks (10 MCQ + 15 SAQ)

---

## Comparison: Set A vs Set B

### Set A Focus:
- Module 3: Basic concepts (channels, noise removal, edge detection, thresholding, LBP)
- Module 4: Convolution, Max pooling, Dropout placement, LeNet-5, Transfer learning effectiveness
- SAQs: Feature extraction comparison, BatchNorm placement, Transfer learning strategy

### Set B Focus:
- Module 3: Fundamentals (pixel values, histogram, Laplacian, watershed, OpenCV)
- Module 4: Padding, GlobalAvgPool, BatchNorm placement, AlexNet, Freezing layers
- SAQs: Segmentation comparison, Dropout strategy, Data augmentation

### Both Sets Cover:
- ✅ Balanced Module 3-4 distribution (5-5 MCQs)
- ✅ CNN-specific applications (BatchNorm, Dropout, Augmentation)
- ✅ Transfer learning concepts
- ✅ No overlap with FT1 topics
- ✅ Conceptual focus, light calculations
- ✅ Complete CO-PO-PI mapping
