# Formative Test 2 (FT2) - Set A

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

| Q.No | Question                                                                                                                                                                                                                                                               | Marks | BL  | CO   | PO  | PI Code           |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- | --- | ---- | --- | ----------------- |
| 1    | In a color image, how many channels are typically present in the RGB format?<br>**QBQN: MCQ-1**<br>a) 1<br>b) 2<br>c) 3<br>d) 4                                                                                                                                                           | 1     | L1  | CO-3 | PO1 | 21CSE558T.CO3.PO1 |
| 2    | Which filter is most effective for removing salt-and-pepper noise?<br>**QBQN: MCQ-7**<br>a) Gaussian filter<br>b) Mean filter<br>c) Median filter<br>d) Sobel filter                                                                                                                      | 1     | L2  | CO-3 | PO1 | 21CSE558T.CO3.PO1 |
| 3    | The Canny edge detector is superior to simple gradient-based methods because it:<br>**QBQN: MCQ-10**<br>a) Uses only one kernel<br>b) Performs non-maximum suppression and hysteresis thresholding<br>c) Works only on binary images<br>d) Requires no parameters                          | 1     | L2  | CO-3 | PO1 | 21CSE558T.CO3.PO1 |
| 4    | Otsu's method is used for:<br>**QBQN: MCQ-12**<br>a) Edge detection<br>b) Automatic threshold selection for binarization<br>c) Image sharpening<br>d) Noise removal                                                                                                                        | 1     | L1  | CO-3 | PO1 | 21CSE558T.CO3.PO1 |
| 5    | Local Binary Pattern (LBP) is primarily used for extracting:<br>**QBQN: MCQ-15**<br>a) Color features<br>b) Texture features<br>c) Edge features<br>d) Shape features                                                                                                                      | 1     | L1  | CO-3 | PO1 | 21CSE558T.CO3.PO1 |
| 6    | In a CNN, what is the primary purpose of the convolution operation?<br>**QBQN: MCQ-21**<br>a) Reduce image size<br>b) Extract local features from input<br>c) Classify images<br>d) Normalize pixel values                                                                                 | 1     | L2  | CO-4 | PO1 | 21CSE558T.CO4.PO1 |
| 7    | What is the main advantage of using Max Pooling over Average Pooling?<br>**QBQN: MCQ-24**<br>a) Preserves the strongest features/activations<br>b) Preserves all information equally<br>c) Requires more computation<br>d) Always produces better accuracy                                 | 1     | L2  | CO-4 | PO1 | 21CSE558T.CO4.PO1 |
| 8    | In a CNN architecture, where should Dropout typically be applied?<br>**QBQN: MCQ-29**<br>a) Before the first convolutional layer<br>b) After pooling layers and before dense layers<br>c) After every convolutional layer<br>d) After the output layer                                     | 1     | L2  | CO-4 | PO2 | 21CSE558T.CO4.PO2 |
| 9    | LeNet-5, the pioneering CNN architecture, was originally designed for:<br>**QBQN: MCQ-33**<br>a) ImageNet classification<br>b) Handwritten digit recognition<br>c) Object detection<br>d) Face recognition                                                                                 | 1     | L1  | CO-4 | PO1 | 21CSE558T.CO4.PO1 |
| 10   | Transfer learning is most effective when:<br>**QBQN: MCQ-37**<br>a) You have millions of labeled images in your target domain<br>b) Target dataset is very different from source dataset<br>c) You have limited labeled data and domains are similar<br>d) Training from scratch is faster | 1     | L2  | CO-4 | PO2 | 21CSE558T.CO4.PO2 |

---

## PART B: Short Answer Questions (Answer any 3 out of 5 × 5 = 15 marks)

**Answer ANY THREE questions. Each question carries 5 marks. Provide complete explanations with proper reasoning.**

---

| Q.No | Question | Marks | BL | CO | PO | PI Code |
|------|----------|-------|----|----|----|----|
| 11 | You are building an image classification system and need to choose between traditional feature extraction methods (like LBP or GLCM) and deep learning automatic feature extraction. Explain the key differences and when you would choose each approach.<br>**QBQN: SAQ-1** | 5 | L3 | CO-3 | PO2 | 21CSE558T.CO3.PO2 |
| 12 | You have a medical image dataset where you need to separate tumor regions from healthy tissue. Compare thresholding-based segmentation with region-based segmentation approaches. Explain which would be more suitable for this task and why.<br>**QBQN: SAQ-2** | 5 | L3 | CO-3 | PO2 | 21CSE558T.CO3.PO2 |
| 13 | You are building a CNN and see two different implementations online. Implementation A places BatchNormalization AFTER the activation function (Conv2D → ReLU → BatchNorm), while Implementation B places it BEFORE activation (Conv2D → BatchNorm → ReLU). Explain which is the modern best practice and why it matters for training.<br>**QBQN: SAQ-4** | 5 | L3 | CO-4 | PO2 | 21CSE558T.CO4.PO2 |
| 14 | You build a CNN for CIFAR-10 classification with three convolutional blocks. Your colleague suggests placing Dropout(0.5) after every layer including after the final Dense(10) output layer. Explain what is wrong with this approach and describe the correct dropout placement strategy.<br>**QBQN: SAQ-5** | 5 | L3 | CO-4 | PO2 | 21CSE558T.CO4.PO2 |
| 15 | You have a dataset of 500 medical X-ray images to classify lung diseases (4 classes). You consider two approaches: (A) Train a CNN from scratch, (B) Use transfer learning with VGG16 pre-trained on ImageNet. Explain which approach is more suitable and describe the transfer learning strategy you would use.<br>**QBQN: SAQ-7** | 5 | L4 | CO-4 | PO2 | 21CSE558T.CO4.PO2 |

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

**END OF SET A**

---

## Summary - Set A

### Part A Distribution:
- Module 3 (Image Processing): 5 MCQs (Q1-Q5)
  - Image representation (Q1)
  - Noise removal (Q2)
  - Edge detection (Q3)
  - Segmentation (Q4)
  - Feature extraction (Q5)

- Module 4 (CNNs): 5 MCQs (Q6-Q10)
  - Convolution operation (Q6)
  - Pooling layers (Q7)
  - Dropout placement (Q8)
  - Famous architectures (Q9)
  - Transfer learning (Q10)

### Part B Distribution (Answer any 3 out of 5):
- Module 3: 2 SAQs
  - Q11: Feature extraction comparison
  - Q12: Segmentation techniques
- Module 4: 3 SAQs
  - Q13: BatchNorm placement
  - Q14: Dropout placement strategy
  - Q15: Transfer learning strategy

### Difficulty:
- Easy: 3 questions
- Moderate: 7 questions

### Total: 25 marks (10 MCQ + 15 SAQ)
