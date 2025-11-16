---
tags:
  - ft-2
  - formative-test-2
  - mtech
  - 21CSE558T
---
REG No:

|                                                                                      |
| ------------------------------------------------------------------------------------ |
| **M.Tech FORMATIVE TEST 2, NOVEMBER 2025**                                           |
| **Course:** 21CSE558T - Deep Neural Network Architectures                            |
|                                                                                      |
| Time: One and Half Hours                                              Max. Marks: 50 |
| Answer **ANY FIVE** Questions<br><br>**PART – A (5** **×** **10 = 50 Marks)**        |


|     |                                                                                                                                                                                                                                                                                                                                                                       |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.  | You are building an image classification system and need to choose between traditional feature extraction methods (like LBP or GLCM) and deep learning automatic feature extraction. Explain the key differences and when you would choose each approach.                                                                                                             |
| 2.  | You have a medical image dataset where you need to separate tumor regions from healthy tissue. Compare thresholding-based segmentation with region-based segmentation approaches. Explain which would be more suitable for this task and why.                                                                                                                         |
| 3.  | You are developing an image processing pipeline for two applications: (1) Medical imaging where images have Gaussian noise from sensor electronics, (2) Old scanned documents with salt-and-pepper noise from scanning artifacts. Explain which filter (Gaussian blur vs Median filter) you would use for each application and justify your choices.                  |
| 4.  | You are working with two datasets: (1) Low-contrast medical CT scans where important details are hard to see, (2) Outdoor photographs taken in bright sunlight with good contrast. Your colleague suggests applying histogram equalization to both datasets. Explain when histogram equalization helps image quality and when it can hurt, using these two scenarios. |
| 5.  | You are building a CNN and see two different implementations online. Implementation A places BatchNormalization AFTER the activation function (Conv2D → ReLU → BatchNorm), while Implementation B places it BEFORE activation (Conv2D → BatchNorm → ReLU). Explain which is the modern best practice and why it matters for training.                                 |
| 6.  | You train a CNN on CIFAR-10 (airplanes, cars, animals) with the following augmentation: rotation_range=180, vertical_flip=True, horizontal_flip=True. Your validation accuracy is poor despite good training accuracy. Explain what is wrong with this augmentation strategy and propose appropriate augmentation for CIFAR-10.                                       |
| 7.  | You have a dataset of 500 medical X-ray images to classify lung diseases (4 classes). You consider two approaches: (A) Train a CNN from scratch, (B) Use transfer learning with VGG16 pre-trained on ImageNet. Explain which approach is more suitable and describe the transfer learning strategy you would use.                                                     |

---

## Question Mapping to Question Bank

| Paper Q# | Bank Q# | Module | Topic                                    |
| -------- | ------- | ------ | ---------------------------------------- |
| 1        | Q1      | 3      | Feature Extraction Comparison            |
| 2        | Q2      | 3      | Image Segmentation Techniques            |
| 3        | Q8      | 3      | Image Noise and Filtering                |
| 4        | Q9      | 3      | Histogram Equalization                   |
| 5        | Q4      | 4      | Batch Normalization Placement            |
| 6        | Q6      | 4      | Data Augmentation for Image Classification |
| 7        | Q7      | 4      | Transfer Learning Strategy               |

**Distribution:** 4 questions from Module 3 (Image Processing), 3 questions from Module 4 (CNNs & Transfer Learning)

---

**Date:** November 14, 2025
**Venue:** [To be announced]
**Batch:** M.Tech (Data Science / AI / CSE)
**Semester:** [As applicable]
se 