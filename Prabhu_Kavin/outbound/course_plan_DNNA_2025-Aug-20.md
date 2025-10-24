# SRM INSTITUTE OF SCIENCE AND TECHNOLOGY				
## COLLEGE OF ENGINEERING AND TECHNOLOGY, KATTANKULATHUR CAMPUS				
### DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING				


| Course Code                     | 21CSE558T                              | L  T  P  C          | 2     1     0   3                                                             |
| ------------------------------- | -------------------------------------- | ------------------- | ----------------------------------------------------------------------------- |
| Course Title                    | Deep Neural Network Architectures      | Faculty Name        | Prof.  Ramesh Babu                                                            |
| Semester                        | ODD 2025-26                            | Email:              | [rameshbabu@srmist.edu.in](mailto:rameshbabu@srmist.edu.in)                   |
| Pre-requisite /<br>Co-Requisite | Basic Python, Linear Algebra, Calculus | Slot & Class Hours: | TBD/ day order TBD: TBD - TBD, day order TBD: TBD-TBD/ day order TBD: TBD-TBD |
| Lecture Hall                    |                                        |                     |                                                                               |

| Organisation of the course                                                                                                                                                                                                                                                 |                                                                                   |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Briefly explain how the course faculty plans to organise the course delivery:<br>Course will be delivered through a progressive learning approach starting with basic perceptron concepts and advancing to complex architectures like CNNs and object detection. Classes will combine theoretical lectures with hands-on TensorFlow/Keras implementations, tutorial sessions, Google Colab demonstrations, and practical assignments. Students will complete 15 tutorial tasks (T1-T15) covering the complete deep learning pipeline. |                                                                                   |


| Course  Outcome(s) |                                                                                   |
| ------------------ | --------------------------------------------------------------------------------- |
| 1                  | Understand and implement simple deep neural networks with appropriate activation functions |
| 2                  | Design and optimize multi-layer neural networks using advanced optimization techniques |
| 3                  | Apply deep learning techniques for image processing and feature extraction tasks |
| 4                  | Implement convolutional neural networks and transfer learning for image classification |
| 5                  | Develop object detection systems using modern architectures like YOLO and R-CNN |


| S.No | Learning Materials | Author(s) | Edition/Year | Publisher | ISBN |
|------|-------------------|-----------|--------------|-----------|------|
| 1 | Deep Learning with Python | François Chollet | 2nd Edition, 2021 | Manning Publications | 978-1-617-29681-4 |
| 2 | Deep Learning | Ian Goodfellow, Yoshua Bengio, Aaron Courville | 2016 | MIT Press | 978-0-262-03561-3 |
| 3 | Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow | Aurélien Géron | 2nd Edition, 2019 | O'Reilly | 978-1-492-03264-9 |
| 4 | Deep Learning for Computer Vision with Python | Adrian Rosebrock | 2017 | PyImageSearch | - |
| 5 | Deep Learning for Computer Vision | Rajalingappaa Shanmugamani | 2018 | Packt Publishing | 978-1-788-29564-4 |
| 6 | Deep Learning Toolbox Documentation | MathWorks | Online Resource | MathWorks | - |


## Course Topics and Schedule

### Module 1: Introduction to Deep Learning

| Unit / Module No. | Topic Name                                        | No. of Hours | Method of delivery (Multi-Select) | Assignment(s) / Activities          | Learning materials to be referred | CO  | Related PO     |
| ----------------- | ------------------------------------------------- | ------------ | --------------------------------- | ----------------------------------- | --------------------------------- | --- | -------------- |
| 1                 | Introduction to Deep Learning and Neural Networks | 1            | Lecture                           | Quiz on basic concepts              | 1, 2, 3                           | CO1 | PO1, PO2, PO12 |
| 1                 | Perceptron and Multi-Layer Perceptron             | 1            | Lecture + Tutorial                | T1: Implement basic perceptron      | 1, 2, 3                           | CO1 | PO1, PO2, PO12 |
| 1                 | Introduction to TensorFlow and Keras              | 1            | Hands-on/Lab                      | T2: TensorFlow basics               | 1, 3                              | CO1 | PO1, PO2, PO5  |
| 1                 | Activation Functions: Sigmoid, ReLU, Softmax      | 1            | Lecture                           | T3: Compare activation functions    | 1, 2, 3                           | CO1 | PO1, PO2, PO12 |
| 1                 | Loss Functions and Performance Metrics            | 1            | Lecture + Tutorial                | Assignment: Loss function analysis  | 1, 2, 3                           | CO1 | PO1, PO2, PO12 |
| 1                 | Forward and Backward Propagation                  | 1            | Lecture                           | T4: Manual backprop implementation  | 1, 2                              | CO1 | PO1, PO2, PO12 |
| 1                 | Building First Neural Network with Keras          | 1            | Hands-on/Lab                      | T5: First neural network project    | 1, 3                              | CO1 | PO1, PO2, PO5  |
| 1                 | Overfitting and Underfitting Concepts             | 1            | Lecture                           | Case study analysis                 | 1, 2, 3                           | CO1 | PO1, PO2, PO12 |
| 1                 | Model Evaluation and Validation Techniques        | 1            | Lecture + Tutorial                | T6: Cross-validation implementation | 1, 2, 3                           | CO1 | PO1, PO2, PO5  |

### Module 2: Optimization and Regularization

| 2   | Gradient Descent and its Variants                                        | 1   | Lecture | T7: SGD vs Adam comparison                   | 1, 2, 3    | CO2 | PO1, PO2, PO5 |
| --- | ----------------------------------------------------------------------- | --- | ------- | -------------------------------------------- | ---------- | --- | ------------- |
| 2   | Stochastic Gradient Descent (SGD)                                      | 1   | Lecture + Tutorial | Assignment: Optimize learning rates          | 1, 2, 3    | CO2 | PO1, PO2, PO5 |
| 2   | Advanced Optimizers: Adam, RMSprop, AdaGrad                           | 1   | Lecture | T8: Optimizer comparison study               | 1, 2, 3    | CO2 | PO1, PO2, PO5 |
| 2   | Regularization Techniques: L1, L2, Dropout                            | 1   | Lecture + Tutorial | T9: Regularization implementation            | 1, 2, 3    | CO2 | PO1, PO2, PO5 |
| 2   | Batch Normalization and Layer Normalization                           | 1   | Lecture | Assignment: Normalization effects study      | 1, 2, 3    | CO2 | PO1, PO2, PO5 |
| 2   | Early Stopping and Learning Rate Scheduling                           | 1   | Lecture + Tutorial | T10: Learning rate scheduling               | 1, 2, 3    | CO2 | PO1, PO2, PO5 |
| 2   | Weight Initialization Strategies                                       | 1   | Lecture | Activity: Weight initialization demo         | 1, 2       | CO2 | PO1, PO2, PO5 |
| 2   | Hyperparameter Tuning Techniques                                      | 1   | Lecture + Tutorial | T11: Grid search vs random search           | 1, 2, 3    | CO2 | PO1, PO2, PO5 |
| 2   | Advanced Training Techniques                                           | 1   | Lecture | Assignment: Training strategy comparison      | 1, 2, 3    | CO2 | PO1, PO2, PO5 |

### Module 3: Image Processing and Deep Neural Networks

| 3 | Digital Image Fundamentals            | 1 | Lecture | Quiz on image basics                     | 4, 5 | CO3 | PO1, PO2, PO5 |
| - | -------------------------------------------------- | - | ------- | ----------------------------------- | ---- | --- | -------------- |
| 3 | Image Processing with OpenCV                        | 1 | Hands-on/Lab | T12: OpenCV image operations            | 4, 5 | CO3 | PO1, PO2, PO5 |
| 3 | Feature Extraction Techniques                       | 1 | Lecture + Tutorial | Assignment: Feature extraction methods  | 4, 5 | CO3 | PO1, PO2, PO5 |
| 3 | Image Classification using Neural Networks          | 1 | Hands-on/Lab | T13: Image classifier implementation    | 1, 4, 5 | CO3 | PO1, PO2, PO5 |
| 3 | Data Augmentation Techniques                        | 1 | Lecture + Tutorial | Activity: Data augmentation demo        | 1, 4, 5 | CO3 | PO1, PO2, PO5 |
| 3 | Working with Image Datasets                         | 1 | Hands-on/Lab | Assignment: Dataset preprocessing       | 1, 4, 5 | CO3 | PO1, PO2, PO5 |
| 3 | Deep Learning for Computer Vision Introduction      | 1 | Lecture | Case study: Vision applications         | 1, 4, 5 | CO3 | PO1, PO2, PO5 |
| 3 | Image Preprocessing and Normalization               | 1 | Lecture + Tutorial | Activity: Preprocessing pipeline        | 1, 4, 5 | CO3 | PO1, PO2, PO5 |
| 3 | Introduction to Convolutional Operations            | 1 | Lecture | Assignment: Convolution mathematics     | 1, 2, 4 | CO3 | PO1, PO2, PO5 |

### Module 4: Convolutional Neural Networks and Transfer Learning

| 4   | Convolutional Neural Networks (CNN) Architecture | 1   | Lecture | T14: Build basic CNN                      | 1, 4, 5   | CO4 | PO1, PO2, PO5 |
| --- | -------------------------------------------------------------------------------------- | --- | ------- | ------------------------------------- | --------- | --- | -------------- |
| 4   | Convolution and Pooling Layers                                                   | 1   | Lecture + Tutorial | Assignment: Layer analysis                | 1, 4, 5   | CO4 | PO1, PO2, PO5 |
| 4   | CNN Architectures: LeNet, AlexNet, VGG                              | 1   | Lecture | Case study: Architecture evolution        | 1, 4, 5   | CO4 | PO1, PO2, PO5 |
| 4   | Advanced CNN Architectures: ResNet, DenseNet                                              | 1   | Lecture | Assignment: Architecture comparison       | 1, 4, 5   | CO4 | PO1, PO2, PO5 |
| 4   | Transfer Learning Concepts and Applications                                                                  | 1   | Lecture + Tutorial | Activity: Transfer learning demo          | 1, 4, 5   | CO4 | PO1, PO2, PO5 |
| 4   | Pre-trained Models: ImageNet, CIFAR                                                              | 1   | Hands-on/Lab | T15: Transfer learning project            | 1, 4, 5   | CO4 | PO1, PO2, PO5 |
| 4   | Fine-tuning Strategies                                                      | 1   | Lecture + Tutorial | Assignment: Fine-tuning comparison        | 1, 4, 5   | CO4 | PO1, PO2, PO5 |
| 4   | CNN Visualization and Interpretability                                                              | 1   | Lecture | Activity: Feature map visualization       | 1, 4, 5   | CO4 | PO1, PO2, PO5 |
| 4   | Practical CNN Implementation                                                         | 1   | Hands-on/Lab | Lab: Complete CNN pipeline                | 1, 4, 5   | CO4 | PO1, PO2, PO5 |

### Module 5: Object Detection

| 5 | Introduction to Object Detection          | 1 | Lecture | Quiz on detection vs classification      | 1, 4, 5 | CO5 | PO1, PO2, PO5 |
| - | ----------------------------------------------- | - | ------- | --------------------------------------- | ------- | --- | -------------- |
| 5 | R-CNN Family: R-CNN, Fast R-CNN, Faster R-CNN                | 1 | Lecture | Assignment: R-CNN evolution study       | 1, 4, 5 | CO5 | PO1, PO2, PO5 |
| 5 | YOLO (You Only Look Once) Architecture       | 1 | Lecture + Tutorial | Case study: YOLO implementation        | 1, 4, 5 | CO5 | PO1, PO2, PO5 |
| 5 | SSD (Single Shot Detector) and RetinaNet | 1 | Lecture | Assignment: Detector comparison         | 1, 4, 5 | CO5 | PO1, PO2, PO5 |
| 5 | Object Detection Metrics: mAP, IoU          | 1 | Lecture + Tutorial | Activity: Metrics calculation           | 1, 4, 5 | CO5 | PO1, PO2, PO5 |
| 5 | Practical Object Detection Implementation,          | 1 | Hands-on/Lab | Project: Object detection system        | 1, 4, 5 | CO5 | PO1, PO2, PO5 |
| 5 | Advanced Topics: Semantic Segmentation        | 1 | Lecture | Assignment: Segmentation study          | 1, 4, 5 | CO5 | PO1, PO2, PO5 |

## Grading Criteria

| Grading Criteria         |          |           |                          |                                 |     |     |     |
| ------------------------ | -------- | --------- | ------------------------ | ------------------------------- | --- | --- | --- |
| Type of assessments      | Deadline | Weightage | Due Date (if applicable) | Component                       |     |     |     |
| FT-I                     | Week-3   | 5%        | Sep 1st week             | Surprise Quiz                   |     |     |     |
| FT-II                    | Week-6   | 15%       | Sep 3rd week             | Written Test I: Modules 1 & 2   |     |     |     |
| FT-III                   | Week-12  | 15%       | Oct 4th week             | Written Test II: Modules 3 & 4  |     |     |     |
| FT-IV                    | Week-16  | 15%       | Nov 2nd week             | Tutorial Tasks Average (T1-T15) |     |     |     |
| LLT-I                    | Week-16  | 10%       | Nov 3rd week             | Final Project Implementation    |     |     |     |
| Continuous Assessment    | Total    | 60%       |                          |                                 |     |     |     |
| End Semester Examination |          | 40%       |                          |                                 |     |     |     |
|                          | Total    | 100%      |                          |                                 |     |     |     |


# Make-up Policy
## Assignments / Tutorial Tasks    
Proper academic performance depends on students doing their work not only well, but on time. Accordingly, tutorial tasks and assignments for this course must be received on the due date specified. No Make-up will be available for tutorial tasks or assignments. Late submissions will be evaluated at 25% less weight for that component for a delay of up to 24 hours after which no submissions will be accepted.

## Test
- Make-up test is usually not permitted, but in case of very genuine reasons as decided by the course faculty, course coordinator/audit professor special prior Permission is usually required to get a make-up test. It cannot be taken as a right to claim the test marks.

## Academic Integrity
- Plagiarism or cheating will result in a grade of zero for the assignment or exam. Severe cases may result in course failure.
- Code sharing between students is strictly prohibited. All implementations must be original work.

## Contact Information
- Feel free to reach out to Dr. Ramesh Babu via email for any questions regarding the course material. For urgent matters, you can visit the faculty room during office hours.
- Office Hours: TBD
- Course Support: Google Colab tutorials and additional resources will be provided via course portal.



