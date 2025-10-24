# Question Paper 2 - Formative Test I
**Course Code**: 21CSE558T
**Course Title**: Deep Neural Network Architectures
**Test Duration**: 50 minutes
**Total Marks**: 25
**Coverage**: Modules 1-2 (Weeks 1-6)

---

## Instructions:
1. Answer ALL questions in Part A (MCQs)
2. Answer ANY THREE questions from Part B (SAQs)
3. Each MCQ carries 1 mark, each SAQ carries 5 marks
4. Total marks: 25 (Part A: 10 marks + Part B: 15 marks)

---

# PART A: Multiple Choice Questions (10 × 1 = 10 marks)

### Q1. The sigmoid activation function outputs values in the range:
**Module**: 1 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: MCQ Q3

a) (-1, 1)
b) (0, 1)
c) (-∞, ∞)
d) (0, ∞)

---

### Q2. Which loss function is typically used for binary classification?
**Module**: 1 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: MCQ Q4

a) Mean Squared Error
b) Binary Cross-entropy
c) Categorical Cross-entropy
d) Hinge Loss

---

### Q3. Your model achieves 99% accuracy on training data but only 65% on test data. This indicates:
**Module**: 2 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: MCQ Q24

a) Underfitting
b) Overfitting
c) Perfect fit
d) Data leakage

---

### Q4. During training, you randomly set 30% of neurons to zero in each forward pass. This technique is called:
**Module**: 2 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: MCQ Q25

a) Batch normalization
b) Weight decay
c) Dropout
d) Early stopping

---

### Q5. The primary purpose of non-linear activation functions is to:
**Module**: 1 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: MCQ Q11

a) Speed up training
b) Reduce overfitting
c) Enable learning of complex patterns
d) Normalize outputs

---

### Q6. Which regularization technique adds the sum of absolute weights to the loss?
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q27

a) L2 regularization
b) L1 regularization
c) Dropout
d) Batch normalization

---

### Q7. In backpropagation, gradients are computed using:
**Module**: 1 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: MCQ Q19

a) Forward differentiation
b) Numerical differentiation
c) Chain rule of differentiation
d) Symbolic differentiation

---

### Q8. Which optimizer combines the benefits of momentum and adaptive learning rates?
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q36

a) SGD with momentum
b) AdaGrad
c) RMSprop
d) Adam

---

### Q9. Early stopping prevents overfitting by:
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q31

a) Reducing learning rate
b) Adding regularization terms
c) Stopping training when validation loss increases
d) Using smaller batch sizes

---

### Q10. Deep networks with many layers suffer from vanishing gradients primarily due to:
**Module**: 2 | **Difficulty**: Difficult | **CO**: CO-2 | **Source**: MCQ Q43

a) Large learning rates
b) Multiplication of small derivatives during backpropagation
c) Too much training data
d) Poor initialization

---

# PART B: Short Answer Questions (Answer any 3 × 5 = 15 marks)

### Q11. You build a deep neural network but use only linear activation functions. Despite having multiple layers, the network performs like a simple linear model. Explain why this happens.
**Module**: 1 | **Week**: 3 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: 7-SAQ Q2

---

### Q12. You replace sigmoid activations with ReLU in your deep network and observe faster training and better performance. Explain what causes this improvement.
**Module**: 1 | **Week**: 3 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: 7-SAQ Q3

---

### Q13. You apply L1 regularization to your network and find that many weights become exactly zero, while L2 regularization only makes weights smaller. Explain what causes this fundamental difference.
**Module**: 2 | **Week**: 6 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: 7-SAQ Q6

---

### Q14. You observe that a simple model underfits while a complex model overfits the same dataset. Explain this bias-variance trade-off and how it relates to model complexity.
**Module**: 2 | **Week**: 6 | **Difficulty**: Difficult | **CO**: CO-2 | **Source**: 7-SAQ Q7

---

### Q15. You switch from batch gradient descent to stochastic gradient descent and observe that training becomes noisier but sometimes achieves better final results. Explain this trade-off.
**Module**: 2 | **Week**: 4 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: 7-SAQ Q5

---

## Distribution Summary:
- **MCQs**: Module 1: 4, Module 2: 6 | Easy: 4, Moderate: 5, Difficult: 1
- **SAQs**: Module 1: 2, Module 2: 3 | Easy: 1, Moderate: 3, Difficult: 1
- **Course Outcomes**: CO-1: 6, CO-2: 9
- **Total Marks**: 25 | **Time**: 50 minutes

## Updated SAQ Coverage:
- **Linear Activations**: Why multiple layers collapse to single linear model (Week 3)
- **ReLU Benefits**: Gradient flow improvements over sigmoid (Week 3)
- **L1 vs L2 Regularization**: Sparsity vs weight decay mechanisms (Week 6)
- **Bias-Variance Trade-off**: Model complexity relationship (Week 6)
- **SGD vs Batch GD**: Noise benefits and optimization trade-offs (Week 4)