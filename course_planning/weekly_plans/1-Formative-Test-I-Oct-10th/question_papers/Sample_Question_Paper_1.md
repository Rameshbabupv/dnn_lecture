# Sample Question Paper 1 - Formative Test I
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

### Q1. The XOR problem cannot be solved by a single perceptron because:
**Module**: 1 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: MCQ Q1

a) It requires too many inputs
b) It is not linearly separable
c) It needs multiple outputs
d) It requires complex activation functions

---

### Q2. In TensorFlow, which data structure is the fundamental building block?
**Module**: 1 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: MCQ Q2

a) Array
b) Matrix
c) Tensor
d) Vector

---

### Q3. Which gradient descent variant uses the entire dataset for each update?
**Module**: 2 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: MCQ Q21

a) Stochastic Gradient Descent
b) Mini-batch Gradient Descent
c) Batch Gradient Descent
d) Adam Optimizer

---

### Q4. Overfitting occurs when:
**Module**: 2 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: MCQ Q23

a) Model performs well on both training and test data
b) Model performs poorly on training data
c) Model performs well on training but poorly on test data
d) Model cannot learn from data

---

### Q5. The mathematical representation of a perceptron output is:
**Module**: 1 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: MCQ Q16

a) y = wx + b
b) y = σ(wx + b)
c) y = w²x + b
d) y = log(wx + b)

---

### Q6. The purpose of validation set is to:
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q30

a) Train the model
b) Tune hyperparameters and monitor overfitting
c) Test final performance
d) Increase training data

---

### Q7. Which optimizer adapts the learning rate for each parameter individually?
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q35

a) SGD
b) Momentum
c) AdaGrad
d) Standard gradient descent

---

### Q8. Which problem is addressed by batch normalization?
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q40

a) Overfitting only
b) Internal covariate shift
c) Underfitting only
d) Memory optimization

---

### Q9. What is the primary advantage of ReLU over sigmoid activation?
**Module**: 1 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: MCQ Q18

a) ReLU can output negative values
b) ReLU prevents vanishing gradient problem
c) ReLU is always differentiable
d) ReLU has bounded output

---

### Q10. The learning rate finder technique helps to:
**Module**: 2 | **Difficulty**: Difficult | **CO**: CO-2 | **Source**: MCQ Q45

a) Find optimal architecture
b) Determine optimal learning rate range
c) Prevent overfitting
d) Reduce training time

---

# PART B: Short Answer Questions (Answer any 3 × 5 = 15 marks)

### Q11. You try to train a single perceptron to solve the XOR problem but it fails to converge even after many epochs. Explain why this happens and what fundamental limitation causes this failure.
**Module**: 1 | **Week**: 1-2 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: 7-SAQ Q1

---

### Q12. You build a deep neural network but use only linear activation functions. Despite having multiple layers, the network performs like a simple linear model. Explain why this happens.
**Module**: 1 | **Week**: 3 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: 7-SAQ Q2

---

### Q13. You replace sigmoid activations with ReLU in your deep network and observe faster training and better performance. Explain what causes this improvement.
**Module**: 1 | **Week**: 3 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: 7-SAQ Q3

---

### Q14. Your model achieves 99% accuracy on training data but only 65% on test data. Explain what problem this indicates and why it occurs.
**Module**: 2 | **Week**: 6 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: 7-SAQ Q4

---

### Q15. You switch from batch gradient descent to stochastic gradient descent and observe that training becomes noisier but sometimes achieves better final results. Explain this trade-off.
**Module**: 2 | **Week**: 4 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: 7-SAQ Q5

---

## Distribution Summary:
- **MCQs**: Module 1: 4, Module 2: 6 | Easy: 4, Moderate: 5, Difficult: 1
- **SAQs**: Module 1: 3, Module 2: 2 | Easy: 3, Moderate: 2
- **Course Outcomes**: CO-1: 7, CO-2: 8
- **Total Marks**: 25 | **Time**: 50 minutes

## Updated SAQ Coverage:
- **XOR Problem**: Perceptron limitations (Week 1-2)
- **Linear Activations**: Layer composition effects (Week 3)
- **ReLU vs Sigmoid**: Gradient flow improvements (Week 3)
- **Overfitting**: Training vs test performance gap (Week 6)
- **SGD vs Batch**: Optimization trade-offs (Week 4)