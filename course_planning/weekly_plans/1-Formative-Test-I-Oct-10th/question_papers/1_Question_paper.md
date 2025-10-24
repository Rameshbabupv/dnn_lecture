# Question Paper 1 - Formative Test I
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

### Q2. Which activation function is most commonly used in hidden layers of modern deep networks?
**Module**: 1 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: MCQ Q11

a) Sigmoid  
b) Tanh  
c) ReLU  
d) Linear

---

### Q3. The derivative of ReLU function for positive inputs is:
**Module**: 1 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: MCQ Q13

a) 0  
b) 1  
c) x  
d) e^x

---

### Q4. The main advantage of using ReLU over sigmoid activation is:
**Module**: 1 | **Difficulty**: Difficult | **CO**: CO-1 | **Source**: MCQ Q18

a) Smoother gradients  
b) Bounded output range  
c) Mitigation of vanishing gradient problem  
d) Better for binary classification

---

### Q5. Which gradient descent variant uses the entire dataset for each update?
**Module**: 2 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: MCQ Q21

a) Stochastic Gradient Descent  
b) Mini-batch Gradient Descent  
c) Batch Gradient Descent  
d) Adam Optimizer

---

### Q6. Overfitting occurs when:
**Module**: 2 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: MCQ Q23

a) Model performs well on both training and test data  
b) Model performs poorly on training data  
c) Model performs well on training but poorly on test data  
d) Model cannot learn from data

---

### Q7. The purpose of validation set is to:
**Module**: 2 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: MCQ Q30

a) Train the model  
b) Tune hyperparameters and monitor overfitting  
c) Test final performance  
d) Increase training data

---

### Q8. The Adam optimizer combines:
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q32

a) Momentum and RMSprop  
b) SGD and batch gradient descent  
c) L1 and L2 regularization  
d) Dropout and batch normalization

---

### Q9. Batch normalization is applied:
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q37

a) Only at the input layer  
b) Only at the output layer  
c) Between layers during training  
d) Only during testing

---

### Q10. Weight initialization using Xavier/Glorot method aims to:
**Module**: 2 | **Difficulty**: Difficult | **CO**: CO-2 | **Source**: MCQ Q41

a) Minimize the loss function  
b) Maintain gradient flow through layers  
c) Increase computational speed  
d) Reduce memory usage

---

# PART B: Short Answer Questions (Answer any 3 × 5 = 15 marks)

### Q11. You try to train a single perceptron to solve the XOR problem but it fails to converge even after many epochs. Explain why this happens and what fundamental limitation causes this failure.
**Module**: 1 | **Week**: 1-2 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: 7-SAQ Q1

---

### Q12. You replace sigmoid activations with ReLU in your deep network and observe faster training and better performance. Explain what causes this improvement.
**Module**: 1 | **Week**: 3 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: 7-SAQ Q3

---

### Q13. Your model achieves 99% accuracy on training data but only 65% on test data. Explain what problem this indicates and why it occurs.
**Module**: 2 | **Week**: 6 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: 7-SAQ Q4

---

### Q14. You observe that a simple model underfits while a complex model overfits the same dataset. Explain this bias-variance trade-off and how it relates to model complexity.
**Module**: 2 | **Week**: 6 | **Difficulty**: Difficult | **CO**: CO-2 | **Source**: 7-SAQ Q7

---

### Q15. You switch from batch gradient descent to stochastic gradient descent and observe that training becomes noisier but sometimes achieves better final results. Explain this trade-off.
**Module**: 2 | **Week**: 4 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: 7-SAQ Q5

---

## Distribution Summary:
- **MCQs**: Module 1: 4, Module 2: 6 | Easy: 4, Moderate: 4, Difficult: 2
- **SAQs**: Module 1: 2, Module 2: 3 | Easy: 2, Moderate: 2, Difficult: 1
- **Course Outcomes**: CO-1: 6, CO-2: 9
- **Total Marks**: 25 | **Time**: 50 minutes
