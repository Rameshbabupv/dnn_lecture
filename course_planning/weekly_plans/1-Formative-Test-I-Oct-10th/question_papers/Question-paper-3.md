# Question Paper 3 - Formative Test I
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

### Q1. Which activation function can output negative values?
**Module**: 1 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: MCQ Q5

a) Sigmoid
b) ReLU
c) Tanh
d) Softmax

---

### Q2. The universal approximation theorem states that:
**Module**: 1 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: MCQ Q8

a) Any function can be approximated by linear models
b) Neural networks always converge to global minimum
c) Single hidden layer networks can approximate any continuous function
d) Deep networks are always better than shallow ones

---

### Q3. Which technique helps prevent overfitting by monitoring validation loss?
**Module**: 2 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: MCQ Q28

a) Dropout
b) Early stopping
c) Batch normalization
d) Weight decay

---

### Q4. The main problem with using zero initialization for all weights is:
**Module**: 2 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: MCQ Q38

a) Slow convergence
b) All neurons learn identical features
c) Gradient explosion
d) Memory overflow

---

### Q5. Which method combines past gradients to accelerate SGD in relevant directions?
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q33

a) AdaGrad
b) RMSprop
c) Momentum
d) Adam

---

### Q6. The softmax activation function is primarily used for:
**Module**: 1 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: MCQ Q10

a) Binary classification
b) Multi-class classification
c) Regression problems
d) Feature extraction

---

### Q7. In neural networks, the term "epoch" refers to:
**Module**: 1 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: MCQ Q14

a) One forward pass through the network
b) One backward pass through the network
c) One complete pass through the entire training dataset
d) One weight update

---

### Q8. Which regularization technique is most effective for feature selection?
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q29

a) L2 regularization (Ridge)
b) L1 regularization (Lasso)
c) Dropout
d) Batch normalization

---

### Q9. Xavier/Glorot initialization is designed to:
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q39

a) Speed up convergence
b) Prevent vanishing/exploding gradients
c) Reduce overfitting
d) Normalize inputs

---

### Q10. The bias-variance trade-off in machine learning refers to:
**Module**: 2 | **Difficulty**: Difficult | **CO**: CO-2 | **Source**: MCQ Q44

a) Training speed vs accuracy
b) Model complexity vs generalization ability
c) Memory usage vs computational speed
d) Batch size vs learning rate

---

# PART B: Short Answer Questions (Answer any 3 × 5 = 15 marks)

### Q11. You try to train a single perceptron to solve the XOR problem but it fails to converge even after many epochs. Explain why this happens and what fundamental limitation causes this failure.
**Module**: 1 | **Week**: 1-2 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: 7-SAQ Q1

---

### Q12. You build a deep neural network but use only linear activation functions. Despite having multiple layers, the network performs like a simple linear model. Explain why this happens.
**Module**: 1 | **Week**: 3 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: 7-SAQ Q2

---

### Q13. Your model achieves 99% accuracy on training data but only 65% on test data. Explain what problem this indicates and why it occurs.
**Module**: 2 | **Week**: 6 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: 7-SAQ Q4

---

### Q14. You apply L1 regularization to your network and find that many weights become exactly zero, while L2 regularization only makes weights smaller. Explain what causes this fundamental difference.
**Module**: 2 | **Week**: 6 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: 7-SAQ Q6

---

### Q15. You switch from batch gradient descent to stochastic gradient descent and observe that training becomes noisier but sometimes achieves better final results. Explain this trade-off.
**Module**: 2 | **Week**: 4 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: 7-SAQ Q5

---

## Distribution Summary:
- **MCQs**: Module 1: 4, Module 2: 6 | Easy: 4, Moderate: 5, Difficult: 1
- **SAQs**: Module 1: 2, Module 2: 3 | Easy: 3, Moderate: 2
- **Course Outcomes**: CO-1: 6, CO-2: 9
- **Total Marks**: 25 | **Time**: 50 minutes

## Updated SAQ Coverage:
- **XOR Problem**: Fundamental perceptron limitations (Week 1-2)
- **Linear Activations**: Layer composition collapse behavior (Week 3)
- **Overfitting Detection**: Training vs test performance analysis (Week 6)
- **L1 vs L2 Regularization**: Sparsity vs weight decay mathematics (Week 6)
- **SGD vs Batch GD**: Noise effects and optimization landscape (Week 4)