# Question Paper 4 - Formative Test I
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

### Q1. The ReLU activation function is defined as:
**Module**: 1 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: MCQ Q6

a) max(0, x)
b) 1/(1 + e^(-x))
c) tanh(x)
d) x/(1 + |x|)

---

### Q2. Which component is NOT part of the basic perceptron model?
**Module**: 1 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: MCQ Q15

a) Weights
b) Bias
c) Activation function
d) Convolutional layer

---

### Q3. Mini-batch gradient descent is preferred over batch gradient descent because:
**Module**: 2 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: MCQ Q22

a) It always finds better solutions
b) It balances computational efficiency and gradient accuracy
c) It requires no hyperparameter tuning
d) It prevents overfitting

---

### Q4. L2 regularization (weight decay) helps prevent overfitting by:
**Module**: 2 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: MCQ Q26

a) Increasing the learning rate
b) Adding penalty for large weights
c) Reducing the number of epochs
d) Using more training data

---

### Q5. A cost function differs from a loss function in that:
**Module**: 1 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: MCQ Q12

a) Cost is for regression, loss is for classification
b) Cost is average loss over entire dataset
c) Cost is always squared, loss can be any function
d) There is no difference

---

### Q6. The vanishing gradient problem in deep networks is primarily caused by:
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q32

a) Too small learning rates
b) Multiplication of small derivatives during backpropagation
c) Large batch sizes
d) Poor data normalization

---

### Q7. What is the main advantage of using TensorFlow over implementing neural networks from scratch?
**Module**: 1 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: MCQ Q17

a) Better accuracy
b) Automatic differentiation and GPU support
c) Smaller model size
d) Faster training guaranteed

---

### Q8. Which initialization strategy is specifically designed for ReLU activations?
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q37

a) Zero initialization
b) Xavier/Glorot initialization
c) He initialization
d) Random uniform initialization

---

### Q9. Batch normalization helps training by:
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q41

a) Reducing the number of parameters
b) Normalizing layer inputs to reduce internal covariate shift
c) Increasing the learning rate automatically
d) Preventing gradient descent

---

### Q10. In the context of neural networks, what does "generalization" refer to?
**Module**: 2 | **Difficulty**: Difficult | **CO**: CO-2 | **Source**: MCQ Q42

a) Ability to work with different programming languages
b) Performance on unseen test data
c) Ability to handle different data types
d) Scalability to larger datasets

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
- **MCQs**: Module 1: 4, Module 2: 6 | Easy: 4, Moderate: 5, Difficult: 1
- **SAQs**: Module 1: 2, Module 2: 3 | Easy: 2, Moderate: 2, Difficult: 1
- **Course Outcomes**: CO-1: 6, CO-2: 9
- **Total Marks**: 25 | **Time**: 50 minutes

## Updated SAQ Coverage:
- **XOR Problem**: Perceptron linear separability limitations (Week 1-2)
- **ReLU Benefits**: Vanishing gradient solutions vs sigmoid (Week 3)
- **Overfitting Recognition**: Training-test performance gaps (Week 6)
- **Bias-Variance Trade-off**: Model complexity and generalization (Week 6)
- **SGD vs Batch GD**: Noise advantages in optimization (Week 4)