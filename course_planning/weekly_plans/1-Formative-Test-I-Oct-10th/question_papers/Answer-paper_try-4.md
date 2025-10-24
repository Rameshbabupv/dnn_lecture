# Answer Key - Question Paper 4 - Formative Test I
**Course Code**: 21CSE558T
**Course Title**: Deep Neural Network Architectures
**Test Duration**: 50 minutes
**Total Marks**: 25
**Coverage**: Modules 1-2 (Weeks 1-6)

---

# PART A: Multiple Choice Questions (10 × 1 = 10 marks)

### Q1. The ReLU activation function is defined as:
**Module**: 1 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: MCQ Q6
**Answer**: a) max(0, x)

---

### Q2. Which component is NOT part of the basic perceptron model?
**Module**: 1 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: MCQ Q15
**Answer**: d) Convolutional layer

---

### Q3. Mini-batch gradient descent is preferred over batch gradient descent because:
**Module**: 2 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: MCQ Q22
**Answer**: b) It balances computational efficiency and gradient accuracy

---

### Q4. L2 regularization (weight decay) helps prevent overfitting by:
**Module**: 2 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: MCQ Q26
**Answer**: b) Adding penalty for large weights

---

### Q5. A cost function differs from a loss function in that:
**Module**: 1 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: MCQ Q12
**Answer**: b) Cost is average loss over entire dataset

---

### Q6. The vanishing gradient problem in deep networks is primarily caused by:
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q32
**Answer**: b) Multiplication of small derivatives during backpropagation

---

### Q7. What is the main advantage of using TensorFlow over implementing neural networks from scratch?
**Module**: 1 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: MCQ Q17
**Answer**: b) Automatic differentiation and GPU support

---

### Q8. Which initialization strategy is specifically designed for ReLU activations?
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q37
**Answer**: c) He initialization

---

### Q9. Batch normalization helps training by:
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q41
**Answer**: b) Normalizing layer inputs to reduce internal covariate shift

---

### Q10. In the context of neural networks, what does "generalization" refer to?
**Module**: 2 | **Difficulty**: Difficult | **CO**: CO-2 | **Source**: MCQ Q42
**Answer**: b) Performance on unseen test data

---

# PART B: Short Answer Questions (Answer any 3 × 5 = 15 marks)

### Q11. You try to train a single perceptron to solve the XOR problem but it fails to converge even after many epochs. Explain why this happens and what fundamental limitation causes this failure.
**Module**: 1 | **Week**: 1-2 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: 7-SAQ Q1

**Answer**: The XOR problem requires separating points (0,1) and (1,0) from (0,0) and (1,1), which cannot be achieved with a single straight line. A single perceptron can only create linear decision boundaries, but XOR is not linearly separable and requires a non-linear boundary that a single perceptron cannot produce.

---

### Q12. You replace sigmoid activations with ReLU in your deep network and observe faster training and better performance. Explain what causes this improvement.
**Module**: 1 | **Week**: 3 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: 7-SAQ Q3

**Answer**: Sigmoid activations have a maximum derivative of 0.25, causing gradients to vanish exponentially in deep networks, which slows learning in early layers. ReLU has a derivative of 1 for positive inputs, allowing gradients to flow unchanged through many layers, enabling faster training and better performance in deep networks.

---

### Q13. Your model achieves 99% accuracy on training data but only 65% on test data. Explain what problem this indicates and why it occurs.
**Module**: 2 | **Week**: 6 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: 7-SAQ Q4

**Answer**: This indicates overfitting, where the model has memorized the training data rather than learning generalizable patterns. The large gap between training and test accuracy occurs because the model has learned to fit noise and specific details in the training set that don't represent the underlying data distribution, causing poor performance on unseen data.

---

### Q14. You observe that a simple model underfits while a complex model overfits the same dataset. Explain this bias-variance trade-off and how it relates to model complexity.
**Module**: 2 | **Week**: 6 | **Difficulty**: Difficult | **CO**: CO-2 | **Source**: 7-SAQ Q7

**Answer**: Simple models have high bias (underfitting) because they lack the capacity to capture complex patterns, while complex models have high variance (overfitting) because they can memorize training data noise. The bias-variance trade-off shows that as model complexity increases, bias decreases but variance increases, requiring careful balance to achieve optimal generalization performance on unseen data.

---

### Q15. You switch from batch gradient descent to stochastic gradient descent and observe that training becomes noisier but sometimes achieves better final results. Explain this trade-off.
**Module**: 2 | **Week**: 4 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: 7-SAQ Q5

**Answer**: Stochastic gradient descent updates weights after each sample, creating noisy gradients that cause oscillations during training. However, this noise can be beneficial because it helps the optimizer escape local minima and explore the loss landscape more thoroughly, potentially finding better global solutions that batch gradient descent might miss due to its smoother but more constrained path.

---

## Answer Key Summary:

### Part A (MCQs) - 10 marks:
1. a) max(0, x)
2. d) Convolutional layer
3. b) It balances computational efficiency and gradient accuracy
4. b) Adding penalty for large weights
5. b) Cost is average loss over entire dataset
6. b) Multiplication of small derivatives during backpropagation
7. b) Automatic differentiation and GPU support
8. c) He initialization
9. b) Normalizing layer inputs to reduce internal covariate shift
10. b) Performance on unseen test data

### Part B (SAQs) - 15 marks:
- Q11: XOR linear separability limitation (5 marks)
- Q12: ReLU vs sigmoid gradient flow (5 marks)
- Q13: Overfitting identification and causes (5 marks)
- Q14: Bias-variance trade-off with model complexity (5 marks)
- Q15: SGD noise benefits vs batch GD stability (5 marks)

**Total Marks**: 25
