# 1-Mark Multiple Choice Questions
**Course Code**: 21CSE558T
**Course Title**: Deep Neural Network Architectures
**Test**: Formative Assessment I
**Coverage**: Module 1 & Module 2
**Total Questions**: 45 (1-mark MCQ only)

---

# PART A: 1-MARK QUESTIONS (45 Questions)

## Module 1: Introduction to Deep Learning (20 questions)

### Q1. The XOR problem cannot be solved by a single perceptron because:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) It requires too many inputs
b) It is not linearly separable
c) It needs multiple outputs
d) It requires complex activation functions

**Answer**: b) It is not linearly separable

---

### Q2. In TensorFlow, which data structure is the fundamental building block?
**PLO**: PLO-2 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI2 | **Marks**: 1 | **Difficulty**: Easy

a) Array
b) Matrix
c) Tensor
d) Vector

**Answer**: c) Tensor

---

### Q3. The sigmoid activation function outputs values in the range:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) (-1, 1)
b) (0, 1)
c) (-∞, ∞)
d) (0, ∞)

**Answer**: b) (0, 1)

---

### Q4. Which loss function is typically used for binary classification?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI4 | **Marks**: 1 | **Difficulty**: Easy

a) Mean Squared Error
b) Mean Absolute Error
c) Binary Cross-entropy
d) Categorical Cross-entropy

**Answer**: c) Binary Cross-entropy

---

### Q5. Which activation function can output negative values?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) Sigmoid
b) ReLU
c) Tanh
d) Softmax

**Answer**: c) Tanh

---

### Q6. The perceptron learning algorithm can solve:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) Only linearly separable problems
b) Any classification problem
c) Only regression problems
d) Non-linear problems directly

**Answer**: a) Only linearly separable problems

---

### Q7. Which TensorFlow function is used to create a tensor filled with zeros?
**PLO**: PLO-2 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI2 | **Marks**: 1 | **Difficulty**: Easy

a) tf.ones()
b) tf.zeros()
c) tf.empty()
d) tf.fill()

**Answer**: b) tf.zeros()

---

### Q8. Which type of neural network layer performs weighted sum of inputs?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) Activation layer
b) Dense layer
c) Pooling layer
d) Dropout layer

**Answer**: b) Dense layer

---

### Q9. In forward propagation, the output of each layer is:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI4 | **Marks**: 1 | **Difficulty**: Easy

a) Input to the previous layer
b) Input to the next layer
c) Stored for backward pass only
d) Discarded immediately

**Answer**: b) Input to the next layer

---

### Q10. The softmax activation function is primarily used for:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) Binary classification
b) Multi-class classification
c) Regression problems
d) Feature extraction

**Answer**: b) Multi-class classification

---

### Q11. Which activation function is most commonly used in hidden layers of modern deep networks?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Moderate

a) Sigmoid
b) Tanh
c) ReLU
d) Linear

**Answer**: c) ReLU

---

### Q12. The backpropagation algorithm uses which mathematical concept?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI4 | **Marks**: 1 | **Difficulty**: Moderate

a) Integration
b) Chain rule of differentiation
c) Matrix multiplication only
d) Linear algebra only

**Answer**: b) Chain rule of differentiation

---

### Q13. The derivative of ReLU function for positive inputs is:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Moderate

a) 0
b) 1
c) x
d) e^x

**Answer**: b) 1

---

### Q14. The bias term in a neuron:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) Prevents overfitting
b) Allows shifting of the activation function
c) Reduces computational complexity
d) Eliminates the need for weights

**Answer**: b) Allows shifting of the activation function

---

### Q15. In a multilayer perceptron, the universal approximation theorem states:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) Any function can be approximated with infinite layers
b) A single hidden layer can approximate any continuous function
c) Only linear functions can be approximated
d) Approximation is impossible with finite neurons

**Answer**: b) A single hidden layer can approximate any continuous function

---

### Q16. The mathematical representation of a perceptron output is:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) y = wx + b
b) y = σ(wx + b)
c) y = w²x + b
d) y = log(wx + b)

**Answer**: b) y = σ(wx + b)

---

### Q17. TensorFlow operations are executed:
**PLO**: PLO-2 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI2 | **Marks**: 1 | **Difficulty**: Moderate

a) Immediately when defined
b) In a computational graph
c) Only during compilation
d) Randomly during runtime

**Answer**: b) In a computational graph

---

### Q18. The main advantage of using ReLU over sigmoid activation is:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL3 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Difficult

a) Smoother gradients
b) Bounded output range
c) Mitigation of vanishing gradient problem
d) Better for binary classification

**Answer**: c) Mitigation of vanishing gradient problem

---

### Q19. In TensorFlow, eager execution means:
**PLO**: PLO-2 | **CO**: CO-1 | **BL**: BL3 | **PI**: CO1-PI2 | **Marks**: 1 | **Difficulty**: Difficult

a) Operations are executed immediately
b) Operations are cached for later
c) Operations run in parallel
d) Operations are optimized automatically

**Answer**: a) Operations are executed immediately

---

### Q20. The vanishing gradient problem primarily affects:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL3 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Difficult

a) Shallow networks only
b) Deep networks with sigmoid activations
c) Networks with ReLU activations
d) Output layer only

**Answer**: b) Deep networks with sigmoid activations

---

## Module 2: Optimization & Regularization (25 questions)

### Q21. Which gradient descent variant uses the entire dataset for each update?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) Stochastic Gradient Descent
b) Mini-batch Gradient Descent
c) Batch Gradient Descent
d) Adam Optimizer

**Answer**: c) Batch Gradient Descent

---

### Q22. Learning rate determines:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) Number of epochs
b) Size of steps toward minimum
c) Number of hidden layers
d) Batch size

**Answer**: b) Size of steps toward minimum

---

### Q23. Overfitting occurs when:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) Model performs well on both training and test data
b) Model performs poorly on training data
c) Model performs well on training but poorly on test data
d) Model cannot learn from data

**Answer**: c) Model performs well on training but poorly on test data

---

### Q24. Which regularization technique randomly sets some neurons to zero during training?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI4 | **Marks**: 1 | **Difficulty**: Easy

a) L1 regularization
b) L2 regularization
c) Dropout
d) Early stopping

**Answer**: c) Dropout

---

### Q25. Early stopping prevents overfitting by:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) Stopping training when validation loss increases
b) Reducing the learning rate
c) Adding regularization terms
d) Increasing the batch size

**Answer**: a) Stopping training when validation loss increases

---

### Q26. Underfitting can be reduced by:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) Adding more regularization
b) Reducing model complexity
c) Increasing model complexity
d) Using smaller datasets

**Answer**: c) Increasing model complexity

---

### Q27. Which normalization technique normalizes across the batch dimension?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI5 | **Marks**: 1 | **Difficulty**: Easy

a) Layer normalization
b) Instance normalization
c) Batch normalization
d) Group normalization

**Answer**: c) Batch normalization

---

### Q28. The dropout rate typically ranges between:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI4 | **Marks**: 1 | **Difficulty**: Easy

a) 0 to 0.1
b) 0.2 to 0.5
c) 0.6 to 0.9
d) 0.9 to 1.0

**Answer**: b) 0.2 to 0.5

---

### Q29. Which statement about stochastic gradient descent is true?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) Uses entire dataset for each update
b) Uses one sample for each update
c) Always converges to global minimum
d) Requires large memory

**Answer**: b) Uses one sample for each update

---

### Q30. The purpose of validation set is to:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) Train the model
b) Tune hyperparameters and monitor overfitting
c) Test final performance
d) Increase training data

**Answer**: b) Tune hyperparameters and monitor overfitting

---

### Q31. The vanishing gradient problem is most severe with which activation function?
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI2 | **Marks**: 1 | **Difficulty**: Moderate

a) ReLU
b) Sigmoid
c) Leaky ReLU
d) Swish

**Answer**: b) Sigmoid

---

### Q32. The Adam optimizer combines:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) Momentum and RMSprop
b) SGD and batch gradient descent
c) L1 and L2 regularization
d) Dropout and batch normalization

**Answer**: a) Momentum and RMSprop

---

### Q33. L2 regularization adds which penalty to the loss function?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI4 | **Marks**: 1 | **Difficulty**: Moderate

a) Sum of absolute values of weights
b) Sum of squared weights
c) Product of weights
d) Maximum weight value

**Answer**: b) Sum of squared weights

---

### Q34. The exploding gradient problem can be mitigated by:
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI2 | **Marks**: 1 | **Difficulty**: Moderate

a) Using smaller learning rates
b) Gradient clipping
c) Adding more layers
d) Using sigmoid activation

**Answer**: b) Gradient clipping

---

### Q35. Which optimizer adapts the learning rate for each parameter individually?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) SGD
b) Momentum
c) AdaGrad
d) Standard gradient descent

**Answer**: c) AdaGrad

---

### Q36. The momentum parameter in gradient descent:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) Increases the learning rate
b) Accelerates convergence in relevant directions
c) Prevents overfitting
d) Normalizes the inputs

**Answer**: b) Accelerates convergence in relevant directions

---

### Q37. Batch normalization is applied:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI5 | **Marks**: 1 | **Difficulty**: Moderate

a) Only at the input layer
b) Only at the output layer
c) Between layers during training
d) Only during testing

**Answer**: c) Between layers during training

---

### Q38. In mini-batch gradient descent, the batch size affects:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) Only computational efficiency
b) Only gradient noise
c) Both computational efficiency and gradient noise
d) Neither efficiency nor noise

**Answer**: c) Both computational efficiency and gradient noise

---

### Q39. L1 regularization tends to produce:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI4 | **Marks**: 1 | **Difficulty**: Moderate

a) Dense weight matrices
b) Sparse weight matrices
c) Larger weight values
d) Negative weight values

**Answer**: b) Sparse weight matrices

---

### Q40. Which problem is addressed by batch normalization?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI5 | **Marks**: 1 | **Difficulty**: Moderate

a) Overfitting only
b) Internal covariate shift
c) Underfitting only
d) Memory optimization

**Answer**: b) Internal covariate shift

---

### Q41. Weight initialization using Xavier/Glorot method aims to:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI2 | **Marks**: 1 | **Difficulty**: Difficult

a) Minimize the loss function
b) Maintain gradient flow through layers
c) Increase computational speed
d) Reduce memory usage

**Answer**: b) Maintain gradient flow through layers

---

### Q42. The RMSprop optimizer addresses which problem of AdaGrad?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Difficult

a) Slow convergence
b) High memory usage
c) Aggressive learning rate decay
d) Poor generalization

**Answer**: c) Aggressive learning rate decay

---

### Q43. The bias-variance tradeoff is related to:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI3 | **Marks**: 1 | **Difficulty**: Difficult

a) Computational complexity
b) Overfitting and underfitting
c) Memory usage
d) Training time

**Answer**: b) Overfitting and underfitting

---

### Q44. Which technique can help with both vanishing and exploding gradients?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI2 | **Marks**: 1 | **Difficulty**: Difficult

a) Dropout
b) Residual connections
c) L2 regularization
d) Early stopping

**Answer**: b) Residual connections

---

### Q45. The learning rate finder technique helps to:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Difficult

a) Find optimal architecture
b) Determine optimal learning rate range
c) Prevent overfitting
d) Reduce training time

**Answer**: b) Determine optimal learning rate range

---

## Summary Statistics

### Question Distribution:
- **Total Questions**: 45 (All 1-mark MCQ)
- **Module 1**: 20 questions (44%)
- **Module 2**: 25 questions (56%)

### Difficulty Distribution:
- **Easy**: 20 questions (44%)
- **Moderate**: 20 questions (44%)
- **Difficult**: 5 questions (12%)

### Course Outcome Distribution:
- **CO-1**: 20 questions (44%)
- **CO-2**: 25 questions (56%)

### Bloom's Taxonomy Distribution:
- **BL1 (Remember)**: 20 questions (44%)
- **BL2 (Understand)**: 20 questions (44%)
- **BL3 (Apply)**: 5 questions (12%)

This file contains all 45 one-mark multiple choice questions extracted from the complete question bank, maintaining the same quality and CO-PO mapping while providing a focused MCQ-only resource.