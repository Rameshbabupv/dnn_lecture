# Deep Neural Network Architectures - Complete Question Bank
**Course Code**: 21CSE558T
**Course Title**: Deep Neural Network Architectures
**Test**: Formative Assessment I
**Coverage**: Module 1 & Module 2
**Total Questions**: 75
**Academic Year**: 2025-26
**Department**: School of Computing, SRM University

---

## Question Distribution Summary
- **1-mark questions**: 45 (MCQ format)
- **2-mark questions**: 20 (Short answer)
- **5-mark questions**: 10 (Descriptive)
- **Difficulty**: Easy (30), Moderate (35), Difficult (10)
- **Course Outcomes**: CO-1 (40 questions), CO-2 (35 questions)

---

# PART A: 1-MARK QUESTIONS (45 Questions)

## Module 1: Introduction to Deep Learning

### Q1. Which of the following problems can be solved by a single perceptron?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) XOR problem
b) AND problem
c) Non-linearly separable problems
d) Multi-class classification

**Answer**: b) AND problem

---

### Q2. The XOR problem cannot be solved by a single perceptron because:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) It requires too many inputs
b) It is not linearly separable
c) It needs multiple outputs
d) It requires complex activation functions

**Answer**: b) It is not linearly separable

---

### Q3. In TensorFlow, which data structure is the fundamental building block?
**PLO**: PLO-2 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI2 | **Marks**: 1 | **Difficulty**: Easy

a) Array
b) Matrix
c) Tensor
d) Vector

**Answer**: c) Tensor

---

### Q4. The sigmoid activation function outputs values in the range:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) (-1, 1)
b) (0, 1)
c) (-∞, ∞)
d) (0, ∞)

**Answer**: b) (0, 1)

---

### Q5. Which activation function is most commonly used in hidden layers of modern deep networks?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Moderate

a) Sigmoid
b) Tanh
c) ReLU
d) Linear

**Answer**: c) ReLU

---

### Q6. The backpropagation algorithm uses which mathematical concept?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI4 | **Marks**: 1 | **Difficulty**: Moderate

a) Integration
b) Chain rule of differentiation
c) Matrix multiplication only
d) Linear algebra only

**Answer**: b) Chain rule of differentiation

---

### Q7. Which loss function is typically used for binary classification?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI4 | **Marks**: 1 | **Difficulty**: Easy

a) Mean Squared Error
b) Mean Absolute Error
c) Binary Cross-entropy
d) Categorical Cross-entropy

**Answer**: c) Binary Cross-entropy

---

### Q8. In a multilayer perceptron, the universal approximation theorem states:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) Any function can be approximated with infinite layers
b) A single hidden layer can approximate any continuous function
c) Only linear functions can be approximated
d) Approximation is impossible with finite neurons

**Answer**: b) A single hidden layer can approximate any continuous function

---

### Q9. Which TensorFlow function is used to create a tensor filled with zeros?
**PLO**: PLO-2 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI2 | **Marks**: 1 | **Difficulty**: Easy

a) tf.ones()
b) tf.zeros()
c) tf.empty()
d) tf.fill()

**Answer**: b) tf.zeros()

---

### Q10. The derivative of ReLU function for positive inputs is:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Moderate

a) 0
b) 1
c) x
d) e^x

**Answer**: b) 1

---

### Q11. Which type of neural network layer performs weighted sum of inputs?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) Activation layer
b) Dense layer
c) Pooling layer
d) Dropout layer

**Answer**: b) Dense layer

---

### Q12. The bias term in a neuron:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) Prevents overfitting
b) Allows shifting of the activation function
c) Reduces computational complexity
d) Eliminates the need for weights

**Answer**: b) Allows shifting of the activation function

---

### Q13. In forward propagation, the output of each layer is:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI4 | **Marks**: 1 | **Difficulty**: Easy

a) Input to the previous layer
b) Input to the next layer
c) Stored for backward pass only
d) Discarded immediately

**Answer**: b) Input to the next layer

---

### Q14. Which activation function can output negative values?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) Sigmoid
b) ReLU
c) Tanh
d) Softmax

**Answer**: c) Tanh

---

### Q15. The softmax activation function is primarily used for:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Moderate

a) Binary classification
b) Multi-class classification
c) Regression problems
d) Feature extraction

**Answer**: b) Multi-class classification

---

## Module 2: Optimization and Regularization

### Q16. Which gradient descent variant uses the entire dataset for each update?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) Stochastic Gradient Descent
b) Mini-batch Gradient Descent
c) Batch Gradient Descent
d) Adam Optimizer

**Answer**: c) Batch Gradient Descent

---

### Q17. The vanishing gradient problem is most severe with which activation function?
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI2 | **Marks**: 1 | **Difficulty**: Moderate

a) ReLU
b) Sigmoid
c) Leaky ReLU
d) Swish

**Answer**: b) Sigmoid

---

### Q18. Learning rate determines:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) Number of epochs
b) Size of steps toward minimum
c) Number of hidden layers
d) Batch size

**Answer**: b) Size of steps toward minimum

---

### Q19. Overfitting occurs when:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) Model performs well on both training and test data
b) Model performs poorly on training data
c) Model performs well on training but poorly on test data
d) Model cannot learn from data

**Answer**: c) Model performs well on training but poorly on test data

---

### Q20. Which regularization technique randomly sets some neurons to zero during training?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI4 | **Marks**: 1 | **Difficulty**: Easy

a) L1 regularization
b) L2 regularization
c) Dropout
d) Early stopping

**Answer**: c) Dropout

---

### Q21. The Adam optimizer combines:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) Momentum and RMSprop
b) SGD and batch gradient descent
c) L1 and L2 regularization
d) Dropout and batch normalization

**Answer**: a) Momentum and RMSprop

---

### Q22. L2 regularization adds which penalty to the loss function?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI4 | **Marks**: 1 | **Difficulty**: Moderate

a) Sum of absolute values of weights
b) Sum of squared weights
c) Product of weights
d) Maximum weight value

**Answer**: b) Sum of squared weights

---

### Q23. Batch normalization is applied:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI5 | **Marks**: 1 | **Difficulty**: Moderate

a) Only at the input layer
b) Only at the output layer
c) Between layers during training
d) Only during testing

**Answer**: c) Between layers during training

---

### Q24. The exploding gradient problem can be mitigated by:
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI2 | **Marks**: 1 | **Difficulty**: Moderate

a) Using smaller learning rates
b) Gradient clipping
c) Adding more layers
d) Using sigmoid activation

**Answer**: b) Gradient clipping

---

### Q25. Early stopping prevents overfitting by:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) Stopping training when validation loss increases
b) Reducing the learning rate
c) Adding regularization terms
d) Increasing the batch size

**Answer**: a) Stopping training when validation loss increases

---

### Q26. Which optimizer adapts the learning rate for each parameter individually?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) SGD
b) Momentum
c) AdaGrad
d) Standard gradient descent

**Answer**: c) AdaGrad

---

### Q27. The momentum parameter in gradient descent:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) Increases the learning rate
b) Accelerates convergence in relevant directions
c) Prevents overfitting
d) Normalizes the inputs

**Answer**: b) Accelerates convergence in relevant directions

---

### Q28. Underfitting can be reduced by:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) Adding more regularization
b) Reducing model complexity
c) Increasing model complexity
d) Using smaller datasets

**Answer**: c) Increasing model complexity

---

### Q29. Which normalization technique normalizes across the batch dimension?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI5 | **Marks**: 1 | **Difficulty**: Easy

a) Layer normalization
b) Instance normalization
c) Batch normalization
d) Group normalization

**Answer**: c) Batch normalization

---

### Q30. The learning rate schedule that reduces learning rate over time is called:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) Learning rate decay
b) Learning rate explosion
c) Learning rate normalization
d) Learning rate regularization

**Answer**: a) Learning rate decay

---

### Q31. In mini-batch gradient descent, the batch size affects:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) Only computational efficiency
b) Only gradient noise
c) Both computational efficiency and gradient noise
d) Neither efficiency nor noise

**Answer**: c) Both computational efficiency and gradient noise

---

### Q32. Which activation function helps mitigate the vanishing gradient problem?
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI2 | **Marks**: 1 | **Difficulty**: Moderate

a) Sigmoid
b) Tanh
c) ReLU
d) Linear

**Answer**: c) ReLU

---

### Q33. Weight initialization using Xavier/Glorot method aims to:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI2 | **Marks**: 1 | **Difficulty**: Difficult

a) Minimize the loss function
b) Maintain gradient flow through layers
c) Increase computational speed
d) Reduce memory usage

**Answer**: b) Maintain gradient flow through layers

---

### Q34. The RMSprop optimizer addresses which problem of AdaGrad?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Difficult

a) Slow convergence
b) High memory usage
c) Aggressive learning rate decay
d) Poor generalization

**Answer**: c) Aggressive learning rate decay

---

### Q35. Gradient clipping is implemented by:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI2 | **Marks**: 1 | **Difficulty**: Moderate

a) Setting gradients to zero
b) Limiting the magnitude of gradients
c) Reversing gradient direction
d) Adding noise to gradients

**Answer**: b) Limiting the magnitude of gradients

---

### Q36. Which problem is addressed by batch normalization?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI5 | **Marks**: 1 | **Difficulty**: Moderate

a) Overfitting only
b) Internal covariate shift
c) Underfitting only
d) Memory optimization

**Answer**: b) Internal covariate shift

---

### Q37. The dropout rate typically ranges between:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI4 | **Marks**: 1 | **Difficulty**: Easy

a) 0 to 0.1
b) 0.2 to 0.5
c) 0.6 to 0.9
d) 0.9 to 1.0

**Answer**: b) 0.2 to 0.5

---

### Q38. L1 regularization tends to produce:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI4 | **Marks**: 1 | **Difficulty**: Moderate

a) Dense weight matrices
b) Sparse weight matrices
c) Larger weight values
d) Negative weight values

**Answer**: b) Sparse weight matrices

---

### Q39. The plateau in loss function can be addressed by:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Difficult

a) Increasing batch size
b) Learning rate scheduling
c) Adding more data
d) Reducing model complexity

**Answer**: b) Learning rate scheduling

---

### Q40. Which statement about stochastic gradient descent is true?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) Uses entire dataset for each update
b) Uses one sample for each update
c) Always converges to global minimum
d) Requires large memory

**Answer**: b) Uses one sample for each update

---

### Q41. The purpose of validation set is to:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) Train the model
b) Tune hyperparameters and monitor overfitting
c) Test final performance
d) Increase training data

**Answer**: b) Tune hyperparameters and monitor overfitting

---

### Q42. Which normalization technique is independent of batch size?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI5 | **Marks**: 1 | **Difficulty**: Moderate

a) Batch normalization
b) Layer normalization
c) Group normalization
d) Both b and c

**Answer**: d) Both b and c

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

# PART B: 2-MARK QUESTIONS (20 Questions)

### Q46. Compare sigmoid and ReLU activation functions in terms of their properties and use cases.
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL4 | **PI**: CO1-PI3 | **Marks**: 2 | **Difficulty**: Moderate

**Expected Answer**:
- Sigmoid: Range (0,1), smooth, causes vanishing gradients, used in output layer for binary classification
- ReLU: Range [0,∞), computationally efficient, helps with vanishing gradients, used in hidden layers

---

### Q47. Explain the difference between batch gradient descent and stochastic gradient descent.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI1 | **Marks**: 2 | **Difficulty**: Easy

**Expected Answer**:
- Batch GD: Uses entire dataset, stable but slow convergence, high memory requirement
- SGD: Uses one sample, faster updates, noisy but can escape local minima, low memory

---

### Q48. What is the XOR problem and why can't a single perceptron solve it?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL4 | **PI**: CO1-PI1 | **Marks**: 2 | **Difficulty**: Easy

**Expected Answer**:
- XOR problem: Output is 1 when inputs differ, 0 when same
- Single perceptron can only solve linearly separable problems
- XOR is not linearly separable, requires non-linear decision boundary

---

### Q49. Define overfitting and mention two techniques to prevent it.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI3 | **Marks**: 2 | **Difficulty**: Easy

**Expected Answer**:
- Overfitting: Model memorizes training data, poor generalization to new data
- Prevention techniques: Dropout, L2 regularization, early stopping, data augmentation

---

### Q50. Explain the concept of vanishing gradient problem and its impact on deep networks.
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI2 | **Marks**: 2 | **Difficulty**: Moderate

**Expected Answer**:
- Gradients become exponentially small in early layers during backpropagation
- Caused by repeated multiplication of small derivatives (especially sigmoid)
- Results in slow/no learning in early layers, affecting network performance

---

### Q51. Compare L1 and L2 regularization techniques.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI4 | **Marks**: 2 | **Difficulty**: Moderate

**Expected Answer**:
- L1: Adds sum of absolute weights, promotes sparsity, feature selection
- L2: Adds sum of squared weights, prevents large weights, smooth weight distribution

---

### Q52. What is the universal approximation theorem and its significance?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL4 | **PI**: CO1-PI1 | **Marks**: 2 | **Difficulty**: Moderate

**Expected Answer**:
- Any continuous function can be approximated by a single hidden layer MLP with sufficient neurons
- Theoretical foundation for neural network capabilities
- Justifies use of neural networks for complex problems

---

### Q53. Explain the role of bias in neural networks.
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL3 | **PI**: CO1-PI1 | **Marks**: 2 | **Difficulty**: Easy

**Expected Answer**:
- Bias allows shifting activation function horizontally
- Provides flexibility in modeling by allowing non-zero output when all inputs are zero
- Essential for learning patterns that don't pass through origin

---

### Q54. What is batch normalization and what problem does it solve?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI5 | **Marks**: 2 | **Difficulty**: Moderate

**Expected Answer**:
- Normalizes inputs to each layer during training
- Solves internal covariate shift problem
- Allows higher learning rates, reduces dependence on initialization

---

### Q55. Explain the difference between loss function and cost function.
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI4 | **Marks**: 2 | **Difficulty**: Easy

**Expected Answer**:
- Loss function: Measures error for single training example
- Cost function: Average loss over entire training dataset
- Cost function is what we minimize during training

---

### Q56. What is momentum in gradient descent and how does it help?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Marks**: 2 | **Difficulty**: Moderate

**Expected Answer**:
- Momentum accumulates past gradients to maintain direction
- Helps accelerate convergence and overcome local minima
- Reduces oscillations in gradient descent path

---

### Q57. Compare underfitting and overfitting with their causes and solutions.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI3 | **Marks**: 2 | **Difficulty**: Moderate

**Expected Answer**:
- Underfitting: High bias, low variance, too simple model, solution: increase complexity
- Overfitting: Low bias, high variance, too complex model, solution: regularization

---

### Q58. What is the purpose of activation functions in neural networks?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL3 | **PI**: CO1-PI3 | **Marks**: 2 | **Difficulty**: Easy

**Expected Answer**:
- Introduce non-linearity to enable learning complex patterns
- Control output range and behavior of neurons
- Enable networks to approximate non-linear functions

---

### Q59. Explain the Adam optimizer and its advantages.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Marks**: 2 | **Difficulty**: Moderate

**Expected Answer**:
- Combines momentum and RMSprop
- Adaptive learning rates for each parameter
- Works well with sparse gradients and noisy problems

---

### Q60. What is gradient clipping and when is it used?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI2 | **Marks**: 2 | **Difficulty**: Moderate

**Expected Answer**:
- Technique to limit gradient magnitude during backpropagation
- Used to prevent exploding gradient problem
- Clips gradients to maximum threshold value

---

### Q61. Explain the concept of weight initialization and its importance.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI2 | **Marks**: 2 | **Difficulty**: Moderate

**Expected Answer**:
- Process of setting initial weights before training
- Poor initialization can cause vanishing/exploding gradients
- Good initialization (Xavier, He) ensures proper gradient flow

---

### Q62. What is early stopping and how does it prevent overfitting?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI3 | **Marks**: 2 | **Difficulty**: Easy

**Expected Answer**:
- Stops training when validation loss starts increasing
- Prevents model from memorizing training data
- Simple regularization technique without modifying architecture

---

### Q63. Compare different types of gradient descent based on data usage.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI1 | **Marks**: 2 | **Difficulty**: Moderate

**Expected Answer**:
- Batch: Entire dataset, stable but slow
- Stochastic: One sample, fast but noisy
- Mini-batch: Small batches, balance of stability and speed

---

### Q64. What is the difference between parameters and hyperparameters?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Marks**: 2 | **Difficulty**: Easy

**Expected Answer**:
- Parameters: Learned by the model (weights, biases)
- Hyperparameters: Set before training (learning rate, architecture, regularization)

---

### Q65. Explain dropout technique and its mechanism.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI4 | **Marks**: 2 | **Difficulty**: Easy

**Expected Answer**:
- Randomly sets neurons to zero during training
- Forces network to not rely on specific neurons
- Prevents overfitting by reducing co-adaptation

---

# PART C: 5-MARK QUESTIONS (10 Questions)

### Q66. Derive the backpropagation algorithm for a simple two-layer neural network with mathematical equations.
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL5 | **PI**: CO2-PI2 | **Marks**: 5 | **Difficulty**: Difficult

**Expected Answer**:
- Forward pass equations: z₁ = W₁x + b₁, a₁ = σ(z₁), z₂ = W₂a₁ + b₂, ŷ = σ(z₂)
- Loss function: L = ½(y - ŷ)²
- Backward pass: ∂L/∂W₂ = ∂L/∂ŷ × ∂ŷ/∂z₂ × ∂z₂/∂W₂
- Chain rule application for all weights and biases
- Update equations using gradient descent

---

### Q67. Write a complete Python program using TensorFlow/Keras to create and train a neural network for the XOR problem.
**PLO**: PLO-2 | **CO**: CO-1 | **BL**: BL6 | **PI**: CO1-PI2 | **Marks**: 5 | **Difficulty**: Moderate

**Expected Answer**:
```python
import tensorflow as tf
import numpy as np

# XOR data
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=1000, verbose=0)

# Test
predictions = model.predict(X)
print(predictions)
```

---

### Q68. Explain the vanishing gradient problem in detail. Discuss three different solutions with their mechanisms.
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI2 | **Marks**: 5 | **Difficulty**: Moderate

**Expected Answer**:
- Problem: Gradients become exponentially small in deep networks due to repeated multiplication
- Causes: Sigmoid activation derivatives <0.25, weight initialization, depth
- Solutions: 1) ReLU activation (derivative=1 for positive), 2) Better initialization (Xavier/He), 3) Residual connections (skip connections preserve gradient flow)
- Impact: Early layers learn slowly, affecting overall performance

---

### Q69. Design and implement a comprehensive regularization strategy for a deep neural network prone to overfitting.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL6 | **PI**: CO2-PI4 | **Marks**: 5 | **Difficulty**: Difficult

**Expected Answer**:
- Multiple techniques: Dropout (0.2-0.5), L2 regularization (λ=0.01), Early stopping
- Implementation in Keras with validation monitoring
- Data augmentation for increased training variety
- Batch normalization for stable training
- Learning rate scheduling for better convergence

---

### Q70. Compare and contrast five different optimization algorithms used in deep learning.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI1 | **Marks**: 5 | **Difficulty**: Moderate

**Expected Answer**:
- SGD: Simple, requires tuning, good with momentum
- Adam: Adaptive, combines momentum and RMSprop, works well generally
- RMSprop: Adaptive learning rates, good for non-stationary objectives
- AdaGrad: Accumulates gradients, good for sparse data, learning rate decay
- Momentum: Accelerates SGD, reduces oscillations
- Comparison on convergence speed, memory usage, hyperparameter sensitivity

---

### Q71. Analyze the impact of different activation functions on neural network performance with mathematical justification.
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL4 | **PI**: CO1-PI3 | **Marks**: 5 | **Difficulty**: Moderate

**Expected Answer**:
- Sigmoid: f(x)=1/(1+e^(-x)), derivative maximum 0.25, vanishing gradients
- ReLU: f(x)=max(0,x), derivative 0 or 1, solves vanishing gradients
- Tanh: f(x)=tanh(x), range [-1,1], zero-centered, still has vanishing gradients
- Leaky ReLU: Small negative slope, prevents dead neurons
- Performance comparison: convergence speed, gradient flow, computational efficiency

---

### Q72. Implement a neural network from scratch (without using high-level frameworks) for binary classification.
**PLO**: PLO-2 | **CO**: CO-1 | **BL**: BL6 | **PI**: CO1-PI2 | **Marks**: 5 | **Difficulty**: Difficult

**Expected Answer**:
```python
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        m = X.shape[0]
        dz2 = output - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.a1 * (1 - self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        return dW1, db1, dW2, db2
```

---

### Q73. Explain the mathematical foundation of gradient descent and derive the update rules for weights and biases.
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL5 | **PI**: CO2-PI1 | **Marks**: 5 | **Difficulty**: Difficult

**Expected Answer**:
- Objective: Minimize loss function L(θ) where θ represents parameters
- Gradient: ∇L(θ) = [∂L/∂θ₁, ∂L/∂θ₂, ..., ∂L/∂θₙ]
- Update rule: θ(t+1) = θ(t) - α∇L(θ(t))
- For neural networks: W(t+1) = W(t) - α(∂L/∂W), b(t+1) = b(t) - α(∂L/∂b)
- Learning rate α determines step size, convergence analysis

---

### Q74. Design a comprehensive strategy to diagnose and solve training problems in deep neural networks.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL6 | **PI**: CO2-PI3 | **Marks**: 5 | **Difficulty**: Difficult

**Expected Answer**:
- Diagnosis: Monitor training/validation loss curves, gradient norms, activation distributions
- Overfitting: Validation loss increases, solutions: regularization, early stopping, more data
- Underfitting: Both losses high, solutions: increase capacity, reduce regularization
- Vanishing gradients: Small gradient norms, solutions: better initialization, ReLU, residual connections
- Exploding gradients: Large gradient norms, solutions: gradient clipping, lower learning rate
- Implementation strategy with monitoring tools

---

### Q75. Analyze the bias-variance tradeoff in the context of neural networks and provide strategies to balance them.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL5 | **PI**: CO2-PI3 | **Marks**: 5 | **Difficulty**: Difficult

**Expected Answer**:
- Bias: Error from overly simplistic assumptions, underfitting
- Variance: Error from sensitivity to small fluctuations in training set, overfitting
- Total Error = Bias² + Variance + Irreducible Error
- High bias solutions: Increase model complexity, reduce regularization
- High variance solutions: More data, regularization, ensemble methods
- Optimal point balances both through cross-validation and model selection

---

## Summary Statistics
- **Total Questions**: 75
- **1-mark**: 45 questions (Easy: 20, Moderate: 20, Difficult: 5)
- **2-mark**: 20 questions (Easy: 8, Moderate: 12, Difficult: 0)
- **5-mark**: 10 questions (Easy: 0, Moderate: 5, Difficult: 5)
- **Module 1 Coverage**: 35 questions
- **Module 2 Coverage**: 40 questions
- **CO-1 Focus**: 40 questions
- **CO-2 Focus**: 35 questions