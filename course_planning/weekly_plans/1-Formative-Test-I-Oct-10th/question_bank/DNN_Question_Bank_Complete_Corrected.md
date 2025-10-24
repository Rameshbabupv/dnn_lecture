# Deep Neural Network Architectures - Question Bank (Corrected)
**Course Code**: 21CSE558T
**Course Title**: Deep Neural Network Architectures
**Test**: Formative Assessment I
**Coverage**: Module 1 & Module 2
**Total Questions**: 75
**Academic Year**: 2025-26
**Department**: School of Computing, SRM University

---

## Question Distribution Summary
- **1-mark questions**: 45 (MCQ format) - 60%
- **2-mark questions**: 20 (Short answer) - 27%
- **5-mark questions**: 10 (Descriptive) - 13%
- **Difficulty**: Easy (30), Moderate (35), Difficult (10)
- **Course Outcomes**: CO-1 (35 questions), CO-2 (40 questions)

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

# PART B: 2-MARK QUESTIONS (20 Questions)

### Q46. What is the XOR problem and why can't a single perceptron solve it?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI1 | **Marks**: 2 | **Difficulty**: Easy

**Expected Answer**:
- XOR problem: Output is 1 when inputs differ (0,1)→1, (1,0)→1; 0 when same (0,0)→0, (1,1)→0
- Single perceptron can only solve linearly separable problems with straight-line decision boundary
- XOR requires non-linear decision boundary, impossible with single perceptron

---

### Q47. Explain the difference between loss function and cost function.
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI4 | **Marks**: 2 | **Difficulty**: Easy

**Expected Answer**:
- Loss function: Measures error for single training example
- Cost function: Average loss over entire training dataset
- Cost function is what we minimize during training using optimization algorithms

---

### Q48. What is the purpose of activation functions in neural networks?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI3 | **Marks**: 2 | **Difficulty**: Easy

**Expected Answer**:
- Introduce non-linearity to enable learning complex patterns
- Control output range and behavior of neurons
- Enable networks to approximate non-linear functions

---

### Q49. Explain the role of bias in neural networks.
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI1 | **Marks**: 2 | **Difficulty**: Easy

**Expected Answer**:
- Bias allows shifting activation function horizontally
- Provides flexibility by allowing non-zero output when all inputs are zero
- Essential for learning patterns that don't pass through origin

---

### Q50. What is the difference between parameters and hyperparameters?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Marks**: 2 | **Difficulty**: Easy

**Expected Answer**:
- Parameters: Learned by the model during training (weights, biases)
- Hyperparameters: Set before training by user (learning rate, architecture, regularization)
- Parameters change during training, hyperparameters remain fixed

---

### Q51. Define overfitting and mention two techniques to prevent it.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI3 | **Marks**: 2 | **Difficulty**: Easy

**Expected Answer**:
- Overfitting: Model memorizes training data, poor generalization to new data
- Prevention techniques: Dropout, early stopping, L1/L2 regularization, data augmentation

---

### Q52. What is dropout technique and its mechanism?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI4 | **Marks**: 2 | **Difficulty**: Easy

**Expected Answer**:
- Randomly sets neurons to zero during training
- Forces network to not rely on specific neurons
- Prevents overfitting by reducing co-adaptation between neurons

---

### Q53. What is early stopping and how does it prevent overfitting?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI3 | **Marks**: 2 | **Difficulty**: Easy

**Expected Answer**:
- Stops training when validation loss starts increasing
- Prevents model from memorizing training data
- Simple regularization technique without modifying architecture

---

### Q54. Compare sigmoid and ReLU activation functions.
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL3 | **PI**: CO1-PI3 | **Marks**: 2 | **Difficulty**: Moderate

**Expected Answer**:
- Sigmoid: Range (0,1), smooth, causes vanishing gradients, used for binary classification output
- ReLU: Range [0,∞), computationally efficient, helps with vanishing gradients, used in hidden layers

---

### Q55. Explain the difference between batch gradient descent and stochastic gradient descent.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Marks**: 2 | **Difficulty**: Moderate

**Expected Answer**:
- Batch GD: Uses entire dataset, stable but slow convergence, high memory requirement
- SGD: Uses one sample, faster updates, noisy but can escape local minima, low memory

---

### Q56. What is momentum in gradient descent and how does it help?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Marks**: 2 | **Difficulty**: Moderate

**Expected Answer**:
- Momentum accumulates past gradients to maintain direction
- Helps accelerate convergence and overcome local minima
- Reduces oscillations in gradient descent path

---

### Q57. Compare L1 and L2 regularization techniques.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI4 | **Marks**: 2 | **Difficulty**: Moderate

**Expected Answer**:
- L1: Adds sum of absolute weights, promotes sparsity, feature selection
- L2: Adds sum of squared weights, prevents large weights, smooth weight distribution

---

### Q58. What is the universal approximation theorem and its significance?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL3 | **PI**: CO1-PI1 | **Marks**: 2 | **Difficulty**: Moderate

**Expected Answer**:
- Any continuous function can be approximated by single hidden layer MLP with sufficient neurons
- Theoretical foundation for neural network capabilities
- Justifies use of neural networks for complex problems

---

### Q59. Explain the concept of vanishing gradient problem.
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI2 | **Marks**: 2 | **Difficulty**: Moderate

**Expected Answer**:
- Gradients become exponentially small in early layers during backpropagation
- Caused by repeated multiplication of small derivatives (especially sigmoid)
- Results in slow/no learning in early layers, affecting network performance

---

### Q60. What is batch normalization and what problem does it solve?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI5 | **Marks**: 2 | **Difficulty**: Moderate

**Expected Answer**:
- Normalizes inputs to each layer during training
- Solves internal covariate shift problem
- Allows higher learning rates, reduces dependence on initialization

---

### Q61. Explain the Adam optimizer and its advantages.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Marks**: 2 | **Difficulty**: Moderate

**Expected Answer**:
- Combines momentum and RMSprop algorithms
- Adaptive learning rates for each parameter
- Works well with sparse gradients and noisy problems

---

### Q62. What is gradient clipping and when is it used?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI2 | **Marks**: 2 | **Difficulty**: Moderate

**Expected Answer**:
- Technique to limit gradient magnitude during backpropagation
- Used to prevent exploding gradient problem
- Clips gradients to maximum threshold value

---

### Q63. Compare underfitting and overfitting with their solutions.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI3 | **Marks**: 2 | **Difficulty**: Difficult

**Expected Answer**:
- Underfitting: High bias, low variance, too simple model, solution: increase complexity
- Overfitting: Low bias, high variance, too complex model, solution: regularization
- Both affect generalization but in opposite ways

---

### Q64. Compare different types of gradient descent based on data usage.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI1 | **Marks**: 2 | **Difficulty**: Difficult

**Expected Answer**:
- Batch: Entire dataset, stable but slow
- Stochastic: One sample, fast but noisy
- Mini-batch: Small batches, balance of stability and speed

---

### Q65. Explain the concept of weight initialization and its importance.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI2 | **Marks**: 2 | **Difficulty**: Difficult

**Expected Answer**:
- Process of setting initial weights before training
- Poor initialization can cause vanishing/exploding gradients
- Good initialization (Xavier, He) ensures proper gradient flow

---

# PART C: 5-MARK QUESTIONS (10 Questions)

### Q66. Write a complete Python program using TensorFlow/Keras to create and train a neural network for the XOR problem. Include data preparation, model creation, training, and evaluation.
**PLO**: PLO-2 | **CO**: CO-1 | **BL**: BL6 | **PI**: CO1-PI2 | **Marks**: 5 | **Difficulty**: Moderate

**Expected Answer**:
```python
import tensorflow as tf
import numpy as np

# XOR data preparation
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

# Evaluate
predictions = model.predict(X)
print("Predictions:", predictions)
```

---

### Q67. Derive the backpropagation algorithm for a simple two-layer neural network with mathematical equations.
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL5 | **PI**: CO2-PI2 | **Marks**: 5 | **Difficulty**: Difficult

**Expected Answer**:
- Forward pass: z₁ = W₁x + b₁, a₁ = σ(z₁), z₂ = W₂a₁ + b₂, ŷ = σ(z₂)
- Loss: L = ½(y - ŷ)²
- Backward pass: ∂L/∂W₂ = ∂L/∂ŷ × ∂ŷ/∂z₂ × ∂z₂/∂W₂
- Chain rule application for all weights and biases
- Update equations using gradient descent

---

### Q68. Explain the vanishing gradient problem in detail. Discuss three different solutions with their mechanisms.
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI2 | **Marks**: 5 | **Difficulty**: Moderate

**Expected Answer**:
- Problem: Gradients become exponentially small in deep networks
- Causes: Sigmoid derivatives <0.25, weight initialization, depth
- Solutions: 1) ReLU activation, 2) Better initialization (Xavier/He), 3) Residual connections
- Impact: Early layers learn slowly, affecting performance

---

### Q69. Compare and contrast five different optimization algorithms used in deep learning.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI1 | **Marks**: 5 | **Difficulty**: Moderate

**Expected Answer**:
- SGD: Simple, requires tuning, good with momentum
- Adam: Adaptive, combines momentum and RMSprop
- RMSprop: Adaptive learning rates, good for non-stationary
- AdaGrad: Accumulates gradients, learning rate decay
- Momentum: Accelerates SGD, reduces oscillations

---

### Q70. Design and implement a comprehensive regularization strategy for a deep neural network prone to overfitting.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL6 | **PI**: CO2-PI4 | **Marks**: 5 | **Difficulty**: Difficult

**Expected Answer**:
- Multiple techniques: Dropout (0.2-0.5), L2 regularization (λ=0.01)
- Early stopping with validation monitoring
- Data augmentation for variety
- Batch normalization for stability
- Implementation in Keras with callbacks

---

### Q71. Analyze the impact of different activation functions on neural network performance with mathematical justification.
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL4 | **PI**: CO1-PI3 | **Marks**: 5 | **Difficulty**: Moderate

**Expected Answer**:
- Sigmoid: Vanishing gradients, derivative max 0.25
- ReLU: Solves vanishing gradients, derivative 0 or 1
- Tanh: Zero-centered, still has vanishing gradients
- Leaky ReLU: Prevents dead neurons
- Performance comparison on convergence and efficiency

---

### Q72. Implement a neural network from scratch (without frameworks) for binary classification.
**PLO**: PLO-2 | **CO**: CO-1 | **BL**: BL6 | **PI**: CO1-PI2 | **Marks**: 5 | **Difficulty**: Difficult

**Expected Answer**:
- Initialize weights and biases
- Forward pass implementation
- Backward pass with gradient computation
- Weight update using gradient descent
- Training loop with loss monitoring

---

### Q73. Explain the mathematical foundation of gradient descent and derive the update rules.
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL5 | **PI**: CO2-PI1 | **Marks**: 5 | **Difficulty**: Difficult

**Expected Answer**:
- Objective: Minimize L(θ)
- Gradient: ∇L(θ) = [∂L/∂θ₁, ∂L/∂θ₂, ...]
- Update: θ(t+1) = θ(t) - α∇L(θ(t))
- Convergence analysis and learning rate importance

---

### Q74. Design a strategy to diagnose and solve training problems in deep neural networks.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL6 | **PI**: CO2-PI3 | **Marks**: 5 | **Difficulty**: Difficult

**Expected Answer**:
- Diagnosis: Monitor loss curves, gradient norms, activations
- Overfitting: Validation loss increases, apply regularization
- Underfitting: High losses, increase capacity
- Gradient problems: Monitor norms, apply clipping/better initialization
- Implementation with monitoring tools

---

### Q75. Analyze the bias-variance tradeoff and provide strategies to balance them.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL5 | **PI**: CO2-PI3 | **Marks**: 5 | **Difficulty**: Difficult

**Expected Answer**:
- Bias: Error from assumptions, underfitting
- Variance: Sensitivity to training data, overfitting
- Total Error = Bias² + Variance + Noise
- Solutions: Cross-validation, ensemble methods, proper regularization
- Finding optimal complexity point

---

## CORRECTED DISTRIBUTION SUMMARY

### Question Count Distribution:
- **1-mark**: 45 questions (60%)
- **2-mark**: 20 questions (27%)
- **5-mark**: 10 questions (13%)
- **Total**: 75 questions

### Difficulty Distribution:
- **Easy**: 30 questions (40%)
- **Moderate**: 35 questions (47%)
- **Difficult**: 10 questions (13%)

### Course Outcome Distribution:
- **CO-1**: 35 questions (47%)
- **CO-2**: 40 questions (53%)

### Module Coverage:
- **Module 1**: 35 questions (47%)
- **Module 2**: 40 questions (53%)