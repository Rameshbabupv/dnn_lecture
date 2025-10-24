# Sample Test Sets for Formative Assessment I (Corrected)
**Course**: 21CSE558T - Deep Neural Network Architectures
**Assessment**: Formative Test I | **Coverage**: Modules 1-2
**Total Sets**: 3 (Set A, Set B, Set C) | **Questions per Set**: 25
**Duration**: 90 minutes | **Maximum Marks**: 50

---

# TEST SET A
**Balanced Difficulty Distribution**: Easy (10), Moderate (12), Difficult (3)
**Mark Distribution**: 1-mark (15), 2-mark (7), 5-mark (3)

---

## PART A: Multiple Choice Questions (15 marks)
*Choose the correct answer. Each question carries 1 mark.*

### 1. The XOR problem cannot be solved by a single perceptron because:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) It requires too many inputs
b) It is not linearly separable
c) It needs multiple outputs
d) It requires complex activation functions

**Answer**: b) It is not linearly separable

---

### 2. The sigmoid activation function outputs values in the range:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) (-1, 1)
b) (0, 1)
c) (-∞, ∞)
d) (0, ∞)

**Answer**: b) (0, 1)

---

### 3. Which gradient descent variant uses the entire dataset for each update?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) Stochastic Gradient Descent
b) Mini-batch Gradient Descent
c) Batch Gradient Descent
d) Adam Optimizer

**Answer**: c) Batch Gradient Descent

---

### 4. Overfitting occurs when:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) Model performs well on both training and test data
b) Model performs poorly on training data
c) Model performs well on training but poorly on test data
d) Model cannot learn from data

**Answer**: c) Model performs well on training but poorly on test data

---

### 5. Which regularization technique randomly sets some neurons to zero during training?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI4 | **Marks**: 1 | **Difficulty**: Easy

a) L1 regularization
b) L2 regularization
c) Dropout
d) Early stopping

**Answer**: c) Dropout

---

### 6. Which activation function is most commonly used in hidden layers of modern deep networks?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Moderate

a) Sigmoid
b) Tanh
c) ReLU
d) Linear

**Answer**: c) ReLU

---

### 7. The vanishing gradient problem is most severe with which activation function?
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI2 | **Marks**: 1 | **Difficulty**: Moderate

a) ReLU
b) Sigmoid
c) Leaky ReLU
d) Swish

**Answer**: b) Sigmoid

---

### 8. The Adam optimizer combines:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) Momentum and RMSprop
b) SGD and batch gradient descent
c) L1 and L2 regularization
d) Dropout and batch normalization

**Answer**: a) Momentum and RMSprop

---

### 9. The derivative of ReLU function for positive inputs is:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Moderate

a) 0
b) 1
c) x
d) e^x

**Answer**: b) 1

---

### 10. L2 regularization adds which penalty to the loss function?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI4 | **Marks**: 1 | **Difficulty**: Moderate

a) Sum of absolute values of weights
b) Sum of squared weights
c) Product of weights
d) Maximum weight value

**Answer**: b) Sum of squared weights

---

### 11. In a multilayer perceptron, the universal approximation theorem states:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) Any function can be approximated with infinite layers
b) A single hidden layer can approximate any continuous function
c) Only linear functions can be approximated
d) Approximation is impossible with finite neurons

**Answer**: b) A single hidden layer can approximate any continuous function

---

### 12. Batch normalization is applied:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI5 | **Marks**: 1 | **Difficulty**: Moderate

a) Only at the input layer
b) Only at the output layer
c) Between layers during training
d) Only during testing

**Answer**: c) Between layers during training

---

### 13. Weight initialization using Xavier/Glorot method aims to:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI2 | **Marks**: 1 | **Difficulty**: Difficult

a) Minimize the loss function
b) Maintain gradient flow through layers
c) Increase computational speed
d) Reduce memory usage

**Answer**: b) Maintain gradient flow through layers

---

### 14. The bias-variance tradeoff is related to:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI3 | **Marks**: 1 | **Difficulty**: Difficult

a) Computational complexity
b) Overfitting and underfitting
c) Memory usage
d) Training time

**Answer**: b) Overfitting and underfitting

---

### 15. The main advantage of using ReLU over sigmoid activation is:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL3 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Difficult

a) Smoother gradients
b) Bounded output range
c) Mitigation of vanishing gradient problem
d) Better for binary classification

**Answer**: c) Mitigation of vanishing gradient problem

---

## PART B: Short Answer Questions (14 marks)

### 16. What is the XOR problem and why can't a single perceptron solve it? (2 marks)
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI1 | **Difficulty**: Easy

**Expected Answer**:
- XOR problem: Output is 1 when inputs differ, 0 when same
- Single perceptron can only solve linearly separable problems
- XOR requires non-linear decision boundary

---

### 17. Explain the difference between loss function and cost function. (2 marks)
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI4 | **Difficulty**: Easy

**Expected Answer**:
- Loss function: Measures error for single training example
- Cost function: Average loss over entire training dataset
- Cost function is what we minimize during training

---

### 18. Define overfitting and mention two techniques to prevent it. (2 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI3 | **Difficulty**: Easy

**Expected Answer**:
- Overfitting: Model memorizes training data, poor generalization
- Prevention: Dropout, early stopping, L1/L2 regularization, data augmentation

---

### 19. What is the purpose of activation functions in neural networks? (2 marks)
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI3 | **Difficulty**: Easy

**Expected Answer**:
- Introduce non-linearity to enable learning complex patterns
- Control output range and behavior of neurons
- Enable networks to approximate non-linear functions

---

### 20. Compare sigmoid and ReLU activation functions. (2 marks)
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL3 | **PI**: CO1-PI3 | **Difficulty**: Moderate

**Expected Answer**:
- Sigmoid: Range (0,1), smooth, causes vanishing gradients, used for binary classification
- ReLU: Range [0,∞), computationally efficient, helps with vanishing gradients, used in hidden layers

---

### 21. Explain the concept of vanishing gradient problem. (2 marks)
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI2 | **Difficulty**: Moderate

**Expected Answer**:
- Gradients become exponentially small in early layers during backpropagation
- Caused by repeated multiplication of small derivatives (especially sigmoid)
- Results in slow/no learning in early layers

---

### 22. What is momentum in gradient descent and how does it help? (2 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Difficulty**: Moderate

**Expected Answer**:
- Momentum accumulates past gradients to maintain direction
- Helps accelerate convergence and overcome local minima
- Reduces oscillations in gradient descent path

---

## PART C: Descriptive Questions (21 marks)

### 23. Write a complete Python program using TensorFlow/Keras to create and train a neural network for the XOR problem. Include data preparation, model creation, training, and evaluation. (5 marks)
**PLO**: PLO-2 | **CO**: CO-1 | **BL**: BL6 | **PI**: CO1-PI2 | **Difficulty**: Moderate

**Expected Solution**:
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

### 24. Explain the vanishing gradient problem in detail. Discuss three different solutions with their mechanisms. (5 marks)
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI2 | **Difficulty**: Moderate

**Expected Answer**:
- Problem: Gradients become exponentially small in deep networks due to repeated multiplication
- Causes: Sigmoid activation derivatives <0.25, weight initialization, depth
- Solutions: 1) ReLU activation (derivative=1 for positive), 2) Better initialization (Xavier/He), 3) Residual connections (skip connections preserve gradient flow)
- Impact: Early layers learn slowly, affecting overall performance

---

### 25. Compare and contrast five different optimization algorithms used in deep learning. Include their advantages, disadvantages, and use cases. (5 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI1 | **Difficulty**: Moderate

**Expected Answer**:
- SGD: Simple, requires tuning, good with momentum; slow convergence
- Adam: Adaptive, combines momentum and RMSprop; works well generally but may overshoot
- RMSprop: Adaptive learning rates, good for non-stationary objectives; addresses AdaGrad decay
- AdaGrad: Accumulates gradients, good for sparse data; aggressive learning rate decay
- Momentum: Accelerates SGD, reduces oscillations; helps escape local minima

---

### 26. Analyze the impact of different activation functions on neural network performance with mathematical justification. (6 marks)
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL4 | **PI**: CO1-PI3 | **Difficulty**: Difficult

**Expected Answer**:
- Sigmoid: f(x)=1/(1+e^(-x)), derivative maximum 0.25, causes vanishing gradients
- ReLU: f(x)=max(0,x), derivative 0 or 1, solves vanishing gradients but can cause dead neurons
- Tanh: f(x)=tanh(x), range [-1,1], zero-centered, still has vanishing gradients
- Leaky ReLU: Small negative slope, prevents dead neurons
- Performance comparison: convergence speed, gradient flow, computational efficiency

---

---

# TEST SET B
**Balanced Difficulty Distribution**: Easy (10), Moderate (12), Difficult (3)
**Mark Distribution**: 1-mark (15), 2-mark (7), 5-mark (3)

---

## PART A: Multiple Choice Questions (15 marks)

### 1. In TensorFlow, which data structure is the fundamental building block?
**PLO**: PLO-2 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI2 | **Marks**: 1 | **Difficulty**: Easy

a) Array
b) Matrix
c) Tensor
d) Vector

**Answer**: c) Tensor

---

### 2. Learning rate determines:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) Number of epochs
b) Size of steps toward minimum
c) Number of hidden layers
d) Batch size

**Answer**: b) Size of steps toward minimum

---

### 3. Which loss function is typically used for binary classification?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI4 | **Marks**: 1 | **Difficulty**: Easy

a) Mean Squared Error
b) Mean Absolute Error
c) Binary Cross-entropy
d) Categorical Cross-entropy

**Answer**: c) Binary Cross-entropy

---

### 4. Early stopping prevents overfitting by:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) Stopping training when validation loss increases
b) Reducing the learning rate
c) Adding regularization terms
d) Increasing the batch size

**Answer**: a) Stopping training when validation loss increases

---

### 5. Which normalization technique normalizes across the batch dimension?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI5 | **Marks**: 1 | **Difficulty**: Easy

a) Layer normalization
b) Instance normalization
c) Batch normalization
d) Group normalization

**Answer**: c) Batch normalization

---

### 6. The backpropagation algorithm uses which mathematical concept?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI4 | **Marks**: 1 | **Difficulty**: Moderate

a) Integration
b) Chain rule of differentiation
c) Matrix multiplication only
d) Linear algebra only

**Answer**: b) Chain rule of differentiation

---

### 7. The exploding gradient problem can be mitigated by:
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI2 | **Marks**: 1 | **Difficulty**: Moderate

a) Using smaller learning rates
b) Gradient clipping
c) Adding more layers
d) Using sigmoid activation

**Answer**: b) Gradient clipping

---

### 8. Which optimizer adapts the learning rate for each parameter individually?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) SGD
b) Momentum
c) AdaGrad
d) Standard gradient descent

**Answer**: c) AdaGrad

---

### 9. The bias term in a neuron:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) Prevents overfitting
b) Allows shifting of the activation function
c) Reduces computational complexity
d) Eliminates the need for weights

**Answer**: b) Allows shifting of the activation function

---

### 10. The momentum parameter in gradient descent:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) Increases the learning rate
b) Accelerates convergence in relevant directions
c) Prevents overfitting
d) Normalizes the inputs

**Answer**: b) Accelerates convergence in relevant directions

---

### 11. The softmax activation function is primarily used for:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Moderate

a) Binary classification
b) Multi-class classification
c) Regression problems
d) Feature extraction

**Answer**: b) Multi-class classification

---

### 12. In mini-batch gradient descent, the batch size affects:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) Only computational efficiency
b) Only gradient noise
c) Both computational efficiency and gradient noise
d) Neither efficiency nor noise

**Answer**: c) Both computational efficiency and gradient noise

---

### 13. The RMSprop optimizer addresses which problem of AdaGrad?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Difficult

a) Slow convergence
b) High memory usage
c) Aggressive learning rate decay
d) Poor generalization

**Answer**: c) Aggressive learning rate decay

---

### 14. Which technique can help with both vanishing and exploding gradients?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI2 | **Marks**: 1 | **Difficulty**: Difficult

a) Dropout
b) Residual connections
c) L2 regularization
d) Early stopping

**Answer**: b) Residual connections

---

### 15. TensorFlow operations are executed:
**PLO**: PLO-2 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI2 | **Marks**: 1 | **Difficulty**: Moderate

a) Immediately when defined
b) In a computational graph
c) Only during compilation
d) Randomly during runtime

**Answer**: b) In a computational graph

---

## PART B: Short Answer Questions (14 marks)

### 16. Explain the role of bias in neural networks. (2 marks)
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI1 | **Difficulty**: Easy

---

### 17. What is the difference between parameters and hyperparameters? (2 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Difficulty**: Easy

---

### 18. What is dropout technique and its mechanism? (2 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI4 | **Difficulty**: Easy

---

### 19. What is early stopping and how does it prevent overfitting? (2 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI3 | **Difficulty**: Easy

---

### 20. Explain the difference between batch gradient descent and stochastic gradient descent. (2 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Difficulty**: Moderate

---

### 21. Compare L1 and L2 regularization techniques. (2 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI4 | **Difficulty**: Moderate

---

### 22. What is the universal approximation theorem and its significance? (2 marks)
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL3 | **PI**: CO1-PI1 | **Difficulty**: Moderate

---

## PART C: Descriptive Questions (21 marks)

### 23. Derive the backpropagation algorithm for a simple two-layer neural network with mathematical equations. Include forward pass, loss computation, and gradient calculations. (5 marks)
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL5 | **PI**: CO2-PI2 | **Difficulty**: Difficult

---

### 24. Design and implement a comprehensive regularization strategy for a deep neural network prone to overfitting. Include multiple techniques and their implementation details using TensorFlow/Keras. (5 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL6 | **PI**: CO2-PI4 | **Difficulty**: Difficult

---

### 25. Analyze the mathematical foundation of gradient descent and derive the update rules for weights and biases. Include convergence analysis. (5 marks)
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL5 | **PI**: CO2-PI1 | **Difficulty**: Difficult

---

### 26. Implement a neural network from scratch (without using high-level frameworks) for binary classification. Include forward pass, backward pass, and training loop. (6 marks)
**PLO**: PLO-2 | **CO**: CO-1 | **BL**: BL6 | **PI**: CO1-PI2 | **Difficulty**: Difficult

---

---

# TEST SET C
**Balanced Difficulty Distribution**: Easy (10), Moderate (12), Difficult (3)
**Mark Distribution**: 1-mark (15), 2-mark (7), 5-mark (3)

---

## PART A: Multiple Choice Questions (15 marks)

### 1. Which activation function can output negative values?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) Sigmoid
b) ReLU
c) Tanh
d) Softmax

**Answer**: c) Tanh

---

### 2. The perceptron learning algorithm can solve:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) Only linearly separable problems
b) Any classification problem
c) Only regression problems
d) Non-linear problems directly

**Answer**: a) Only linearly separable problems

---

### 3. Underfitting can be reduced by:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) Adding more regularization
b) Reducing model complexity
c) Increasing model complexity
d) Using smaller datasets

**Answer**: c) Increasing model complexity

---

### 4. The dropout rate typically ranges between:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI4 | **Marks**: 1 | **Difficulty**: Easy

a) 0 to 0.1
b) 0.2 to 0.5
c) 0.6 to 0.9
d) 0.9 to 1.0

**Answer**: b) 0.2 to 0.5

---

### 5. Which statement about stochastic gradient descent is true?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) Uses entire dataset for each update
b) Uses one sample for each update
c) Always converges to global minimum
d) Requires large memory

**Answer**: b) Uses one sample for each update

---

### 6. The mathematical representation of a perceptron output is:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) y = wx + b
b) y = σ(wx + b)
c) y = w²x + b
d) y = log(wx + b)

**Answer**: b) y = σ(wx + b)

---

### 7. L1 regularization tends to produce:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI4 | **Marks**: 1 | **Difficulty**: Moderate

a) Dense weight matrices
b) Sparse weight matrices
c) Larger weight values
d) Negative weight values

**Answer**: b) Sparse weight matrices

---

### 8. Which problem is addressed by batch normalization?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI5 | **Marks**: 1 | **Difficulty**: Moderate

a) Overfitting only
b) Internal covariate shift
c) Underfitting only
d) Memory optimization

**Answer**: b) Internal covariate shift

---

### 9. Which type of neural network layer performs weighted sum of inputs?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) Activation layer
b) Dense layer
c) Pooling layer
d) Dropout layer

**Answer**: b) Dense layer

---

### 10. In forward propagation, the output of each layer is:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI4 | **Marks**: 1 | **Difficulty**: Easy

a) Input to the previous layer
b) Input to the next layer
c) Stored for backward pass only
d) Discarded immediately

**Answer**: b) Input to the next layer

---

### 11. The purpose of validation set is to:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) Train the model
b) Tune hyperparameters and monitor overfitting
c) Test final performance
d) Increase training data

**Answer**: b) Tune hyperparameters and monitor overfitting

---

### 12. Which TensorFlow function is used to create a tensor filled with zeros?
**PLO**: PLO-2 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI2 | **Marks**: 1 | **Difficulty**: Easy

a) tf.ones()
b) tf.zeros()
c) tf.empty()
d) tf.fill()

**Answer**: b) tf.zeros()

---

### 13. The learning rate finder technique helps to:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Difficult

a) Find optimal architecture
b) Determine optimal learning rate range
c) Prevent overfitting
d) Reduce training time

**Answer**: b) Determine optimal learning rate range

---

### 14. In TensorFlow, eager execution means:
**PLO**: PLO-2 | **CO**: CO-1 | **BL**: BL3 | **PI**: CO1-PI2 | **Marks**: 1 | **Difficulty**: Difficult

a) Operations are executed immediately
b) Operations are cached for later
c) Operations run in parallel
d) Operations are optimized automatically

**Answer**: a) Operations are executed immediately

---

### 15. The vanishing gradient problem primarily affects:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL3 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Difficult

a) Shallow networks only
b) Deep networks with sigmoid activations
c) Networks with ReLU activations
d) Output layer only

**Answer**: b) Deep networks with sigmoid activations

---

## PART B: Short Answer Questions (14 marks)

### 16. What is batch normalization and what problem does it solve? (2 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI5 | **Difficulty**: Moderate

---

### 17. Explain the Adam optimizer and its advantages. (2 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Difficulty**: Moderate

---

### 18. What is gradient clipping and when is it used? (2 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI2 | **Difficulty**: Moderate

---

### 19. Compare underfitting and overfitting with their solutions. (2 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI3 | **Difficulty**: Difficult

---

### 20. Compare different types of gradient descent based on data usage. (2 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI1 | **Difficulty**: Difficult

---

### 21. Explain the concept of weight initialization and its importance. (2 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI2 | **Difficulty**: Difficult

---

### 22. What is the difference between training, validation, and test sets? (2 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI3 | **Difficulty**: Easy

---

## PART C: Descriptive Questions (21 marks)

### 23. Design a comprehensive strategy to diagnose and solve training problems in deep neural networks. Include overfitting, underfitting, and gradient problems. (5 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL6 | **PI**: CO2-PI3 | **Difficulty**: Difficult

---

### 24. Analyze the bias-variance tradeoff in the context of neural networks and provide strategies to balance them. Include mathematical foundations. (5 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL5 | **PI**: CO2-PI3 | **Difficulty**: Difficult

---

### 25. Compare and analyze the performance of different activation functions with their mathematical properties and practical implications. (5 marks)
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL4 | **PI**: CO1-PI3 | **Difficulty**: Moderate

---

### 26. Develop a complete training pipeline for a neural network including data preprocessing, model architecture design, training with callbacks, and performance evaluation. (6 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL6 | **PI**: CO2-PI1 | **Difficulty**: Difficult

---

---

# ANSWER KEYS FOR ALL SETS

## Set A Answer Key:
**Part A (MCQ)**: 1-b, 2-b, 3-c, 4-c, 5-c, 6-c, 7-b, 8-a, 9-b, 10-b, 11-b, 12-c, 13-b, 14-b, 15-c

## Set B Answer Key:
**Part A (MCQ)**: 1-c, 2-b, 3-c, 4-a, 5-c, 6-b, 7-b, 8-c, 9-b, 10-b, 11-b, 12-c, 13-c, 14-b, 15-b

## Set C Answer Key:
**Part A (MCQ)**: 1-c, 2-a, 3-c, 4-b, 5-b, 6-b, 7-b, 8-b, 9-b, 10-b, 11-b, 12-b, 13-b, 14-a, 15-b

---

# ASSESSMENT STATISTICS

## Overall Distribution Across Sets:

| Set | Easy | Moderate | Difficult | CO-1 | CO-2 | Total Marks |
|-----|------|----------|-----------|------|------|-------------|
| A | 10 | 12 | 3 | 12 | 13 | 50 |
| B | 10 | 12 | 3 | 11 | 14 | 50 |
| C | 10 | 12 | 3 | 10 | 15 | 50 |

**Perfect Balance**: Each test set maintains identical difficulty distribution while varying question content to prevent cheating and ensure comprehensive evaluation.