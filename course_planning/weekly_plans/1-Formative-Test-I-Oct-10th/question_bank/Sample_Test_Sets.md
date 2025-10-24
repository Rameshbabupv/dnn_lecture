# Sample Test Sets for Formative Assessment I
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

### 1. Which of the following problems can be solved by a single perceptron?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) XOR problem
b) AND problem
c) Non-linearly separable problems
d) Multi-class classification

---

### 2. The sigmoid activation function outputs values in the range:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) (-1, 1)
b) (0, 1)
c) (-∞, ∞)
d) (0, ∞)

---

### 3. Which gradient descent variant uses the entire dataset for each update?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) Stochastic Gradient Descent
b) Mini-batch Gradient Descent
c) Batch Gradient Descent
d) Adam Optimizer

---

### 4. The vanishing gradient problem is most severe with which activation function?
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI2 | **Marks**: 1 | **Difficulty**: Moderate

a) ReLU
b) Sigmoid
c) Leaky ReLU
d) Swish

---

### 5. Which regularization technique randomly sets some neurons to zero during training?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI4 | **Marks**: 1 | **Difficulty**: Easy

a) L1 regularization
b) L2 regularization
c) Dropout
d) Early stopping

---

### 6. The universal approximation theorem states that:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) Any function can be approximated by a linear model
b) A single hidden layer MLP can approximate any continuous function
c) Deep networks always outperform shallow networks
d) Neural networks can only solve linear problems

---

### 7. Which optimizer adapts the learning rate for each parameter individually?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) SGD
b) Momentum
c) AdaGrad
d) Standard gradient descent

---

### 8. The derivative of ReLU function for positive inputs is:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Moderate

a) 0
b) 1
c) x
d) e^x

---

### 9. Overfitting occurs when:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) Model performs well on both training and test data
b) Model performs poorly on training data
c) Model performs well on training but poorly on test data
d) Model cannot learn from data

---

### 10. Which loss function is typically used for binary classification?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI4 | **Marks**: 1 | **Difficulty**: Easy

a) Mean Squared Error
b) Mean Absolute Error
c) Binary Cross-entropy
d) Categorical Cross-entropy

---

### 11. The Adam optimizer combines:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) Momentum and RMSprop
b) SGD and batch gradient descent
c) L1 and L2 regularization
d) Dropout and batch normalization

---

### 12. In TensorFlow, which data structure is the fundamental building block?
**PLO**: PLO-2 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI2 | **Marks**: 1 | **Difficulty**: Easy

a) Array
b) Matrix
c) Tensor
d) Vector

---

### 13. Batch normalization is applied:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI5 | **Marks**: 1 | **Difficulty**: Moderate

a) Only at the input layer
b) Only at the output layer
c) Between layers during training
d) Only during testing

---

### 14. Weight initialization using Xavier/Glorot method aims to:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI2 | **Marks**: 1 | **Difficulty**: Difficult

a) Minimize the loss function
b) Maintain gradient flow through layers
c) Increase computational speed
d) Reduce memory usage

---

### 15. The learning rate in gradient descent determines:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) The number of epochs
b) The size of steps taken towards minimum
c) The number of hidden layers
d) The activation function type

---

## PART B: Short Answer Questions (21 marks)

### 16. What is the XOR problem and why can't a single perceptron solve it? (3 marks)
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL4 | **PI**: CO1-PI1 | **Difficulty**: Easy

---

### 17. Compare sigmoid and ReLU activation functions in terms of their properties and use cases. (3 marks)
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL4 | **PI**: CO1-PI3 | **Difficulty**: Moderate

---

### 18. Explain the difference between batch gradient descent and stochastic gradient descent. (3 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI1 | **Difficulty**: Easy

---

### 19. Define overfitting and mention three techniques to prevent it. (3 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI3 | **Difficulty**: Easy

---

### 20. What is momentum in gradient descent and how does it help? (3 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Difficulty**: Moderate

---

### 21. Explain the concept of vanishing gradient problem and its impact on deep networks. (3 marks)
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI2 | **Difficulty**: Moderate

---

### 22. What is dropout technique and its mechanism? (3 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI4 | **Difficulty**: Easy

---

## PART C: Descriptive Questions (14 marks)

### 23. Write a complete Python program using TensorFlow/Keras to create and train a neural network for the XOR problem. Include data preparation, model creation, training, and evaluation. (7 marks)
**PLO**: PLO-2 | **CO**: CO-1 | **BL**: BL6 | **PI**: CO1-PI2 | **Difficulty**: Moderate

---

### 24. Analyze the impact of different activation functions on neural network performance with mathematical justification. Compare at least four activation functions. (7 marks)
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL4 | **PI**: CO1-PI3 | **Difficulty**: Moderate

---

# TEST SET B
**Balanced Difficulty Distribution**: Easy (10), Moderate (12), Difficult (3)
**Mark Distribution**: 1-mark (15), 2-mark (7), 5-mark (3)

---

## PART A: Multiple Choice Questions (15 marks)

### 1. The XOR problem cannot be solved by a single perceptron because:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) It requires too many inputs
b) It is not linearly separable
c) It needs multiple outputs
d) It requires complex activation functions

---

### 2. Which activation function is most commonly used in hidden layers of modern deep networks?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Moderate

a) Sigmoid
b) Tanh
c) ReLU
d) Linear

---

### 3. Learning rate determines:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) Number of epochs
b) Size of steps toward minimum
c) Number of hidden layers
d) Batch size

---

### 4. The exploding gradient problem can be mitigated by:
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI2 | **Marks**: 1 | **Difficulty**: Moderate

a) Using smaller learning rates
b) Gradient clipping
c) Adding more layers
d) Using sigmoid activation

---

### 5. L2 regularization adds which penalty to the loss function?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI4 | **Marks**: 1 | **Difficulty**: Moderate

a) Sum of absolute values of weights
b) Sum of squared weights
c) Product of weights
d) Maximum weight value

---

### 6. The backpropagation algorithm uses which mathematical concept?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI4 | **Marks**: 1 | **Difficulty**: Moderate

a) Integration
b) Chain rule of differentiation
c) Matrix multiplication only
d) Linear algebra only

---

### 7. Which normalization technique normalizes across the batch dimension?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI5 | **Marks**: 1 | **Difficulty**: Easy

a) Layer normalization
b) Instance normalization
c) Batch normalization
d) Group normalization

---

### 8. Which activation function can output negative values?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) Sigmoid
b) ReLU
c) Tanh
d) Softmax

---

### 9. Early stopping prevents overfitting by:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) Stopping training when validation loss increases
b) Reducing the learning rate
c) Adding regularization terms
d) Increasing the batch size

---

### 10. Which TensorFlow function is used to create a tensor filled with zeros?
**PLO**: PLO-2 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI2 | **Marks**: 1 | **Difficulty**: Easy

a) tf.ones()
b) tf.zeros()
c) tf.empty()
d) tf.fill()

---

### 11. The momentum parameter in gradient descent:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) Increases the learning rate
b) Accelerates convergence in relevant directions
c) Prevents overfitting
d) Normalizes the inputs

---

### 12. The softmax activation function is primarily used for:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Moderate

a) Binary classification
b) Multi-class classification
c) Regression problems
d) Feature extraction

---

### 13. Underfitting can be reduced by:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) Adding more regularization
b) Reducing model complexity
c) Increasing model complexity
d) Using smaller datasets

---

### 14. The RMSprop optimizer addresses which problem of AdaGrad?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Difficult

a) Slow convergence
b) High memory usage
c) Aggressive learning rate decay
d) Poor generalization

---

### 15. Which type of neural network layer performs weighted sum of inputs?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL1 | **PI**: CO1-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) Activation layer
b) Dense layer
c) Pooling layer
d) Dropout layer

---

## PART B: Short Answer Questions (21 marks)

### 16. Compare L1 and L2 regularization techniques. (3 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI4 | **Difficulty**: Moderate

---

### 17. What is the universal approximation theorem and its significance? (3 marks)
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL4 | **PI**: CO1-PI1 | **Difficulty**: Moderate

---

### 18. Explain the role of bias in neural networks. (3 marks)
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL3 | **PI**: CO1-PI1 | **Difficulty**: Easy

---

### 19. What is batch normalization and what problem does it solve? (3 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI5 | **Difficulty**: Moderate

---

### 20. Compare underfitting and overfitting with their causes and solutions. (3 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI3 | **Difficulty**: Moderate

---

### 21. What is the purpose of activation functions in neural networks? (3 marks)
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL3 | **PI**: CO1-PI3 | **Difficulty**: Easy

---

### 22. What is gradient clipping and when is it used? (3 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI2 | **Difficulty**: Moderate

---

## PART C: Descriptive Questions (14 marks)

### 23. Explain the vanishing gradient problem in detail. Discuss three different solutions with their mechanisms. (7 marks)
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI2 | **Difficulty**: Moderate

---

### 24. Compare and contrast five different optimization algorithms used in deep learning. Include convergence properties, memory requirements, and use cases. (7 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI1 | **Difficulty**: Moderate

---

# TEST SET C
**Balanced Difficulty Distribution**: Easy (10), Moderate (12), Difficult (3)
**Mark Distribution**: 1-mark (15), 2-mark (7), 5-mark (3)

---

## PART A: Multiple Choice Questions (15 marks)

### 1. The bias term in a neuron:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) Prevents overfitting
b) Allows shifting of the activation function
c) Reduces computational complexity
d) Eliminates the need for weights

---

### 2. In forward propagation, the output of each layer is:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI4 | **Marks**: 1 | **Difficulty**: Easy

a) Input to the previous layer
b) Input to the next layer
c) Stored for backward pass only
d) Discarded immediately

---

### 3. In mini-batch gradient descent, the batch size affects:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) Only computational efficiency
b) Only gradient noise
c) Both computational efficiency and gradient noise
d) Neither efficiency nor noise

---

### 4. Which activation function helps mitigate the vanishing gradient problem?
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI2 | **Marks**: 1 | **Difficulty**: Moderate

a) Sigmoid
b) Tanh
c) ReLU
d) Linear

---

### 5. The dropout rate typically ranges between:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI4 | **Marks**: 1 | **Difficulty**: Easy

a) 0 to 0.1
b) 0.2 to 0.5
c) 0.6 to 0.9
d) 0.9 to 1.0

---

### 6. In backpropagation, gradients are computed using:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI2 | **Marks**: 1 | **Difficulty**: Moderate

a) Forward pass only
b) Chain rule of calculus
c) Random initialization
d) Linear algebra only

---

### 7. The learning rate schedule that reduces learning rate over time is called:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL1 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) Learning rate decay
b) Learning rate explosion
c) Learning rate normalization
d) Learning rate regularization

---

### 8. Which of the following is NOT a hyperparameter in neural networks?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL3 | **PI**: CO1-PI1 | **Marks**: 1 | **Difficulty**: Moderate

a) Learning rate
b) Number of hidden layers
c) Weight values after training
d) Batch size

---

### 9. L1 regularization tends to produce:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI4 | **Marks**: 1 | **Difficulty**: Moderate

a) Dense weight matrices
b) Sparse weight matrices
c) Larger weight values
d) Negative weight values

---

### 10. The perceptron learning algorithm can solve:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) Only linearly separable problems
b) Any classification problem
c) Only regression problems
d) Non-linear problems directly

---

### 11. Which problem is addressed by batch normalization?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI5 | **Marks**: 1 | **Difficulty**: Moderate

a) Overfitting only
b) Internal covariate shift
c) Underfitting only
d) Memory optimization

---

### 12. Gradient clipping is implemented by:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI2 | **Marks**: 1 | **Difficulty**: Moderate

a) Setting gradients to zero
b) Limiting the magnitude of gradients
c) Reversing gradient direction
d) Adding noise to gradients

---

### 13. Which statement about stochastic gradient descent is true?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Marks**: 1 | **Difficulty**: Easy

a) Uses entire dataset for each update
b) Uses one sample for each update
c) Always converges to global minimum
d) Requires large memory

---

### 14. The bias-variance tradeoff is related to:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI3 | **Marks**: 1 | **Difficulty**: Difficult

a) Computational complexity
b) Overfitting and underfitting
c) Memory usage
d) Training time

---

### 15. The purpose of validation set is to:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI3 | **Marks**: 1 | **Difficulty**: Easy

a) Train the model
b) Tune hyperparameters and monitor overfitting
c) Test final performance
d) Increase training data

---

## PART B: Short Answer Questions (21 marks)

### 16. Explain the Adam optimizer and its advantages. (3 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Difficulty**: Moderate

---

### 17. Explain the concept of weight initialization and its importance. (3 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI2 | **Difficulty**: Moderate

---

### 18. What is early stopping and how does it prevent overfitting? (3 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI3 | **Difficulty**: Easy

---

### 19. Compare different types of gradient descent based on data usage. (3 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI1 | **Difficulty**: Moderate

---

### 20. What is the difference between parameters and hyperparameters? (3 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Difficulty**: Easy

---

### 21. Explain the difference between loss function and cost function. (3 marks)
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI4 | **Difficulty**: Easy

---

### 22. Compare different normalization techniques and their applications. (3 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI5 | **Difficulty**: Moderate

---

## PART C: Descriptive Questions (14 marks)

### 23. Design and implement a comprehensive regularization strategy for a deep neural network prone to overfitting. Include multiple techniques and their implementation details. (7 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL6 | **PI**: CO2-PI4 | **Difficulty**: Difficult

---

### 24. Derive the backpropagation algorithm for a simple two-layer neural network with mathematical equations. Include forward pass, loss computation, and gradient calculations. (7 marks)
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL5 | **PI**: CO2-PI2 | **Difficulty**: Difficult

---

# ANSWER KEYS FOR ALL SETS

## Set A Answer Key:
1. b) AND problem
2. b) (0, 1)
3. c) Batch Gradient Descent
4. b) Sigmoid
5. c) Dropout
6. b) A single hidden layer MLP can approximate any continuous function
7. c) AdaGrad
8. b) 1
9. c) Model performs well on training but poorly on test data
10. c) Binary Cross-entropy
11. a) Momentum and RMSprop
12. c) Tensor
13. c) Between layers during training
14. b) Maintain gradient flow through layers
15. b) The size of steps taken towards minimum

## Set B Answer Key:
1. b) It is not linearly separable
2. c) ReLU
3. b) Size of steps toward minimum
4. b) Gradient clipping
5. b) Sum of squared weights
6. b) Chain rule of differentiation
7. c) Batch normalization
8. c) Tanh
9. a) Stopping training when validation loss increases
10. b) tf.zeros()
11. b) Accelerates convergence in relevant directions
12. b) Multi-class classification
13. c) Increasing model complexity
14. c) Aggressive learning rate decay
15. b) Dense layer

## Set C Answer Key:
1. b) Allows shifting of the activation function
2. b) Input to the next layer
3. c) Both computational efficiency and gradient noise
4. c) ReLU
5. b) 0.2 to 0.5
6. b) Chain rule of calculus
7. a) Learning rate decay
8. c) Weight values after training
9. b) Sparse weight matrices
10. a) Only linearly separable problems
11. b) Internal covariate shift
12. b) Limiting the magnitude of gradients
13. b) Uses one sample for each update
14. b) Overfitting and underfitting
15. b) Tune hyperparameters and monitor overfitting

---

# ASSESSMENT STATISTICS

## Overall Distribution Across Sets:

| Set | Easy | Moderate | Difficult | CO-1 | CO-2 | Total Marks |
|-----|------|----------|-----------|------|------|-------------|
| A | 10 | 12 | 3 | 12 | 13 | 50 |
| B | 10 | 12 | 3 | 12 | 13 | 50 |
| C | 10 | 12 | 3 | 12 | 13 | 50 |

Each test set provides balanced assessment while maintaining variety in question selection to prevent cheating and ensure comprehensive evaluation of student understanding.