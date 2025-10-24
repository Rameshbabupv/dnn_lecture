# FORMATIVE ASSESSMENT 1 - SET A
**Course**: Deep Neural Network Architectures (21CSE558T)
**Duration**: 90 minutes
**Maximum Marks**: 50
**Date**: [To be filled]
**Semester**: II M.Tech CSE

---

## PART A - Multiple Choice Questions (20 marks)
*Choose the correct answer. Each question carries 2 marks.*

### Q1. Which activation function suffers from the vanishing gradient problem in deep networks?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI2 | **Marks**: 2

a) ReLU
b) Sigmoid
c) Tanh
d) Leaky ReLU

**Answer**: b) Sigmoid

---

### Q2. In TensorFlow/Keras, which layer type is used for regularization to prevent overfitting?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Marks**: 2

a) Dense
b) Dropout
c) Flatten
d) Embedding

**Answer**: b) Dropout

---

### Q3. The gradient descent variant that uses the entire dataset for each parameter update is:
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI3 | **Marks**: 2

a) Stochastic Gradient Descent
b) Mini-batch Gradient Descent
c) Batch Gradient Descent
d) Adam Optimizer

**Answer**: c) Batch Gradient Descent

---

### Q4. Which of the following is NOT a hyperparameter in neural networks?
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL3 | **PI**: CO1-PI1 | **Marks**: 2

a) Learning rate
b) Number of hidden layers
c) Weight values after training
d) Batch size

**Answer**: c) Weight values after training

---

### Q5. The perceptron learning algorithm can solve:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI3 | **Marks**: 2

a) Only linearly separable problems
b) Any classification problem
c) Only regression problems
d) Non-linear problems directly

**Answer**: a) Only linearly separable problems

---

### Q6. In backpropagation, gradients are computed using:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI2 | **Marks**: 2

a) Forward pass only
b) Chain rule of calculus
c) Random initialization
d) Linear algebra only

**Answer**: b) Chain rule of calculus

---

### Q7. Which regularization technique adds a penalty term to the loss function?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI4 | **Marks**: 2

a) Dropout
b) Early stopping
c) L2 regularization
d) Data augmentation

**Answer**: c) L2 regularization

---

### Q8. The universal approximation theorem states that:
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI4 | **Marks**: 2

a) Any function can be approximated by a linear model
b) A single hidden layer MLP can approximate any continuous function
c) Deep networks always outperform shallow networks
d) Neural networks can only solve linear problems

**Answer**: b) A single hidden layer MLP can approximate any continuous function

---

### Q9. In TensorFlow, which function is used to compile a Keras model?
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Marks**: 2

a) model.fit()
b) model.compile()
c) model.evaluate()
d) model.predict()

**Answer**: b) model.compile()

---

### Q10. The learning rate in gradient descent determines:
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI3 | **Marks**: 2

a) The number of epochs
b) The size of steps taken towards minimum
c) The number of hidden layers
d) The activation function type

**Answer**: b) The size of steps taken towards minimum

---

## PART B - Short Answer Questions (30 marks)

### Q11. Explain the vanishing gradient problem in deep neural networks. How does ReLU activation help mitigate this issue?
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI2 | **Marks**: 6

**Expected Answer**:
- Vanishing gradient problem occurs when gradients become exponentially small as they propagate backward through deep networks
- This happens due to repeated multiplication of small derivatives (especially with sigmoid/tanh activations)
- Results in very slow learning or no learning in earlier layers
- ReLU has derivative of 1 for positive inputs and 0 for negative inputs
- This prevents gradient vanishing for positive activations and allows better gradient flow
- ReLU is computationally efficient and helps train deeper networks effectively

---

### Q12. Write Python code using TensorFlow/Keras to create a simple MLP for binary classification with one hidden layer of 64 neurons. Include model compilation.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Marks**: 8

**Expected Answer**:
```python
import tensorflow as tf
from tensorflow import keras

# Create the model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()
```

---

### Q13. Compare and contrast Batch Gradient Descent, Stochastic Gradient Descent, and Mini-batch Gradient Descent. Include advantages and disadvantages of each.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI3 | **Marks**: 8

**Expected Answer**:

**Batch Gradient Descent:**
- Uses entire dataset for each update
- Advantages: Stable convergence, smooth gradient estimates
- Disadvantages: Slow for large datasets, high memory requirements

**Stochastic Gradient Descent:**
- Uses one sample for each update
- Advantages: Fast updates, can escape local minima, low memory
- Disadvantages: Noisy convergence, may not reach exact minimum

**Mini-batch Gradient Descent:**
- Uses small batches (typically 32-256 samples)
- Advantages: Balances speed and stability, efficient vectorization
- Disadvantages: Requires tuning batch size hyperparameter

---

### Q14. Explain the concept of overfitting in neural networks and describe three techniques to prevent it.
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI4 | **Marks**: 8

**Expected Answer**:

**Overfitting**: When model performs well on training data but poorly on unseen data due to memorizing training patterns rather than learning generalizable features.

**Prevention Techniques:**

1. **Dropout**: Randomly sets some neurons to zero during training, forcing network to not rely on specific neurons
2. **Early Stopping**: Monitor validation loss and stop training when it starts increasing
3. **L1/L2 Regularization**: Add penalty terms to loss function to discourage large weights
4. **Data Augmentation**: Increase training data variety through transformations
5. **Validation Split**: Use separate validation set to monitor generalization performance

---

## ANSWER KEY SUMMARY

### Part A (MCQs): 20 marks
1. b) Sigmoid
2. b) Dropout
3. c) Batch Gradient Descent
4. c) Weight values after training
5. a) Only linearly separable problems
6. b) Chain rule of calculus
7. c) L2 regularization
8. b) A single hidden layer MLP can approximate any continuous function
9. b) model.compile()
10. b) The size of steps taken towards minimum

### Part B Marking Scheme:
- Q11: 6 marks (2 marks explanation + 2 marks ReLU benefits + 2 marks connection)
- Q12: 8 marks (4 marks model creation + 2 marks compilation + 2 marks syntax)
- Q13: 8 marks (2 marks each method + 2 marks comparison)
- Q14: 8 marks (3 marks overfitting concept + 5 marks techniques)

**Total: 50 marks**

---

## Assessment Mapping Summary

| Question | PLO | CO | BL | PI | Topic Coverage |
|----------|-----|----|----|-------|----------------|
| Q1-Q5 | PLO-1 | CO-1 | BL2-BL3 | CO1-PI1 to PI4 | Basic NN concepts |
| Q6-Q10 | PLO-2 | CO-2 | BL2-BL3 | CO2-PI1 to PI4 | Optimization & Implementation |
| Q11 | PLO-1 | CO-2 | BL4 | CO2-PI2 | Deep learning challenges |
| Q12 | PLO-2 | CO-2 | BL3 | CO2-PI1 | Practical implementation |
| Q13-Q14 | PLO-2 | CO-2 | BL4 | CO2-PI3,PI4 | Advanced optimization concepts |

This assessment covers Modules 1-2 comprehensively, testing both theoretical understanding and practical implementation skills required for the Deep Neural Network Architectures course.