# Unit Test 1 Preparation Guide
**Modules 1 & 2 Comprehensive Review | Test Date: September 19, 2025**

---

## 🎯 Test Coverage Overview

### **Module 1: Introduction to Deep Learning (35% of test)**
- Perceptron and Multi-Layer Perceptrons
- XOR Problem and Linear Separability
- Activation Functions (Sigmoid, ReLU, Tanh)
- TensorFlow Basics and Tensor Operations

### **Module 2: Optimization & Regularization (65% of test)**
- Gradient Descent Variants (Batch, SGD, Mini-batch)
- Vanishing/Exploding Gradient Problems
- Regularization Techniques (L1, L2, Dropout, BatchNorm)
- Overfitting vs Underfitting
- Bias-Variance Tradeoff

---

## 📖 Module 1: Critical Concepts

### **1. Perceptron Fundamentals**

#### **Mathematical Foundation**
```
Perceptron Output: y = σ(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
                  y = σ(w·x + b)

Where σ is the activation function
```

#### **Key Questions to Master**
- Why can't a single perceptron solve XOR?
- What makes a problem linearly separable?
- How do weights and biases affect decision boundaries?

#### **Expected Problem Types**
```python
# Implementation question
def perceptron_forward(X, weights, bias):
    """Implement perceptron forward pass"""
    z = np.dot(X, weights) + bias
    return sigmoid(z)
```

### **2. XOR Problem - The Classic**

#### **Why It's Important**
- Demonstrates limitation of linear models
- Shows need for hidden layers
- Historical significance in neural networks

#### **Mathematical Proof**
```
XOR Truth Table:
(0,0) → 0
(0,1) → 1
(1,0) → 1
(1,1) → 0

No single line can separate 1s from 0s!
```

#### **MLP Solution**
```python
# Two-layer solution
hidden = sigmoid(W1·x + b1)
output = sigmoid(W2·hidden + b2)
```

### **3. Activation Functions Comparison**

| Function | Formula | Range | Gradient | Use Case |
|----------|---------|-------|----------|----------|
| **Sigmoid** | 1/(1+e^(-x)) | (0,1) | σ(x)(1-σ(x)) | Output layer (binary) |
| **Tanh** | (e^x-e^(-x))/(e^x+e^(-x)) | (-1,1) | 1-tanh²(x) | Hidden layers |
| **ReLU** | max(0,x) | [0,∞) | 1 if x>0, 0 if x≤0 | Hidden layers (preferred) |

#### **Critical Understanding**
- **Vanishing Gradients**: Why sigmoid/tanh cause problems in deep networks
- **ReLU Advantages**: No saturation, computationally efficient
- **When to Use Each**: Context-dependent selection

---

## 📖 Module 2: Critical Concepts

### **1. Gradient Descent Variants**

#### **Batch Gradient Descent**
```python
# Uses entire dataset
for epoch in range(num_epochs):
    gradient = compute_gradient(X_all, y_all, weights)
    weights = weights - learning_rate * gradient
```
- **Pros**: Stable convergence, accurate gradients
- **Cons**: Slow for large datasets, memory intensive

#### **Stochastic Gradient Descent (SGD)**
```python
# Uses one sample at a time
for epoch in range(num_epochs):
    for i in range(len(X)):
        gradient = compute_gradient(X[i], y[i], weights)
        weights = weights - learning_rate * gradient
```
- **Pros**: Fast updates, can escape local minima
- **Cons**: Noisy convergence, high variance

#### **Mini-batch Gradient Descent**
```python
# Uses small batches
for epoch in range(num_epochs):
    for batch in create_batches(X, y, batch_size):
        gradient = compute_gradient(batch_X, batch_y, weights)
        weights = weights - learning_rate * gradient
```
- **Pros**: Balance of stability and speed
- **Cons**: Hyperparameter tuning needed

### **2. Regularization Techniques Deep Dive**

#### **L1 Regularization (LASSO)**
```
Loss_total = Loss_original + λ∑|wᵢ|

Properties:
- Creates sparse solutions (drives weights to 0)
- Feature selection capability
- Diamond-shaped constraint region
```

#### **L2 Regularization (Ridge)**
```
Loss_total = Loss_original + λ∑wᵢ²

Properties:
- Shrinks weights toward zero
- Keeps all features
- Circular constraint region
```

#### **Dropout**
```python
# During training
def dropout_layer(x, keep_prob):
    mask = np.random.binomial(1, keep_prob, size=x.shape)
    return x * mask / keep_prob

# During inference (automatic in TensorFlow)
output = x  # No modification needed
```

#### **Batch Normalization**
```
Step 1: μ = (1/m)∑xᵢ
Step 2: σ² = (1/m)∑(xᵢ - μ)²
Step 3: x̂ᵢ = (xᵢ - μ)/√(σ² + ε)
Step 4: yᵢ = γx̂ᵢ + β

Where γ and β are learnable parameters
```

### **3. Overfitting vs Underfitting**

#### **Identification Patterns**
```
Overfitting:
- Training accuracy: 95%+
- Validation accuracy: 70%-
- Gap > 10%

Underfitting:
- Training accuracy: 70%-
- Validation accuracy: 65%-
- Both low, small gap

Good Fit:
- Training accuracy: 85%
- Validation accuracy: 82%
- Gap < 5%
```

#### **Solutions**
```
Overfitting Solutions:
1. Add regularization (L1/L2/Dropout)
2. Reduce model complexity
3. Increase training data
4. Early stopping

Underfitting Solutions:
1. Increase model complexity
2. Reduce regularization
3. Train longer
4. Better feature engineering
```

---

## 🧮 Mathematical Derivations You Must Know

### **1. Sigmoid Derivative**
```
Given: σ(x) = 1/(1 + e^(-x))
Prove: σ'(x) = σ(x)(1 - σ(x))

Solution:
σ'(x) = d/dx[1/(1 + e^(-x))]
      = e^(-x)/(1 + e^(-x))²
      = [1/(1 + e^(-x))] × [e^(-x)/(1 + e^(-x))]
      = σ(x) × [1 - 1/(1 + e^(-x))]
      = σ(x)(1 - σ(x))
```

### **2. L2 Regularization Gradient**
```
Given: L = Loss_original + λ∑wᵢ²
Find: ∂L/∂w

Solution:
∂L/∂w = ∂Loss_original/∂w + λ(2w)
      = ∂Loss_original/∂w + 2λw

Update rule:
w_new = w_old - lr × (∂Loss_original/∂w + 2λw_old)
```

### **3. Batch Normalization Backward Pass**
```
Forward: y = γ((x-μ)/σ) + β

Backward gradients:
∂L/∂x = (γ/σ)[∂L/∂y - (1/m)∑∂L/∂y - ((x-μ)/σ²)(1/m)∑(x-μ)∂L/∂y]
∂L/∂γ = ∑∂L/∂y × ((x-μ)/σ)
∂L/∂β = ∑∂L/∂y
```

---

## 💻 Implementation Patterns to Memorize

### **1. Complete Neural Network from Scratch**
```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []

        # Initialize weights
        for i in range(len(layers)-1):
            w = np.random.randn(layers[i], layers[i+1]) * 0.1
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, X):
        self.activations = [X]
        current_input = X

        for i in range(len(self.weights)):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            if i == len(self.weights) - 1:  # Output layer
                current_input = self.sigmoid(z)
            else:  # Hidden layers
                current_input = np.maximum(0, z)  # ReLU
            self.activations.append(current_input)

        return current_input

    def compute_loss(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred + 1e-8) +
                       (1 - y_true) * np.log(1 - y_pred + 1e-8))
```

### **2. TensorFlow Implementation Template**
```python
import tensorflow as tf

def create_regularized_model(input_dim, hidden_units, output_dim,
                           dropout_rate=0.3, l2_reg=0.01):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_units[0], activation='relu',
                             input_shape=(input_dim,),
                             kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Dense(hidden_units[1], activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Dense(output_dim, activation='sigmoid')
    ])

    return model

# Callbacks for training
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                   restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]
```

---

## 🎯 Practice Problems

### **Problem 1: Mathematical Derivation**
**Question**: Given a neural network with sigmoid activation, derive why the vanishing gradient problem occurs in deep networks.

**Solution Framework**:
1. Show sigmoid derivative formula
2. Demonstrate maximum gradient value (0.25)
3. Show how gradients multiply through layers
4. Calculate gradient magnitude after n layers

### **Problem 2: Implementation**
**Question**: Implement L1 regularization in a custom gradient descent optimizer.

```python
def gradient_descent_with_l1(X, y, weights, lr=0.01, l1_lambda=0.01, epochs=100):
    for epoch in range(epochs):
        # Forward pass
        predictions = sigmoid(np.dot(X, weights))

        # Compute gradients
        gradient = np.dot(X.T, (predictions - y)) / len(y)

        # Add L1 regularization
        l1_gradient = l1_lambda * np.sign(weights)
        total_gradient = gradient + l1_gradient

        # Update weights
        weights -= lr * total_gradient

    return weights
```

### **Problem 3: Analysis**
**Question**: Given training curves, identify the problem and suggest solutions.

**Analysis Framework**:
1. Compare training vs validation performance
2. Look at convergence patterns
3. Identify overfitting/underfitting
4. Suggest appropriate regularization

---

## ⏰ Day-of-Test Strategy

### **Time Management (50 minutes total)**
- **MCQs (10 questions)**: 15 minutes (1.5 min each)
- **SAQs (3 of 5 questions)**: 30 minutes (10 min each)
- **Review**: 5 minutes

### **Question Priority**
1. **Easy MCQs first**: Build confidence
2. **Known SAQs second**: Secure marks
3. **Challenging problems last**: Risk management

### **Common Pitfalls to Avoid**
- Don't spend too long on one question
- Show all mathematical steps
- Include proper TensorFlow syntax
- Explain conceptual reasoning clearly

---

## 📚 Final Checklist

### **Mathematical Formulas** ✅
- [ ] Perceptron equation
- [ ] Sigmoid derivative
- [ ] L1/L2 regularization formulas
- [ ] Batch normalization equations
- [ ] Gradient descent update rules

### **Implementation Patterns** ✅
- [ ] TensorFlow model creation
- [ ] Regularization layer additions
- [ ] Callback configurations
- [ ] Training loop structure

### **Conceptual Understanding** ✅
- [ ] XOR problem explanation
- [ ] Activation function selection criteria
- [ ] Regularization technique comparison
- [ ] Overfitting identification and solutions

### **Problem-Solving Skills** ✅
- [ ] Mathematical derivation strategies
- [ ] Code debugging approaches
- [ ] Performance analysis techniques
- [ ] Hyperparameter selection logic

---

**Final Tip**: Focus on understanding the "why" behind each concept, not just memorizing formulas. The test will reward deep understanding over rote memorization!

**Good luck! 🚀 You've got this!**