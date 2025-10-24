# Deep Neural Networks - Detailed Answer Key
**Course Code**: 21CSE558T | **Test**: Formative Assessment I | **Coverage**: Modules 1-2

---

# PART A: 1-MARK QUESTIONS - DETAILED SOLUTIONS

## Q1. Which of the following problems can be solved by a single perceptron?
**Answer**: b) AND problem

**Detailed Explanation**:
- A single perceptron can only solve linearly separable problems
- AND gate: Inputs (0,0)→0, (0,1)→0, (1,0)→0, (1,1)→1
- These points can be separated by a straight line: w₁x₁ + w₂x₂ + b ≥ 0
- XOR is not linearly separable, requiring a non-linear decision boundary
- Weights: w₁=1, w₂=1, b=-1.5 can solve AND gate

---

## Q2. The XOR problem cannot be solved by a single perceptron because:
**Answer**: b) It is not linearly separable

**Detailed Explanation**:
- XOR truth table: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0
- No single straight line can separate the positive and negative cases
- Requires a non-linear decision boundary (multiple lines or curves)
- Historical significance: Led to the development of multi-layer perceptrons
- Proof: Minsky & Papert (1969) showed mathematical impossibility

---

## Q3. In TensorFlow, which data structure is the fundamental building block?
**Answer**: c) Tensor

**Detailed Explanation**:
- Tensor: N-dimensional array that generalizes scalars, vectors, and matrices
- TensorFlow operations work on tensors of various ranks (0D, 1D, 2D, 3D, etc.)
- Examples: Scalar (rank 0), Vector (rank 1), Matrix (rank 2), Image batch (rank 4)
- All computations in TensorFlow are represented as computational graphs of tensor operations
- Data flows through the graph as tensors, hence "TensorFlow"

---

## Q4. The sigmoid activation function outputs values in the range:
**Answer**: b) (0, 1)

**Detailed Explanation**:
- Sigmoid function: σ(x) = 1/(1 + e^(-x))
- As x → +∞, σ(x) → 1 (asymptotically approaches but never reaches)
- As x → -∞, σ(x) → 0 (asymptotically approaches but never reaches)
- Useful for binary classification as output layer (probability interpretation)
- Smooth, differentiable function with S-shaped curve

---

## Q5. Which activation function is most commonly used in hidden layers of modern deep networks?
**Answer**: c) ReLU

**Detailed Explanation**:
- ReLU: f(x) = max(0, x) - simple and computationally efficient
- Advantages: Solves vanishing gradient problem, sparse activation, fast computation
- Derivative: 1 for x > 0, 0 for x ≤ 0 (enables better gradient flow)
- Introduced biological inspiration: neurons either fire or don't fire
- Replaced sigmoid/tanh in hidden layers due to superior performance in deep networks

---

## Q6. The backpropagation algorithm uses which mathematical concept?
**Answer**: b) Chain rule of differentiation

**Detailed Explanation**:
- Chain rule: d/dx[f(g(x))] = f'(g(x)) × g'(x)
- Neural networks are compositions of functions: output = f₃(f₂(f₁(x)))
- Backpropagation computes gradients by applying chain rule layer by layer
- ∂Loss/∂w₁ = ∂Loss/∂f₃ × ∂f₃/∂f₂ × ∂f₂/∂f₁ × ∂f₁/∂w₁
- Enables efficient gradient computation in O(n) time instead of exponential

---

## Q7. Which loss function is typically used for binary classification?
**Answer**: c) Binary Cross-entropy

**Detailed Explanation**:
- Binary cross-entropy: L = -[y×log(ŷ) + (1-y)×log(1-ŷ)]
- Suitable for sigmoid output layer (probability interpretation)
- Penalizes wrong predictions more severely as confidence increases
- Convex function ensuring global minimum exists
- Information-theoretic interpretation: measures difference between distributions

---

## Q8. In a multilayer perceptron, the universal approximation theorem states:
**Answer**: b) A single hidden layer can approximate any continuous function

**Detailed Explanation**:
- Proven by Cybenko (1989) and Hornik (1991)
- Given sufficient neurons, one hidden layer can approximate any continuous function
- Practical limitation: May require exponentially many neurons
- Doesn't specify how to find the weights or guarantee efficient learning
- Deep networks often more efficient for complex functions than wide shallow networks

---

## Q9. Which TensorFlow function is used to create a tensor filled with zeros?
**Answer**: b) tf.zeros()

**Detailed Explanation**:
- tf.zeros(shape, dtype=tf.float32): Creates tensor filled with zeros
- Example: tf.zeros([3, 4]) creates 3×4 matrix of zeros
- Commonly used for weight initialization (though not optimal)
- Other initialization functions: tf.ones(), tf.random.normal(), tf.random.uniform()
- Shape parameter defines dimensions: [batch_size, features] for 2D

---

## Q10. The derivative of ReLU function for positive inputs is:
**Answer**: b) 1

**Detailed Explanation**:
- ReLU: f(x) = max(0, x)
- For x > 0: f(x) = x, so f'(x) = 1
- For x < 0: f(x) = 0, so f'(x) = 0
- At x = 0: derivative is undefined (usually taken as 0 or 1 in practice)
- Constant derivative of 1 prevents vanishing gradient problem
- Enables deep networks to train effectively

---

[Continue with remaining questions Q11-Q45...]

---

# PART B: 2-MARK QUESTIONS - DETAILED SOLUTIONS

## Q46. Compare sigmoid and ReLU activation functions in terms of their properties and use cases.

**Detailed Answer** (2 marks):

**Sigmoid Function: σ(x) = 1/(1 + e^(-x))**
- **Range**: (0, 1) - bounded output
- **Properties**: Smooth, differentiable everywhere, S-shaped curve
- **Derivative**: σ'(x) = σ(x)(1-σ(x)), maximum value 0.25
- **Problems**: Vanishing gradient problem in deep networks
- **Use cases**: Output layer for binary classification (probability interpretation)

**ReLU Function: f(x) = max(0, x)**
- **Range**: [0, ∞) - unbounded for positive inputs
- **Properties**: Piecewise linear, not differentiable at x=0
- **Derivative**: 1 for x>0, 0 for x<0
- **Advantages**: Solves vanishing gradients, computationally efficient, sparse activation
- **Use cases**: Hidden layers in deep networks, default choice for most applications

**Key Differences**: ReLU enables training of deeper networks due to better gradient flow, while sigmoid is limited to shallow networks or output layers.

---

## Q47. Explain the difference between batch gradient descent and stochastic gradient descent.

**Detailed Answer** (2 marks):

**Batch Gradient Descent (BGD)**:
- **Data Usage**: Entire dataset for each parameter update
- **Update Rule**: θ = θ - α × (1/m) × Σ∇L(x_i)
- **Advantages**: Stable convergence, exact gradient computation, guaranteed convergence to global minimum (convex functions)
- **Disadvantages**: Slow for large datasets, high memory requirements, can get stuck in local minima

**Stochastic Gradient Descent (SGD)**:
- **Data Usage**: Single random sample for each update
- **Update Rule**: θ = θ - α × ∇L(x_i)
- **Advantages**: Fast updates, low memory usage, can escape local minima due to noise
- **Disadvantages**: Noisy convergence, may not reach exact minimum, requires learning rate scheduling

**Practical Impact**: SGD is preferred for large datasets and online learning, while BGD is used for smaller datasets requiring precise convergence.

---

## Q48. What is the XOR problem and why can't a single perceptron solve it?

**Detailed Answer** (2 marks):

**XOR Problem Definition**:
- **Truth Table**: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0
- **Logic**: Output is 1 when inputs are different, 0 when same
- **Real-world significance**: Represents non-linear relationship that appears in many practical problems

**Why Single Perceptron Fails**:
- **Linear Separation**: Perceptron creates linear decision boundary: w₁x₁ + w₂x₂ + b = 0
- **XOR Requirement**: Need to separate points (0,1) and (1,0) from (0,0) and (1,1)
- **Geometric Impossibility**: No single straight line can achieve this separation
- **Mathematical Proof**: Any linear combination cannot satisfy all four XOR constraints simultaneously

**Historical Significance**: XOR problem led to "AI winter" in 1970s and motivated development of multi-layer perceptrons with hidden layers.

---

[Continue with remaining Q49-Q65...]

---

# PART C: 5-MARK QUESTIONS - COMPREHENSIVE SOLUTIONS

## Q66. Derive the backpropagation algorithm for a simple two-layer neural network with mathematical equations.

**Comprehensive Solution** (5 marks):

**Network Architecture**:
- Input layer: x ∈ ℝⁿ
- Hidden layer: h₁ ∈ ℝᵐ with weights W₁ ∈ ℝⁿˣᵐ, bias b₁ ∈ ℝᵐ
- Output layer: ŷ ∈ ℝᵏ with weights W₂ ∈ ℝᵐˣᵏ, bias b₂ ∈ ℝᵏ

**Forward Pass Equations**:
1. **Hidden layer**: z₁ = W₁ᵀx + b₁
2. **Hidden activation**: a₁ = σ(z₁) where σ is activation function
3. **Output layer**: z₂ = W₂ᵀa₁ + b₂
4. **Final output**: ŷ = σ(z₂)

**Loss Function**: L = ½||y - ŷ||² (mean squared error)

**Backward Pass (Backpropagation)**:

**Step 1: Output Layer Gradients**
- ∂L/∂ŷ = -(y - ŷ)
- ∂L/∂z₂ = ∂L/∂ŷ × ∂ŷ/∂z₂ = -(y - ŷ) ⊙ σ'(z₂)
- ∂L/∂W₂ = a₁ ⊗ (∂L/∂z₂) = a₁(∂L/∂z₂)ᵀ
- ∂L/∂b₂ = ∂L/∂z₂

**Step 2: Hidden Layer Gradients (Chain Rule)**
- ∂L/∂a₁ = W₂(∂L/∂z₂)
- ∂L/∂z₁ = ∂L/∂a₁ ⊙ σ'(z₁)
- ∂L/∂W₁ = x ⊗ (∂L/∂z₁)
- ∂L/∂b₁ = ∂L/∂z₁

**Update Rules** (Gradient Descent):
- W₂ ← W₂ - α(∂L/∂W₂)
- b₂ ← b₂ - α(∂L/∂b₂)
- W₁ ← W₁ - α(∂L/∂W₁)
- b₁ ← b₁ - α(∂L/∂b₁)

**Computational Complexity**: O(nm + mk) for forward pass, O(nm + mk) for backward pass, where n=input size, m=hidden size, k=output size.

---

## Q67. Write a complete Python program using TensorFlow/Keras to create and train a neural network for the XOR problem.

**Complete Implementation** (5 marks):

```python
# Import required libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Prepare XOR dataset
def create_xor_dataset():
    """Create XOR dataset with all possible combinations"""
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=np.float32)

    y = np.array([[0],
                  [1],
                  [1],
                  [0]], dtype=np.float32)

    return X, y

# Create the dataset
X_train, y_train = create_xor_dataset()

# Build the neural network model
def create_xor_model():
    """Create a neural network capable of solving XOR"""
    model = tf.keras.Sequential([
        # Hidden layer with 4 neurons (sufficient for XOR)
        tf.keras.layers.Dense(
            units=4,
            activation='relu',
            input_shape=(2,),
            name='hidden_layer'
        ),

        # Output layer for binary classification
        tf.keras.layers.Dense(
            units=1,
            activation='sigmoid',
            name='output_layer'
        )
    ])

    return model

# Create and compile the model
model = create_xor_model()

# Compile with appropriate loss and optimizer
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

# Display model architecture
print("Model Architecture:")
model.summary()

# Train the model
print("\nTraining the model...")
history = model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=4,
    verbose=0,  # Silent training
    validation_data=(X_train, y_train)
)

# Evaluate the model
print("\nEvaluation Results:")
loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
print(f"Final Loss: {loss:.4f}")
print(f"Final Accuracy: {accuracy:.4f}")

# Make predictions
print("\nPredictions:")
predictions = model.predict(X_train, verbose=0)
rounded_predictions = np.round(predictions)

for i, (input_vals, actual, pred, rounded) in enumerate(
    zip(X_train, y_train, predictions, rounded_predictions)
):
    print(f"Input: {input_vals}, Actual: {actual[0]}, "
          f"Predicted: {pred[0]:.4f}, Rounded: {int(rounded[0])}")

# Verify XOR logic
print(f"\nXOR Logic Verification:")
print(f"0 XOR 0 = {int(rounded_predictions[0][0])} (Expected: 0)")
print(f"0 XOR 1 = {int(rounded_predictions[1][0])} (Expected: 1)")
print(f"1 XOR 0 = {int(rounded_predictions[2][0])} (Expected: 1)")
print(f"1 XOR 1 = {int(rounded_predictions[3][0])} (Expected: 0)")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['binary_accuracy'], label='Training Accuracy')
plt.plot(history.history['val_binary_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Extract and display learned weights
print("\nLearned Weights and Biases:")
for i, layer in enumerate(model.layers):
    weights, biases = layer.get_weights()
    print(f"Layer {i+1} ({layer.name}):")
    print(f"  Weights shape: {weights.shape}")
    print(f"  Weights:\n{weights}")
    print(f"  Biases: {biases}")
    print()
```

**Key Features of the Implementation**:
1. **Data Preparation**: Complete XOR truth table with proper data types
2. **Architecture**: 4 neurons in hidden layer (sufficient for XOR complexity)
3. **Training**: 1000 epochs with Adam optimizer and binary cross-entropy loss
4. **Evaluation**: Comprehensive testing with accuracy metrics
5. **Visualization**: Training curves and learned parameters analysis

---

[Continue with remaining Q68-Q75...]

---

# MARKING SCHEME SUMMARY

## 1-Mark Questions (45 questions):
- **Correct Answer**: 1 mark
- **No partial credit**
- **Clear reasoning expected for true/false questions**

## 2-Mark Questions (20 questions):
- **Complete Answer**: 2 marks
- **Partial Answer**: 1 mark (missing one key component)
- **Incorrect/No Answer**: 0 marks

## 5-Mark Questions (10 questions):
- **Excellent (4.5-5 marks)**: Complete solution with clear explanation
- **Good (3.5-4 marks)**: Correct approach with minor errors
- **Satisfactory (2.5-3 marks)**: Basic understanding shown
- **Needs Improvement (1-2 marks)**: Partial understanding
- **Unsatisfactory (0-0.5 marks)**: No understanding demonstrated

## Course Outcome Assessment:
- **CO-1**: Questions 1-45 (varying complexity)
- **CO-2**: Questions 46-75 (application focused)

## Program Outcome Mapping:
- **PO-1**: Engineering knowledge application
- **PO-2**: Problem analysis and solution design
- **PO-3**: Modern tool usage and implementation