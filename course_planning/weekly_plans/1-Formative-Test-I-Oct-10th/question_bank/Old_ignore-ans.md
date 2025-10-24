# 5-Mark Questions - Answer Key with Detailed Explanations
**Deep Neural Network Architectures (21CSE558T)**
**Formative Test I - October 10th, 2024**

---

## Question 1: Single Perceptron Forward Pass [5 Marks]

### **ANSWER:**
**Final Output: 0.5250**

### **Step-by-Step Solution:**
1. **Calculate weighted sum (z):** z = (0.8 × 0.4) + (0.6 × -0.7) + 0.3 = 0.32 - 0.42 + 0.3 = 0.20
2. **Apply sigmoid activation:** σ(z) = 1/(1 + e^(-0.20)) = 1/(1 + 0.8187) = 1/1.8187 = **0.5250**

### **Detailed Explanation:**
This question tests your understanding of **fundamental perceptron operations** from Week 2 lectures. A perceptron performs two key operations:

**1. Linear Combination (Weighted Sum):**
- The perceptron computes z = Σ(wi × xi) + b
- This is essentially a dot product between input and weight vectors plus bias
- In our case: z = w1×x1 + w2×x2 + b = 0.4×0.8 + (-0.7)×0.6 + 0.3
- Each weight determines how much influence each input has on the final decision

**2. Non-linear Activation:**
- The sigmoid function σ(z) = 1/(1 + e^(-z)) maps any real number to (0,1)
- This introduces non-linearity, allowing the network to learn complex patterns
- Sigmoid is smooth and differentiable, making it suitable for gradient-based learning

**Key Concepts:**
- **Bias term (b):** Shifts the activation function, allowing the perceptron to fire even when inputs are zero
- **Weighted inputs:** Different weights give different importance to each input feature
- **Activation function:** Converts linear combination to a bounded, interpretable output

This forms the foundation for all neural network computations in TensorFlow/Keras implementations.

---

## Question 2: AND Gate Perceptron Implementation [5 Marks]

### **ANSWER:**
- **[0,0] → 0** ✓ Correct
- **[0,1] → 0** ✓ Correct
- **[1,0] → 0** ✓ Correct
- **[1,1] → 1** ✓ Correct

### **Step-by-Step Solution:**
1. **Input [0,0]:** z = (0×0.5) + (0×0.5) - 0.7 = -0.7 → Step(-0.7) = **0**
2. **Input [0,1]:** z = (0×0.5) + (1×0.5) - 0.7 = -0.2 → Step(-0.2) = **0**
3. **Input [1,0]:** z = (1×0.5) + (0×0.5) - 0.7 = -0.2 → Step(-0.2) = **0**
4. **Input [1,1]:** z = (1×0.5) + (1×0.5) - 0.7 = 0.3 → Step(0.3) = **1**

### **Detailed Explanation:**
This demonstrates the **linearly separable nature** of the AND gate, a core concept from Week 3.

**Linear Separability Concept:**
- AND gate is linearly separable because we can draw a straight line to separate TRUE outputs (1,1) from FALSE outputs
- The decision boundary is defined by: 0.5x₁ + 0.5x₂ - 0.7 = 0, or x₁ + x₂ = 1.4
- Points above this line output 1, points below output 0

**Weight and Bias Analysis:**
- **Weights [0.5, 0.5]:** Both inputs have equal importance
- **Bias -0.7:** Creates a threshold - both inputs must be significantly active to overcome this negative bias
- **Step function:** Hard threshold creates crisp binary decisions

**Historical Context:**
This is exactly how Rosenblatt's original perceptron (1957) worked. The limitation is that perceptrons cannot solve non-linearly separable problems like XOR, which led to the "AI winter" until multi-layer networks were developed.

**TensorFlow Implementation:**
```python
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='linear', input_shape=(2,))
])
# Then apply step function manually
```

---

## Question 3: Multi-Layer Forward Propagation [5 Marks]

### **ANSWER:**
**Final Output: 0.6900**

### **Step-by-Step Solution:**

**Hidden Layer Calculation:**
1. **Linear transformation:**
   - h₁ = (1.2×0.5) + (-0.8×0.2) + 0.1 = 0.6 - 0.16 + 0.1 = 0.54
   - h₂ = (1.2×-0.3) + (-0.8×0.9) + (-0.2) = -0.36 - 0.72 - 0.2 = -1.28
   - h₃ = (1.2×0.7) + (-0.8×-0.4) + 0.3 = 0.84 + 0.32 + 0.3 = 1.46

2. **ReLU activation:**
   - a₁ = ReLU(0.54) = 0.54
   - a₂ = ReLU(-1.28) = 0.00
   - a₃ = ReLU(1.46) = 1.46

**Output Layer Calculation:**
3. **Linear transformation:** z = (0.54×0.6) + (0.00×-0.5) + (1.46×0.8) + 0.2 = 0.324 + 0 + 1.168 + 0.2 = 1.692

4. **Sigmoid activation:** σ(1.692) = 1/(1 + e^(-1.692)) = 1/(1 + 0.1835) = **0.6900**

### **Detailed Explanation:**
This problem demonstrates **multi-layer perceptron (MLP) architecture**, the foundation of deep learning from Week 4.

**Key Architectural Components:**

**1. Hidden Layer with ReLU:**
- **ReLU function:** f(x) = max(0, x) - eliminates negative values
- **Benefits:** Solves vanishing gradient problem, computationally efficient, sparse activation
- **Effect:** Neuron h₂ becomes inactive (0), creating sparsity

**2. Matrix Operations in TensorFlow:**
- Hidden layer: H = X·W₁ + b₁, then apply ReLU
- Output layer: Y = H·W₂ + b₂, then apply sigmoid
- This is exactly how `tf.keras.layers.Dense` works internally

**3. Universal Approximation:**
- MLPs with even one hidden layer can approximate any continuous function
- The combination of linear transformations and non-linear activations creates this power
- More layers = more complex patterns but harder to train

**Network Capacity:**
- **3 hidden neurons:** Can create up to 3 decision boundaries in input space
- **ReLU activation:** Creates piecewise linear decision regions
- **Sigmoid output:** Maps to probability-like interpretation (0-1 range)

**TensorFlow Equivalent:**
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

This demonstrates how simple mathematical operations combine to create powerful learning systems.

---

## Question 4: Gradient Descent Weight Update [5 Marks]

### **ANSWER:**
- **Updated weight: w_new = 0.794**
- **Updated bias: b_new = 0.297**

### **Step-by-Step Solution:**

1. **Calculate loss:** L = ½(y - ŷ)² = ½(1.0 - 0.7)² = ½(0.3)² = **0.045**

2. **Calculate gradients:**
   - ∂L/∂ŷ = -(y - ŷ) = -(1.0 - 0.7) = **-0.3**
   - ∂L/∂w = (∂L/∂ŷ) × (∂ŷ/∂w) = -0.3 × x = -0.3 × 2.0 = **-0.6**
   - ∂L/∂b = (∂L/∂ŷ) × (∂ŷ/∂b) = -0.3 × 1 = **-0.3**

3. **Update parameters:**
   - w_new = w - α × (∂L/∂w) = 0.8 - 0.01 × (-0.6) = 0.8 + 0.006 = **0.806**
   - b_new = b - α × (∂L/∂b) = 0.3 - 0.01 × (-0.3) = 0.3 + 0.003 = **0.303**

### **Detailed Explanation:**
This demonstrates **gradient descent optimization**, the heart of neural network training from Week 4.

**Chain Rule Application:**
The gradients are calculated using the chain rule of calculus:
- We need ∂L/∂w, but L depends on ŷ, which depends on w
- Chain rule: ∂L/∂w = (∂L/∂ŷ) × (∂ŷ/∂w)
- For linear neuron: ŷ = wx + b, so ∂ŷ/∂w = x and ∂ŷ/∂b = 1

**Gradient Interpretation:**
- **Negative gradient (-0.6 for weight):** Loss decreases when weight increases
- **Direction:** Gradient points toward steepest increase; we move opposite direction
- **Magnitude:** Larger gradient = steeper slope = faster learning

**Learning Rate Effect:**
- **α = 0.01:** Small learning rate ensures stable convergence
- **Too large:** Could overshoot minimum and oscillate
- **Too small:** Very slow convergence, many iterations needed

**MSE Loss Function Choice:**
- **Smooth and differentiable:** Enables gradient-based optimization
- **Convex for linear models:** Guarantees global minimum
- **Quadratic penalty:** Larger errors penalized more heavily

**TensorFlow Implementation:**
```python
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_fn = tf.keras.losses.MeanSquaredError()
# TensorFlow automatically computes and applies these gradients
```

This is exactly how `model.compile(optimizer='sgd')` works internally in Keras.

---

## Question 5: Activation Function Comparison [5 Marks]

### **ANSWER:**

| Input | Sigmoid | ReLU | Tanh |
|-------|---------|------|------|
| -2.0  | 0.1192  | 0.0  | -0.9640 |
| 0.0   | 0.5000  | 0.0  | 0.0000 |
| 2.0   | 0.8808  | 2.0  | 0.9640 |

### **Step-by-Step Solution:**

**For input x = -2.0:**
- Sigmoid: σ(-2) = 1/(1 + e²) = 1/(1 + 7.389) = **0.1192**
- ReLU: max(0, -2) = **0.0**
- Tanh: (e² - e⁻²)/(e² + e⁻²) = (7.389 - 0.135)/(7.389 + 0.135) = **-0.9640**

**For input x = 0.0:**
- Sigmoid: σ(0) = 1/(1 + e⁰) = 1/(1 + 1) = **0.5000**
- ReLU: max(0, 0) = **0.0**
- Tanh: (e⁰ - e⁰)/(e⁰ + e⁰) = (1 - 1)/(1 + 1) = **0.0000**

**For input x = 2.0:**
- Sigmoid: σ(2) = 1/(1 + e⁻²) = 1/(1 + 0.135) = **0.8808**
- ReLU: max(0, 2) = **2.0**
- Tanh: (e² - e⁻²)/(e² + e⁻²) = (7.389 - 0.135)/(7.389 + 0.135) = **0.9640**

### **Detailed Explanation:**
This comparison highlights **critical activation function properties** from Week 5 lectures.

**Sigmoid Function (σ(x) = 1/(1 + e⁻ˣ)):**
- **Range:** (0, 1) - always positive
- **Properties:** Smooth, differentiable, S-shaped curve
- **Problems:** Vanishing gradients for large |x|, outputs not zero-centered
- **Use cases:** Binary classification output layer, historically in hidden layers

**ReLU Function (f(x) = max(0, x)):**
- **Range:** [0, ∞) - eliminates negative values
- **Properties:** Simple, computationally efficient, sparse activation
- **Problems:** Dead neurons (always output 0 for negative inputs)
- **Advantages:** Solves vanishing gradient problem, faster training

**Tanh Function (f(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)):**
- **Range:** (-1, 1) - zero-centered outputs
- **Properties:** Similar to sigmoid but symmetric around origin
- **Advantages:** Zero-centered helps with gradient flow
- **Problems:** Still suffers from vanishing gradients

**Computational Efficiency Analysis:**
1. **ReLU:** Fastest - simple max operation
2. **Sigmoid/Tanh:** Slower - require exponential calculations
3. **Memory:** ReLU creates sparse representations (many zeros)

**Gradient Properties:**
- **ReLU gradient:** 1 for positive inputs, 0 for negative (binary)
- **Sigmoid gradient:** Maximum 0.25 at x=0, approaches 0 for large |x|
- **Tanh gradient:** Maximum 1 at x=0, better than sigmoid

**TensorFlow Usage:**
```python
tf.keras.layers.Dense(units, activation='relu')    # Most common
tf.keras.layers.Dense(units, activation='sigmoid') # Output layer for binary
tf.keras.layers.Dense(units, activation='tanh')    # Less common nowadays
```

Modern deep networks predominantly use ReLU in hidden layers due to training efficiency.

---

## Question 6: Dropout Regularization Application [5 Marks]

### **ANSWER:**
**Final Output: [1.143, 0.0, 1.286, 1.0, 0.0]**

### **Step-by-Step Solution:**

1. **Calculate scaling factor:** scale = 1/(1 - dropout_rate) = 1/(1 - 0.3) = 1/0.7 = **1.4286**

2. **Apply dropout mask and scaling:**
   - Position 0: 0.8 × 1 × 1.4286 = **1.143**
   - Position 1: 0.6 × 0 × 1.4286 = **0.0**
   - Position 2: 0.9 × 1 × 1.4286 = **1.286**
   - Position 3: 0.7 × 1 × 1.4286 = **1.0**
   - Position 4: 0.5 × 0 × 1.4286 = **0.0**

### **Detailed Explanation:**
This demonstrates **dropout regularization**, a powerful technique from Week 5 to prevent overfitting.

**Dropout Mechanism:**
1. **Random masking:** Each neuron has probability p of being "dropped" (set to 0)
2. **Scaling compensation:** Remaining neurons scaled by 1/(1-p) to maintain expected output
3. **Training vs. inference:** Dropout only applied during training, not during testing

**Why Scaling is Necessary:**
- **Expected value preservation:** Without scaling, network output would be systematically lower
- **Mathematical justification:** E[output_train] = E[output_test]
- **Example:** If 30% neurons dropped, remaining 70% must be amplified by 1/0.7 = 1.43

**Regularization Effect:**
- **Prevents co-adaptation:** Neurons cannot rely on specific other neurons
- **Improves generalization:** Forces network to use redundant representations
- **Ensemble effect:** Each forward pass uses different sub-network

**Dropout Rates in Practice:**
- **Hidden layers:** Typically 0.2-0.5 (20%-50% dropout)
- **Input layer:** Lower rates 0.1-0.2 if used
- **Output layer:** Never apply dropout
- **Our example:** 0.3 (30%) is moderate and commonly used

**Training vs. Inference Behavior:**
- **Training time:** Apply random masking + scaling
- **Test time:** Use all neurons, no masking, no scaling
- **Inverted dropout:** Scale during training (what we did)
- **Standard dropout:** Scale during testing (alternative approach)

**TensorFlow Implementation:**
```python
tf.keras.layers.Dropout(0.3)  # 30% dropout rate
# Automatically handles scaling and train/test differences
```

**Biological Inspiration:**
- **Brain analogy:** Some neurons may be "damaged" or unavailable
- **Redundancy:** Brain maintains function despite neuron loss
- **Robust representations:** Multiple pathways to same information

Dropout was revolutionary (Hinton et al., 2012) and remains one of the most effective regularization techniques in deep learning.

---

## Question 7: Batch Normalization Calculation [5 Marks]

### **ANSWER:**
**Final Output: [0.14, 0.86, 1.57, 2.29]**

### **Step-by-Step Solution:**

1. **Calculate batch statistics:**
   - Batch mean: μ = (2.0 + 4.0 + 6.0 + 8.0)/4 = **5.0**
   - Batch variance: σ² = [(2-5)² + (4-5)² + (6-5)² + (8-5)²]/4 = [9 + 1 + 1 + 9]/4 = **5.0**

2. **Normalize each input:**
   - x₁: (2.0 - 5.0)/√(5.0 + 1e-5) = -3.0/2.236 = **-1.342**
   - x₂: (4.0 - 5.0)/√(5.0 + 1e-5) = -1.0/2.236 = **-0.447**
   - x₃: (6.0 - 5.0)/√(5.0 + 1e-5) = 1.0/2.236 = **0.447**
   - x₄: (8.0 - 5.0)/√(5.0 + 1e-5) = 3.0/2.236 = **1.342**

3. **Apply scale and shift:**
   - y₁: 1.5 × (-1.342) + 0.5 = -2.013 + 0.5 = **-1.513**
   - y₂: 1.5 × (-0.447) + 0.5 = -0.671 + 0.5 = **-0.171**
   - y₃: 1.5 × (0.447) + 0.5 = 0.671 + 0.5 = **1.171**
   - y₄: 1.5 × (1.342) + 0.5 = 2.013 + 0.5 = **2.513**

### **Detailed Explanation:**
**Batch Normalization** is a crucial technique from Week 6 that revolutionized deep learning training.

**Mathematical Foundation:**
The complete batch norm operation: y = γ × ((x - μ)/√(σ² + ε)) + β
- **μ (mu):** Batch mean - centers the distribution
- **σ² (sigma squared):** Batch variance - controls spread
- **ε (epsilon):** Small constant to prevent division by zero
- **γ (gamma):** Learnable scale parameter
- **β (beta):** Learnable shift parameter

**Why Each Step Matters:**

**1. Normalization Step:**
- **Centers data:** Subtracting mean makes distribution zero-centered
- **Standardizes variance:** Dividing by std deviation makes variance = 1
- **Result:** Standard normal distribution (mean=0, std=1)

**2. Scale and Shift:**
- **Restores expressiveness:** Pure normalization limits what network can represent
- **Learnable parameters:** γ and β adapt during training
- **Identity mapping:** If γ=1, β=0, output equals normalized input

**Benefits of Batch Normalization:**
1. **Faster training:** Higher learning rates possible
2. **Reduced internal covariate shift:** Stabilizes layer inputs
3. **Regularization effect:** Slight noise from batch statistics
4. **Less sensitive to initialization:** Network more robust

**Internal Covariate Shift:**
- **Problem:** As lower layers learn, upper layer inputs change distribution
- **Effect:** Upper layers must constantly readjust
- **Solution:** BN keeps inputs to each layer stable

**Training vs. Inference:**
- **Training:** Use batch statistics (μ, σ² from current batch)
- **Inference:** Use running averages accumulated during training
- **Batch size dependency:** Small batches give noisy statistics

**TensorFlow Implementation:**
```python
tf.keras.layers.BatchNormalization()
# Automatically tracks running mean/variance
# Handles training vs inference mode
```

**Historical Impact:**
- **Before BN:** Very deep networks difficult to train
- **After BN:** Enabled ResNet (152+ layers), faster convergence
- **Modern usage:** Standard component in most architectures

The small ε (1e-5) prevents numerical instability when variance is very small, ensuring stable training.

---

## Question 8: OR Gate Perceptron Verification [5 Marks]

### **ANSWER:**
All outputs **CORRECT** - Successfully implements OR gate

| Input | Calculation | Output | Expected | ✓ |
|-------|-------------|--------|----------|---|
| [0,0] | 0.6×0 + 0.6×0 - 0.2 = -0.2 → 0 | 0 | 0 | ✓ |
| [0,1] | 0.6×0 + 0.6×1 - 0.2 = 0.4 → 1 | 1 | 1 | ✓ |
| [1,0] | 0.6×1 + 0.6×0 - 0.2 = 0.4 → 1 | 1 | 1 | ✓ |
| [1,1] | 0.6×1 + 0.6×1 - 0.2 = 1.0 → 1 | 1 | 1 | ✓ |

### **Detailed Explanation:**
This demonstrates **logical function implementation** using perceptrons, a fundamental concept from Week 3.

**OR Gate Logic:**
- **Truth table:** Output is 1 if ANY input is 1
- **Mathematical:** f(x₁, x₂) = x₁ OR x₂
- **Linear separability:** OR is linearly separable (unlike XOR)

**Decision Boundary Analysis:**
- **Equation:** 0.6x₁ + 0.6x₂ - 0.2 = 0
- **Simplified:** x₁ + x₂ = 0.33
- **Interpretation:** Any point where sum of inputs ≥ 0.33 outputs 1

**Weight Configuration Logic:**
- **Equal weights [0.6, 0.6]:** Both inputs have equal importance
- **Positive weights:** Both inputs contribute positively to output
- **Bias -0.2:** Low threshold - easy to activate with any positive input

**Comparison with AND Gate:**
- **AND gate:** Required bias = -0.7 (higher threshold)
- **OR gate:** Requires bias = -0.2 (lower threshold)
- **Intuition:** OR should activate more easily than AND

**Linear Separability Visualization:**
```
  x₂
   ↑
 1 |  T  |  T    (T = True/1, F = False/0)
   |-----|----
 0 |  F  |  T
   |_____|_____→ x₁
   0     1
```
Decision line x₁ + x₂ = 0.33 separates F from T regions.

**Historical Significance:**
- **Rosenblatt's Perceptron (1957):** Could solve OR, AND, NOT
- **Minsky & Papert (1969):** Showed perceptron limitations (XOR problem)
- **XOR crisis:** Led to reduced AI funding until multilayer networks

**Why XOR Cannot Be Solved:**
XOR truth table: [0,0]→0, [0,1]→1, [1,0]→1, [1,1]→0
- No single straight line can separate the two classes
- Requires non-linear decision boundary
- Solution: Multiple layers or non-linear activation

**TensorFlow Implementation:**
```python
# Single neuron with step activation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='linear', input_shape=(2,))
])
# Apply step function post-processing
outputs = tf.cast(predictions > 0, tf.float32)
```

This foundational understanding led to the development of multi-layer networks that can solve any logical function.

---

## Question 9: Gradient Descent vs. Momentum Comparison [5 Marks]

### **ANSWER:**
- **Standard GD update:** Δw = -0.04, w_new = 0.76
- **Momentum GD update:** Δw = -0.058, w_new = 0.742

### **Step-by-Step Solution:**

**Standard Gradient Descent:**
1. **Weight update:** Δw = -α × ∇w = -0.1 × 0.4 = **-0.04**
2. **New weight:** w_new = w + Δw = 0.8 + (-0.04) = **0.76**

**Momentum-based Gradient Descent:**
1. **Velocity update:** v = β × v_prev + (1-β) × ∇w = 0.9 × 0.2 + 0.1 × 0.4 = 0.18 + 0.04 = **0.22**
2. **Weight update:** Δw = -α × v = -0.1 × 0.22 = **-0.022**
3. **New weight:** w_new = w + Δw = 0.8 + (-0.022) = **0.778**

### **Detailed Explanation:**
This comparison illustrates **momentum optimization**, a key improvement over standard gradient descent from Week 6.

**Standard Gradient Descent Problems:**
1. **Oscillations:** In narrow valleys, updates zigzag between sides
2. **Slow convergence:** Small learning rates needed for stability
3. **Poor conditioning:** Struggles with different curvatures in different directions

**Momentum Solution:**
- **Physical analogy:** Ball rolling down hill gains momentum
- **Mathematical:** Exponentially weighted moving average of gradients
- **Effect:** Smooths out oscillations, accelerates in consistent directions

**Momentum Mathematics:**
- **Velocity equation:** v_t = βv_{t-1} + (1-β)∇w_t
- **Weight update:** w_{t+1} = w_t - αv_t
- **β (momentum coefficient):** Typically 0.9 (90% of previous velocity retained)

**Key Insights from Our Calculation:**

**1. Velocity Interpretation:**
- **Previous velocity (0.2):** Indicates previous movement direction
- **Current gradient (0.4):** Current optimization direction
- **Combined velocity (0.22):** Weighted average of both

**2. Update Magnitude:**
- **Standard GD:** |Δw| = 0.04
- **Momentum GD:** |Δw| = 0.022 (smaller this step)
- **Reason:** Gradient direction conflicts with previous momentum

**3. Momentum Benefits:**
- **Consistent directions:** Momentum accelerates movement
- **Conflicting directions:** Momentum dampens oscillations
- **Our case:** Dampening effect due to direction conflict

**Advanced Momentum Variants:**

**1. Classical Momentum (what we used):**
v_t = βv_{t-1} + (1-β)∇w_t

**2. Nesterov Momentum:**
- Look ahead: compute gradient at w_t - αβv_{t-1}
- More responsive to changes in gradient direction

**Hyperparameter Tuning:**
- **β = 0.9:** Standard choice, good for most problems
- **β = 0.99:** Higher momentum, better for noisy gradients
- **β = 0.5:** Lower momentum, more responsive to changes

**TensorFlow Implementation:**
```python
# Standard momentum
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

# Nesterov momentum
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
```

**Real-world Impact:**
- **Faster convergence:** Often 2-5x speedup over standard SGD
- **Better solutions:** Can escape shallow local minima
- **Industry standard:** Foundation for Adam, RMSprop optimizers

Momentum remains essential in modern deep learning, forming the basis for most advanced optimizers.

---

## Question 10: Complete Neural Network Forward Pass and Loss [5 Marks]

### **ANSWER:**
- **Final prediction:** ŷ = 0.7269
- **MSE Loss:** L = 0.0267
- **Output gradient:** ∂L/∂output = -0.1462

### **Step-by-Step Solution:**

**Forward Propagation:**

**Hidden Layer:**
1. **Linear transformation:**
   - h₁ = (1.0×0.3) + (0.5×0.2) + 0.1 = 0.3 + 0.1 + 0.1 = **0.5**
   - h₂ = (1.0×0.7) + (0.5×0.8) + 0.2 = 0.7 + 0.4 + 0.2 = **1.3**

2. **Sigmoid activation:**
   - a₁ = σ(0.5) = 1/(1 + e⁻⁰·⁵) = 1/(1 + 0.6065) = **0.6225**
   - a₂ = σ(1.3) = 1/(1 + e⁻¹·³) = 1/(1 + 0.2725) = **0.7858**

**Output Layer:**
3. **Linear transformation:**
   - z = (0.6225×0.6) + (0.7858×0.4) + 0.3 = 0.3735 + 0.3143 + 0.3 = **0.9878**

4. **Sigmoid activation:**
   - ŷ = σ(0.9878) = 1/(1 + e⁻⁰·⁹⁸⁷⁸) = **0.7269**

**Loss Calculation:**
5. **MSE Loss:** L = ½(y - ŷ)² = ½(0.8 - 0.7269)² = ½(0.0731)² = **0.0267**

**Gradient Calculation:**
6. **Output gradient:** ∂L/∂output = -(y - ŷ) = -(0.8 - 0.7269) = **-0.1462**

### **Detailed Explanation:**
This comprehensive problem demonstrates **end-to-end neural network computation**, integrating concepts from Weeks 3-6.

**Multi-Layer Architecture Understanding:**

**1. Network Topology:**
- **Input → Hidden:** 2×2 weight matrix (fully connected)
- **Hidden → Output:** 2×1 weight matrix
- **Total parameters:** 2×2 + 2 + 2×1 + 1 = 9 parameters
- **Depth:** 3 layers (input, hidden, output)

**2. Information Flow:**
- **Input [1.0, 0.5]:** Raw feature values
- **Hidden [0.6225, 0.7858]:** Learned feature representations
- **Output [0.7269]:** Final prediction/probability

**Activation Function Roles:**

**1. Hidden Layer Sigmoid:**
- **Squashing function:** Maps (-∞, ∞) to (0, 1)
- **Non-linearity:** Enables complex pattern learning
- **Smooth gradients:** Differentiable for backpropagation

**2. Output Layer Sigmoid:**
- **Probability interpretation:** Output ∈ (0, 1)
- **Binary classification:** Can be thresholded at 0.5
- **Loss compatibility:** Works well with binary cross-entropy

**Loss Function Analysis:**

**1. MSE Choice:**
- **Regression-like:** Treats output as continuous value
- **Quadratic penalty:** Larger errors penalized more
- **Alternative:** Binary cross-entropy for classification

**2. Gradient Interpretation:**
- **Negative gradient (-0.1462):** Loss decreases when output increases
- **Magnitude:** Indicates how strongly to adjust output
- **Direction:** Points toward better prediction

**Backpropagation Foundation:**
This gradient (∂L/∂output) is the starting point for backpropagation:
1. **Output layer:** ∂L/∂W₂ = (∂L/∂output) × (∂output/∂W₂)
2. **Hidden layer:** ∂L/∂W₁ = (∂L/∂output) × (∂output/∂hidden) × (∂hidden/∂W₁)
3. **Chain rule:** Connects output error to all parameters

**Computational Complexity:**
- **Forward pass:** O(n₁×n₂ + n₂×n₃) = O(2×2 + 2×1) = 6 multiplications
- **Memory:** Store all activations for backpropagation
- **Efficiency:** Much faster than traditional ML algorithms

**TensorFlow Equivalent:**
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation='sigmoid', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

loss_fn = tf.keras.losses.MeanSquaredError()
# Forward pass and gradient computation handled automatically
```

**Universal Approximation Theorem:**
This 3-layer network with 2 hidden neurons can approximate any continuous function on a compact set, demonstrating the power of neural networks.

The combination of linear transformations and non-linear activations creates a powerful function approximator that can learn complex patterns from data.

---

## Summary of Key Concepts Covered

1. **Perceptron Operations:** Linear combination + activation function
2. **Logic Gates:** Linear separability and decision boundaries
3. **Multi-layer Networks:** Universal approximation capabilities
4. **Gradient Descent:** Parameter optimization through calculus
5. **Activation Functions:** Non-linearity and their properties
6. **Regularization:** Dropout for preventing overfitting
7. **Batch Normalization:** Stabilizing layer inputs for faster training
8. **Momentum Optimization:** Accelerating convergence
9. **Forward Propagation:** Complete information flow through networks
10. **Loss Functions:** Measuring and optimizing model performance

These problems integrate theoretical understanding with practical TensorFlow/Keras implementation skills essential for deep learning practitioners.