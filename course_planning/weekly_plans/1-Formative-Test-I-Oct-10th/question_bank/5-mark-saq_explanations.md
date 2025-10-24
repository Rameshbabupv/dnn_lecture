# 5-Mark SAQ Questions - Detailed Solutions and Explanations
**Course Code**: 21CSE558T | **Course Title**: Deep Neural Network Architectures
**Test**: Formative Assessment I | **Coverage**: Module 1 & Module 2
**Purpose**: Complete step-by-step solutions for all 5-mark computational questions

---

## 📚 Purpose of This Document

This document provides **detailed step-by-step solutions** for all 10 five-mark computational questions. Each solution includes:
- **Complete mathematical working** with all intermediate steps
- **Final numerical answers** with proper units
- **Conceptual explanations** linking to course concepts
- **Common mistake warnings** to avoid errors

---

# DETAILED SOLUTIONS

## Module 1: Introduction to Deep Learning (4 questions)

### Q1. Perceptron Forward Pass Calculation

**Given:**
- Input vector: X = [0.8, 0.6]
- Weight vector: W = [0.4, -0.7]
- Bias: b = 0.3
- Activation function: Sigmoid

**Solution:**

**Step 1: Calculate weighted sum (z)**
z = w₁x₁ + w₂x₂ + b
z = (0.4)(0.8) + (-0.7)(0.6) + 0.3
z = 0.32 - 0.42 + 0.3
z = 0.2

**Step 2: Apply sigmoid activation function**
σ(z) = 1/(1 + e^(-z))
σ(0.2) = 1/(1 + e^(-0.2))
σ(0.2) = 1/(1 + 0.8187)
σ(0.2) = 1/1.8187
σ(0.2) = 0.5498

**Final Answer:** The perceptron output is **0.5498**

---

### Q2. AND Gate Perceptron Implementation

**Given:**
- Input combinations: [0,0], [0,1], [1,0], [1,1]
- Weights: W = [0.5, 0.5]
- Bias: b = -0.7
- Activation: Step function

**Solution:**

**Case 1: Input [0,0]**
z = (0.5)(0) + (0.5)(0) + (-0.7) = -0.7
Since z = -0.7 < 0, output = 0 ✓ (AND truth table: 0)

**Case 2: Input [0,1]**
z = (0.5)(0) + (0.5)(1) + (-0.7) = 0.5 - 0.7 = -0.2
Since z = -0.2 < 0, output = 0 ✓ (AND truth table: 0)

**Case 3: Input [1,0]**
z = (0.5)(1) + (0.5)(0) + (-0.7) = 0.5 - 0.7 = -0.2
Since z = -0.2 < 0, output = 0 ✓ (AND truth table: 0)

**Case 4: Input [1,1]**
z = (0.5)(1) + (0.5)(1) + (-0.7) = 0.5 + 0.5 - 0.7 = 0.3
Since z = 0.3 > 0, output = 1 ✓ (AND truth table: 1)

**Final Answer:** The perceptron correctly implements the AND gate for all input combinations.

---

### Q3. MLP Forward Propagation

**Given:**
- Input: X = [1.2, -0.8]
- W1 = [[0.5, -0.3, 0.7], [0.2, 0.9, -0.4]], b1 = [0.1, -0.2, 0.3]
- W2 = [[0.6], [-0.5], [0.8]], b2 = [0.2]

**Solution:**

**Step 1: Calculate hidden layer pre-activation**
z1 = W1 · X + b1

For neuron 1: z1₁ = (0.5)(1.2) + (-0.3)(-0.8) + 0.1 = 0.6 + 0.24 + 0.1 = 0.94
For neuron 2: z1₂ = (0.2)(1.2) + (0.9)(-0.8) + (-0.2) = 0.24 - 0.72 - 0.2 = -0.68
For neuron 3: z1₃ = (0.7)(1.2) + (-0.4)(-0.8) + 0.3 = 0.84 + 0.32 + 0.3 = 1.46

**Step 2: Apply ReLU activation to hidden layer**
h1₁ = ReLU(0.94) = max(0, 0.94) = 0.94
h1₂ = ReLU(-0.68) = max(0, -0.68) = 0.0
h1₃ = ReLU(1.46) = max(0, 1.46) = 1.46

Hidden layer output: h1 = [0.94, 0.0, 1.46]

**Step 3: Calculate output layer pre-activation**
z2 = W2 · h1 + b2
z2 = (0.6)(0.94) + (-0.5)(0.0) + (0.8)(1.46) + 0.2
z2 = 0.564 + 0 + 1.168 + 0.2 = 1.932

**Step 4: Apply sigmoid activation to output**
y = σ(1.932) = 1/(1 + e^(-1.932)) = 1/(1 + 0.1448) = 1/1.1448 = 0.8735

**Final Answer:** The final network output is **0.8735**

---

### Q4. Gradient Descent Weight Update

**Given:**
- α = 0.01, w = 0.8, b = 0.3, x = 2.0, y = 1.0, ŷ = 0.7
- Loss: MSE = ½(y - ŷ)²

**Solution:**

**Step 1: Calculate the loss**
L = ½(y - ŷ)² = ½(1.0 - 0.7)² = ½(0.3)² = ½(0.09) = 0.045

**Step 2: Calculate gradient with respect to weight (∂L/∂w)**
∂L/∂ŷ = -(y - ŷ) = -(1.0 - 0.7) = -0.3
∂ŷ/∂w = x = 2.0 (assuming linear model: ŷ = wx + b)
∂L/∂w = ∂L/∂ŷ × ∂ŷ/∂w = (-0.3)(2.0) = -0.6

**Step 3: Calculate gradient with respect to bias (∂L/∂b)**
∂ŷ/∂b = 1
∂L/∂b = ∂L/∂ŷ × ∂ŷ/∂b = (-0.3)(1) = -0.3

**Step 4: Update weight and bias**
w_new = w - α × ∂L/∂w = 0.8 - (0.01)(-0.6) = 0.8 + 0.006 = 0.806
b_new = b - α × ∂L/∂b = 0.3 - (0.01)(-0.3) = 0.3 + 0.003 = 0.303

**Final Answer:**
- Updated weight: **w = 0.806**
- Updated bias: **b = 0.303**

---

## Module 2: Optimization & Regularization (6 questions)

### Q5. Activation Functions Comparison

**Given:** Input values: [-2.0, 0.0, 2.0]

**Solution:**

**Sigmoid: σ(x) = 1/(1 + e^(-x))**
- σ(-2.0) = 1/(1 + e^(2.0)) = 1/(1 + 7.389) = 1/8.389 = 0.1192
- σ(0.0) = 1/(1 + e^(0)) = 1/(1 + 1) = 1/2 = 0.5000
- σ(2.0) = 1/(1 + e^(-2.0)) = 1/(1 + 0.1353) = 1/1.1353 = 0.8808

**ReLU: f(x) = max(0, x)**
- ReLU(-2.0) = max(0, -2.0) = 0.0000
- ReLU(0.0) = max(0, 0.0) = 0.0000
- ReLU(2.0) = max(0, 2.0) = 2.0000

**Tanh: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))**
- tanh(-2.0) = (0.1353 - 7.389)/(0.1353 + 7.389) = -7.254/7.524 = -0.9640
- tanh(0.0) = (1 - 1)/(1 + 1) = 0/2 = 0.0000
- tanh(2.0) = (7.389 - 0.1353)/(7.389 + 0.1353) = 7.254/7.524 = 0.9640

**Final Answer:**
| Input | Sigmoid | ReLU | Tanh |
|-------|---------|------|------|
| -2.0  | 0.1192  | 0.0000 | -0.9640 |
| 0.0   | 0.5000  | 0.0000 | 0.0000  |
| 2.0   | 0.8808  | 2.0000 | 0.9640  |

---

### Q6. Dropout Regularization Application

**Given:**
- Input activations: [0.8, 0.6, 0.9, 0.7, 0.5]
- Dropout rate: 0.3, Mask: [1, 0, 1, 1, 0]
- Scaling factor: 1/(1-0.3) = 1/0.7 ≈ 1.4286

**Solution:**

**Step 1: Apply dropout mask**
Masked activations = input × mask
- Neuron 1: 0.8 × 1 = 0.8
- Neuron 2: 0.6 × 0 = 0.0 (dropped)
- Neuron 3: 0.9 × 1 = 0.9
- Neuron 4: 0.7 × 1 = 0.7
- Neuron 5: 0.5 × 0 = 0.0 (dropped)

**Step 2: Apply scaling factor**
Final output = masked activations × scaling factor
- Neuron 1: 0.8 × 1.4286 = 1.1429
- Neuron 2: 0.0 × 1.4286 = 0.0000
- Neuron 3: 0.9 × 1.4286 = 1.2857
- Neuron 4: 0.7 × 1.4286 = 1.0000
- Neuron 5: 0.0 × 1.4286 = 0.0000

**Final Answer:** [1.1429, 0.0000, 1.2857, 1.0000, 0.0000]

---

### Q7. Batch Normalization Calculation

**Given:**
- Input batch: [2.0, 4.0, 6.0, 8.0]
- γ = 1.5, β = 0.5, ε = 1e-5

**Solution:**

**Step 1: Calculate batch mean**
μ = (2.0 + 4.0 + 6.0 + 8.0)/4 = 20.0/4 = 5.0

**Step 2: Calculate batch variance**
σ² = [(2.0-5.0)² + (4.0-5.0)² + (6.0-5.0)² + (8.0-5.0)²]/4
σ² = [(-3.0)² + (-1.0)² + (1.0)² + (3.0)²]/4
σ² = [9.0 + 1.0 + 1.0 + 9.0]/4 = 20.0/4 = 5.0

**Step 3: Normalize inputs**
x̂ᵢ = (xᵢ - μ)/√(σ² + ε)
√(5.0 + 0.00001) = √5.00001 ≈ 2.236

- x̂₁ = (2.0 - 5.0)/2.236 = -3.0/2.236 = -1.342
- x̂₂ = (4.0 - 5.0)/2.236 = -1.0/2.236 = -0.447
- x̂₃ = (6.0 - 5.0)/2.236 = 1.0/2.236 = 0.447
- x̂₄ = (8.0 - 5.0)/2.236 = 3.0/2.236 = 1.342

**Step 4: Scale and shift**
yᵢ = γx̂ᵢ + β = 1.5x̂ᵢ + 0.5

- y₁ = 1.5(-1.342) + 0.5 = -2.013 + 0.5 = -1.513
- y₂ = 1.5(-0.447) + 0.5 = -0.671 + 0.5 = -0.171
- y₃ = 1.5(0.447) + 0.5 = 0.671 + 0.5 = 1.171
- y₄ = 1.5(1.342) + 0.5 = 2.013 + 0.5 = 2.513

**Final Answer:** [-1.513, -0.171, 1.171, 2.513]

---

### Q8. OR Gate Perceptron Design

**Given:**
- Truth table: [0,0]→0, [0,1]→1, [1,0]→1, [1,1]→1
- W = [0.6, 0.6], b = -0.2

**Solution:**

**Case 1: Input [0,0]**
z = (0.6)(0) + (0.6)(0) + (-0.2) = -0.2
Since z = -0.2 < 0, output = 0 ✓ (OR truth table: 0)

**Case 2: Input [0,1]**
z = (0.6)(0) + (0.6)(1) + (-0.2) = 0.6 - 0.2 = 0.4
Since z = 0.4 > 0, output = 1 ✓ (OR truth table: 1)

**Case 3: Input [1,0]**
z = (0.6)(1) + (0.6)(0) + (-0.2) = 0.6 - 0.2 = 0.4
Since z = 0.4 > 0, output = 1 ✓ (OR truth table: 1)

**Case 4: Input [1,1]**
z = (0.6)(1) + (0.6)(1) + (-0.2) = 0.6 + 0.6 - 0.2 = 1.0
Since z = 1.0 > 0, output = 1 ✓ (OR truth table: 1)

**Final Answer:** The perceptron correctly implements the OR gate for all input combinations.

---

### Q9. Gradient Descent vs Momentum Comparison

**Given:**
- ∇w = 0.4, α = 0.1, w = 0.8, v_prev = 0.2, β = 0.9

**Solution:**

**Standard Gradient Descent:**
w_new = w - α × ∇w
w_new = 0.8 - (0.1)(0.4) = 0.8 - 0.04 = 0.76

**Momentum-based Gradient Descent:**
**Step 1: Update velocity**
v = β × v_prev + α × ∇w
v = (0.9)(0.2) + (0.1)(0.4) = 0.18 + 0.04 = 0.22

**Step 2: Update weight**
w_new = w - v = 0.8 - 0.22 = 0.58

**Comparison:**
- Standard GD update: Δw = -0.04
- Momentum GD update: Δw = -0.22
- Momentum provides **5.5× larger weight update** due to accumulated velocity

**Final Answer:**
- Standard GD: **w = 0.76**
- Momentum GD: **w = 0.58** (larger update magnitude)

---

### Q10. 3-Layer Network Forward Pass and Loss

**Given:**
- Input: X = [1.0, 0.5], Target: y = 0.8
- W1 = [[0.3, 0.7], [0.2, 0.8]], b1 = [0.1, 0.2]
- W2 = [[0.6], [0.4]], b2 = [0.3]

**Solution:**

**Step 1: Hidden layer forward pass**
z1₁ = (0.3)(1.0) + (0.7)(0.5) + 0.1 = 0.3 + 0.35 + 0.1 = 0.75
z1₂ = (0.2)(1.0) + (0.8)(0.5) + 0.2 = 0.2 + 0.4 + 0.2 = 0.8

h1₁ = σ(0.75) = 1/(1 + e^(-0.75)) = 1/(1 + 0.472) = 0.679
h1₂ = σ(0.8) = 1/(1 + e^(-0.8)) = 1/(1 + 0.449) = 0.690

**Step 2: Output layer forward pass**
z2 = (0.6)(0.679) + (0.4)(0.690) + 0.3 = 0.407 + 0.276 + 0.3 = 0.983
ŷ = σ(0.983) = 1/(1 + e^(-0.983)) = 1/(1 + 0.374) = 0.728

**Step 3: Calculate MSE loss**
L = ½(y - ŷ)² = ½(0.8 - 0.728)² = ½(0.072)² = ½(0.005184) = 0.002592

**Step 4: Calculate output gradient**
∂L/∂ŷ = -(y - ŷ) = -(0.8 - 0.728) = -0.072

**Final Answer:**
- Network prediction: **ŷ = 0.728**
- MSE loss: **L = 0.002592**
- Output gradient: **∂L/∂ŷ = -0.072**

---

## Summary Notes

### Key Solution Patterns:
1. **Always show intermediate steps** for full marks
2. **Include proper mathematical notation** (∂, σ, etc.)
3. **Verify results** against expected behavior
4. **Round final answers** to 4 decimal places
5. **Include units** where applicable

### Common Mistakes to Avoid:
- Forgetting bias terms in calculations
- Incorrect activation function applications
- Missing scaling factors in dropout/normalization
- Not showing gradient chain rule steps
- Arithmetic errors in matrix multiplications

**📧 Note**: All solutions follow the mathematical foundations covered in Weeks 1-6 lectures.