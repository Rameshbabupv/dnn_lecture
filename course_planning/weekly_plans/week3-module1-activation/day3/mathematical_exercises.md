# Mathematical Exercises for Day 3 Interactive Elements
**Week 3: Advanced Neural Network Fundamentals**  
**Deep Neural Network Architectures (21CSE558T)**

---

**© 2025 Prof. Ramesh Babu | SRM University | Data Science and Business Systems (DSBS)**

---

## 🧮 ACTIVATION FUNCTION CALCULATIONS

### **Exercise Set 1: Sigmoid Function Practice**
**Mathematical Definition:** σ(x) = 1/(1 + e^(-x))

**Quick Calculations:**
1. Calculate σ(0) = ?
   - **Answer:** σ(0) = 1/(1 + e^0) = 1/(1 + 1) = **0.5**

2. Calculate σ(2) = ?
   - **Answer:** σ(2) = 1/(1 + e^(-2)) = 1/(1 + 0.135) = **0.881**

3. Calculate σ(-2) = ?
   - **Answer:** σ(-2) = 1/(1 + e^2) = 1/(1 + 7.389) = **0.119**

4. **Gradient:** σ'(x) = σ(x)(1 - σ(x))
   - Calculate σ'(0) = ?
   - **Answer:** σ'(0) = 0.5(1 - 0.5) = **0.25**

**Student Activity (2 minutes):** 
> *"Calculate sigmoid and its gradient for x = 1 and x = -1"*

---

### **Exercise Set 2: Tanh Function Practice**
**Mathematical Definition:** tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))

**Quick Calculations:**
1. Calculate tanh(0) = ?
   - **Answer:** tanh(0) = (1 - 1)/(1 + 1) = **0**

2. Calculate tanh(1) = ?
   - **Answer:** tanh(1) = (e - 1/e)/(e + 1/e) = (2.718 - 0.368)/(2.718 + 0.368) = **0.762**

3. **Gradient:** tanh'(x) = 1 - tanh²(x)
   - Calculate tanh'(0) = ?
   - **Answer:** tanh'(0) = 1 - 0² = **1**

**Comparison Question:**
> *"Why is tanh(0) = 0 while sigmoid(0) = 0.5? What's the significance?"*
> **Answer:** Tanh is zero-centered, making it better for hidden layers.

---

### **Exercise Set 3: ReLU and Variants**

**ReLU Calculations:**
1. ReLU(-3) = ?  **Answer:** 0
2. ReLU(0) = ?   **Answer:** 0
3. ReLU(5) = ?   **Answer:** 5
4. ReLU'(2) = ?  **Answer:** 1
5. ReLU'(-1) = ? **Answer:** 0

**Leaky ReLU (α = 0.01):**
1. LeakyReLU(-2) = ?  **Answer:** -0.02
2. LeakyReLU(3) = ?   **Answer:** 3
3. LeakyReLU'(-1) = ? **Answer:** 0.01

**Problem-Solving Question:**
> *"A ReLU neuron has been receiving negative inputs for 1000 training steps. What happens to its weights? How does Leaky ReLU solve this?"*

---

## 🔢 NEURAL LAYER MATHEMATICS

### **Exercise Set 4: Forward Pass Computation**

**Given:**
- Input: x = [2, -1, 3]
- Weights: W = [[0.1, 0.2], [0.3, -0.1], [0.2, 0.4]]
- Bias: b = [0.1, -0.2]

**Step-by-Step Calculation:**

1. **Matrix Multiplication:** z = x·W + b
   ```
   z₁ = (2×0.1) + (-1×0.3) + (3×0.2) + 0.1 = 0.2 - 0.3 + 0.6 + 0.1 = 0.6
   z₂ = (2×0.2) + (-1×-0.1) + (3×0.4) + (-0.2) = 0.4 + 0.1 + 1.2 - 0.2 = 1.5
   ```
   **Result:** z = [0.6, 1.5]

2. **Apply ReLU:**
   ```
   a₁ = ReLU(0.6) = 0.6
   a₂ = ReLU(1.5) = 1.5
   ```
   **Result:** a = [0.6, 1.5]

**Student Practice (3 minutes):**
> *"Apply Sigmoid activation instead of ReLU. What are the outputs?"*
> **Answer:** a = [0.646, 0.818]

---

### **Exercise Set 5: Gradient Computation**

**Scenario:** Simple 2-layer network
- Input: x = [1, -2]
- Target: y = 0.8
- Current output: ŷ = 0.3
- Loss: L = (y - ŷ)² = (0.8 - 0.3)² = 0.25

**Backpropagation Steps:**

1. **Output gradient:**
   ```
   ∂L/∂ŷ = 2(ŷ - y) = 2(0.3 - 0.8) = -1.0
   ```

2. **Activation gradient (Sigmoid at output):**
   ```
   ∂ŷ/∂z = ŷ(1 - ŷ) = 0.3(1 - 0.3) = 0.21
   ```

3. **Chain rule:**
   ```
   ∂L/∂z = ∂L/∂ŷ × ∂ŷ/∂z = -1.0 × 0.21 = -0.21
   ```

**Interactive Question:**
> *"If we used ReLU instead of Sigmoid, what would ∂ŷ/∂z be?"*
> **Answer:** 1 (assuming z > 0)

---

## 🎯 CONCEPTUAL PROBLEM SOLVING

### **Exercise Set 6: Activation Function Selection**

**Scenarios - Choose the best activation:**

1. **Binary Classification Output Layer:**
   - Options: ReLU, Sigmoid, Tanh, Softmax
   - **Answer:** Sigmoid (outputs 0-1 probability)

2. **Multi-class Classification (5 classes):**
   - Options: ReLU, Sigmoid, Tanh, Softmax
   - **Answer:** Softmax (probability distribution)

3. **Hidden Layer in Deep Network:**
   - Options: ReLU, Sigmoid, Tanh, Leaky ReLU
   - **Answer:** ReLU or Leaky ReLU (no vanishing gradients)

4. **Regression Output (can be negative):**
   - Options: ReLU, Sigmoid, Tanh, Linear
   - **Answer:** Linear/None (no bounds)

**Discussion Question (5 minutes):**
> *"You're building a 10-layer deep network for image recognition. Your training loss stops decreasing after epoch 3. What activation function issues might you suspect?"*

---

### **Exercise Set 7: Real-World Application**

**Case Study: MNIST Digit Classification**
- Input: 28×28 = 784 pixels
- Hidden: 128 neurons
- Output: 10 classes

**Architecture Design:**
```
Input(784) → Dense(128, activation=?) → Dense(10, activation=?)
```

**Questions:**
1. Best activation for hidden layer?
   - **Answer:** ReLU (fast, no vanishing gradients)

2. Best activation for output layer?
   - **Answer:** Softmax (10-class probability distribution)

3. Weight initialization for hidden layer?
   - **Answer:** He initialization (√(2/fan_in))

4. If 20% of ReLU neurons become inactive (dead), what's the solution?
   - **Answer:** Switch to Leaky ReLU (α=0.01)

---

## 🔬 HANDS-ON MINI EXPERIMENTS

### **Experiment 1: Gradient Magnitude Analysis**

**Setup:** Compare gradient magnitudes for inputs x ∈ [-5, 5]

| Activation | x = -2 | x = 0 | x = 2 | x = 5 |
|------------|--------|-------|-------|-------|
| Sigmoid    | 0.104  | 0.25  | 0.104 | 0.007 |
| Tanh       | 0.071  | 1.0   | 0.071 | 0.000 |
| ReLU       | 0      | ?     | 1     | 1     |
| Leaky ReLU | 0.01   | ?     | 1     | 1     |

**Student Task:** Fill in the missing values for ReLU at x = 0
**Answer:** Undefined (discontinuous), but typically treated as 0 or 1

---

### **Experiment 2: Activation Function Impact**

**Network Comparison:**
- Same architecture: 2 → 4 → 1
- Same data: XOR problem
- Different activations

**Predictions (after random initialization):**

| Input | Target | ReLU Net | Sigmoid Net | Tanh Net |
|-------|--------|----------|-------------|----------|
| [0,0] | 0      | 0.73     | 0.52        | 0.23     |
| [0,1] | 1      | 0.91     | 0.48        | 0.67     |
| [1,0] | 1      | 0.85     | 0.51        | 0.71     |
| [1,1] | 0      | 0.42     | 0.49        | 0.18     |

**Analysis Questions:**
1. Which network is closest to solving XOR initially?
2. Which activations show more diverse outputs?
3. What does this tell us about initialization sensitivity?

---

## 🎪 INTERACTIVE MOMENTS FOR LECTURE

### **Moment 1: The Vanishing Gradient Demo** (5 minutes)
**Setup:** Show sigmoid gradients through a deep network

```
Layer 1: gradient = 0.25
Layer 2: gradient = 0.25 × 0.25 = 0.0625
Layer 3: gradient = 0.0625 × 0.25 = 0.0156
Layer 10: gradient ≈ 0.000001 (vanished!)
```

**Audience Participation:** "What happens with ReLU?"
**Answer:** Gradient = 1 × 1 × 1 × ... = 1 (no vanishing!)

---

### **Moment 2: Activation Function Personality** (3 minutes)

**Character Assignment:**
- **Sigmoid:** "The Gatekeeper" (0 or 1 decisions)
- **Tanh:** "The Balanced One" (symmetric, zero-centered)
- **ReLU:** "The Minimalist" (simple but effective)
- **Leaky ReLU:** "The Survivor" (never truly dies)

**Student Question:** "Which personality fits your project best?"

---

### **Moment 3: Weight Initialization Game** (4 minutes)

**Show distributions:**
1. All zeros → "Symmetry breaking problem"
2. Too large (σ=10) → "Saturation nightmare"
3. Too small (σ=0.001) → "Signal vanishing"
4. Just right (He/Xavier) → "Goldilocks zone"

**Poll:** "Which would you choose and why?"

---

## 📊 ASSESSMENT PREPARATION (FT-I Quiz)

### **Sample Quiz Questions:**

1. **Multiple Choice:** The sigmoid activation function outputs values in the range:
   - a) (-∞, ∞)  b) (-1, 1)  c) (0, 1)  d) [0, 1]
   - **Answer:** c) (0, 1)

2. **Short Answer:** Why is ReLU preferred over sigmoid in hidden layers of deep networks?
   - **Answer:** ReLU prevents vanishing gradients and is computationally efficient.

3. **Calculation:** If tanh(-1) ≈ -0.76, what is tanh'(-1)?
   - **Answer:** 1 - (-0.76)² = 1 - 0.58 = 0.42

4. **Conceptual:** Explain the "dying ReLU" problem and its solution.
   - **Answer:** Neurons output zero for all negative inputs, stopping learning. Leaky ReLU provides small gradient for negative inputs.

---

## 🎯 HOMEWORK PREPARATION FOR DAY 4

**Pre-Tutorial Checklist:**
- [ ] Review activation function derivatives
- [ ] Install Python environment: TensorFlow, NumPy, Matplotlib
- [ ] Download T3 exercise files
- [ ] Practice matrix multiplication (3×4 matrices)
- [ ] Understand layer output dimensions

**Mathematical Readiness Test:**
1. Can you calculate sigmoid(0.5) by hand?
2. Do you know why tanh is better than sigmoid for hidden layers?
3. Can you explain matrix multiplication for neural layers?

**If any answer is "no" → Review today's materials!**

---

## 🚀 EXTENSION CHALLENGES

### **For Advanced Students:**

1. **Research Question:** How does the Swish activation function compare to ReLU?
2. **Implementation Challenge:** Write pseudocode for backpropagation through a ReLU layer.
3. **Theoretical Analysis:** Prove why sigmoid gradients are maximized at x = 0.

### **For Struggling Students:**

1. **Visual Aid:** Draw the ReLU function graph by hand.
2. **Pattern Recognition:** Identify which activation functions are bounded.
3. **Practical Connection:** Name one real application for each activation type.

---

**Ready for tomorrow's hands-on implementation! 🔥**