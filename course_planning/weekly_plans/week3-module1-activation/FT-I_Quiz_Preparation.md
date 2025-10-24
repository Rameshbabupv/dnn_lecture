# FT-I Quiz Preparation Materials
**Week 3 Assessment: Module 1 Foundations**  
**Deep Neural Network Architectures (21CSE558T)**  
**Date: End of Week 3 (Aug 29, 2025) | Duration: 50 minutes**

---

**© 2025 Prof. Ramesh Babu | SRM University | Data Science and Business Systems (DSBS)**

---

## 📋 QUIZ OVERVIEW

### **Assessment Details:**
- **Type:** Formative Assessment I (Quiz)
- **Weightage:** 5 marks (8.33% of total)
- **Format:** Mixed (MCQ + Short Answer + Calculation)
- **Coverage:** Module 1 (Weeks 1-3)
- **Duration:** 50 minutes
- **Date:** Friday, August 29, 2025

### **Learning Outcomes Tested:**
- **CO-1:** Create simple deep neural networks and explain their functions
- **CO-2:** Build neural networks with multiple layers (introductory level)

---

## 🎯 SYLLABUS COVERAGE

### **Week 1: Deep Learning Fundamentals** (25% of quiz)
- History and applications of deep learning
- Biological vs artificial neurons
- Perceptron model and limitations
- Linear separability concepts

### **Week 2: Multilayer Networks** (35% of quiz)
- Multilayer Perceptron (MLP) architecture
- XOR problem and solution
- TensorFlow basics and data structures
- Basic tensor operations

### **Week 3: Activation Functions & Mathematics** (40% of quiz)
- Activation functions (Sigmoid, Tanh, ReLU, Leaky ReLU)
- Neural layer mathematics
- Forward pass computation
- Basic backpropagation concepts

---

## 📚 KEY CONCEPTS TO MASTER

### **1. Fundamental Concepts**

**Deep Learning Definitions:**
- Deep learning vs machine learning
- Neural network vs traditional algorithms
- Supervised vs unsupervised learning in context

**Historical Knowledge:**
- Perceptron (1957) - Frank Rosenblatt
- Multilayer Perceptrons - solving XOR
- Deep learning resurgence (2006+)

**Applications Awareness:**
- Image recognition, NLP, speech recognition
- Computer vision, autonomous vehicles
- Medical diagnosis, drug discovery

### **2. Biological Foundations**

**Neuron Components:**
- Cell body (soma), dendrites, axon, synapse
- All-or-nothing principle
- Synaptic weights and learning

**Artificial Neuron Mapping:**
- Input weights ← Synaptic strengths
- Bias ← Threshold potential
- Activation function ← Firing behavior
- Output ← Axon signal

### **3. Perceptron Mathematics**

**Mathematical Model:**
```
output = activation(Σ(wᵢxᵢ) + bias)
```

**Key Limitations:**
- Can only learn linearly separable functions
- Cannot solve XOR problem alone
- Single layer restriction

**Linear Separability:**
- AND gate: linearly separable ✓
- OR gate: linearly separable ✓  
- XOR gate: NOT linearly separable ✗

### **4. Multilayer Perceptron (MLP)**

**Architecture Components:**
- Input layer (feature representation)
- Hidden layers (feature transformation)
- Output layer (decision/prediction)

**Universal Function Approximation:**
- MLPs can approximate any continuous function
- Hidden layers enable non-linear mappings
- Depth vs width trade-offs

**XOR Solution:**
```
Input → Hidden(2 neurons) → Output(1 neuron)
Hidden layer creates non-linear feature space
```

### **5. TensorFlow Fundamentals**

**Core Data Structures:**
```python
# Constants
tf.constant([1, 2, 3])

# Variables (learnable parameters)
tf.Variable(tf.random.normal([3, 4]))

# Placeholders (deprecated, use tf.function)
```

**Basic Operations:**
- Element-wise: +, -, *, /
- Matrix operations: tf.matmul()
- Reductions: tf.reduce_mean(), tf.reduce_sum()
- Shape manipulation: tf.reshape(), tf.transpose()

### **6. Activation Functions Mastery**

**Function Definitions & Properties:**

| Function | Formula | Range | Derivative | Use Case |
|----------|---------|-------|------------|----------|
| Sigmoid | 1/(1+e^(-x)) | (0,1) | σ(x)(1-σ(x)) | Binary output |
| Tanh | (e^x-e^(-x))/(e^x+e^(-x)) | (-1,1) | 1-tanh²(x) | Hidden layers |
| ReLU | max(0,x) | [0,∞) | 1 if x>0, else 0 | Hidden layers |
| Leaky ReLU | max(αx,x) | (-∞,∞) | 1 if x>0, else α | Avoid dead neurons |

**Critical Insights:**
- Sigmoid/Tanh: Vanishing gradient problem
- ReLU: Dead neuron problem
- Leaky ReLU: Solves dead neuron issue
- Choice affects training dynamics

### **7. Neural Layer Mathematics**

**Forward Pass:**
```
Layer output = activation(input·weights + bias)
a^(l) = σ(W^(l)·a^(l-1) + b^(l))
```

**Matrix Dimensions:**
- Input: (batch_size, input_features)
- Weights: (input_features, output_features)  
- Bias: (output_features,)
- Output: (batch_size, output_features)

**Weight Initialization:**
- Xavier/Glorot: σ = √(2/(fan_in + fan_out))
- He: σ = √(2/fan_in) → Better for ReLU
- Zero initialization: Symmetry breaking problem

### **8. Backpropagation Basics**

**Chain Rule Application:**
```
∂Loss/∂Weight = ∂Loss/∂Output × ∂Output/∂Activation × ∂Activation/∂Weight
```

**Gradient Flow:**
- Forward pass: Compute predictions
- Backward pass: Compute gradients
- Update step: Adjust parameters

**Common Problems:**
- Vanishing gradients: Sigmoid/Tanh in deep networks
- Exploding gradients: Poor initialization
- Dead gradients: ReLU neurons stuck at zero

---

## 📝 SAMPLE QUIZ QUESTIONS

### **Section A: Multiple Choice Questions (20 marks)**

1. **The perceptron can solve which of the following problems?**
   a) XOR gate  b) AND gate  c) Non-linearly separable  d) All of the above
   **Answer: b**

2. **Which activation function has zero-centered outputs?**
   a) Sigmoid  b) ReLU  c) Tanh  d) Leaky ReLU
   **Answer: c**

3. **In TensorFlow, which creates a learnable parameter?**
   a) tf.constant()  b) tf.Variable()  c) tf.placeholder()  d) tf.tensor()
   **Answer: b**

4. **The vanishing gradient problem is most severe with:**
   a) ReLU  b) Leaky ReLU  c) Sigmoid  d) Linear activation
   **Answer: c**

5. **For multi-class classification output, the best activation is:**
   a) Sigmoid  b) Tanh  c) ReLU  d) Softmax
   **Answer: d**

6. **He initialization is specifically designed for:**
   a) Sigmoid networks  b) ReLU networks  c) Linear networks  d) All networks
   **Answer: b**

7. **Which operation represents a neural layer mathematically?**
   a) y = Wx  b) y = W + x  c) y = σ(Wx + b)  d) y = σ(W) + x
   **Answer: c**

8. **The biological inspiration for neural network bias comes from:**
   a) Dendrites  b) Axon  c) Threshold potential  d) Synapse
   **Answer: c**

9. **XOR problem requires minimum how many layers to solve?**
   a) 1  b) 2  c) 3  d) 4
   **Answer: c (including output layer)**

10. **Leaky ReLU solves which problem of standard ReLU?**
    a) Computational cost  b) Dying neurons  c) Saturation  d) Non-linearity
    **Answer: b**

### **Section B: Short Answer Questions (15 marks)**

1. **Explain why a single perceptron cannot solve the XOR problem. (3 marks)**
   - XOR is not linearly separable
   - Single perceptron can only create linear decision boundaries
   - Need multiple layers to create non-linear boundaries

2. **List three advantages of ReLU over Sigmoid activation. (3 marks)**
   - No vanishing gradient problem
   - Computationally efficient (simple max operation)
   - Sparse activation (many neurons output zero)

3. **What is the role of bias in a neural network layer? (3 marks)**
   - Shifts the activation function
   - Allows fitting data that doesn't pass through origin
   - Provides additional degree of freedom for learning

4. **Define the universal function approximation theorem. (3 marks)**
   - MLPs with sufficient hidden neurons can approximate any continuous function
   - Requires at least one hidden layer with non-linear activation
   - Theoretical guarantee, not practical guide

5. **Why is proper weight initialization important? (3 marks)**
   - Prevents vanishing/exploding gradients
   - Ensures effective learning from start
   - Breaks symmetry between neurons

### **Section C: Calculation Problems (15 marks)**

1. **Calculate the following activation function values: (5 marks)**
   ```
   a) sigmoid(0) = ?
   b) tanh(1) ≈ ?
   c) ReLU(-3) = ?
   d) LeakyReLU(-2, α=0.01) = ?
   e) sigmoid'(0) = ?
   ```
   **Answers:** a) 0.5, b) 0.76, c) 0, d) -0.02, e) 0.25

2. **Forward Pass Calculation (5 marks)**
   ```
   Given: Input x = [1, -1, 2]
          Weights W = [[0.5, -0.2], [0.3, 0.4], [-0.1, 0.6]]
          Bias b = [0.1, -0.3]
          
   Calculate the layer output before and after ReLU activation.
   ```
   **Solution:**
   ```
   z = x·W + b
   z₁ = (1×0.5) + (-1×0.3) + (2×-0.1) + 0.1 = 0.5 - 0.3 - 0.2 + 0.1 = 0.1
   z₂ = (1×-0.2) + (-1×0.4) + (2×0.6) + (-0.3) = -0.2 - 0.4 + 1.2 - 0.3 = 0.3
   
   Before ReLU: z = [0.1, 0.3]
   After ReLU: a = [0.1, 0.3] (both positive, unchanged)
   ```

3. **Gradient Calculation (5 marks)**
   ```
   For sigmoid activation σ(x) = 1/(1+e^(-x)):
   If x = 2 and σ(2) ≈ 0.88, calculate:
   a) The gradient σ'(2)
   b) If this were a layer in backprop with upstream gradient = -0.5, 
      what's the gradient flowing to this layer's input?
   ```
   **Solution:**
   ```
   a) σ'(2) = σ(2)(1-σ(2)) = 0.88(1-0.88) = 0.88×0.12 = 0.106
   b) Gradient to input = upstream_grad × local_grad = -0.5 × 0.106 = -0.053
   ```

---

## 🎯 STUDY STRATEGY

### **Week Before Quiz:**
- [ ] Review all lecture notes (Weeks 1-3)
- [ ] Practice activation function calculations
- [ ] Understand TensorFlow basic operations
- [ ] Solve XOR problem conceptually
- [ ] Review biological neuron concepts

### **Day Before Quiz:**
- [ ] Quick formula review (activation functions)
- [ ] Practice matrix multiplication
- [ ] Understand forward pass computation
- [ ] Review gradient computation basics
- [ ] Get good sleep!

### **Day of Quiz:**
- [ ] Arrive 10 minutes early
- [ ] Bring calculator
- [ ] Read questions carefully
- [ ] Start with easier questions
- [ ] Show all calculation steps

---

## 🔧 COMMON MISTAKES TO AVOID

### **Mathematical Errors:**
- Confusing sigmoid range (0,1) with tanh range (-1,1)
- Forgetting bias term in layer computation
- Wrong matrix dimension ordering
- Sign errors in calculations

### **Conceptual Errors:**
- Saying perceptron can solve XOR
- Confusing biological and artificial neuron components
- Wrong activation function recommendations
- Misunderstanding universal approximation theorem

### **TensorFlow Errors:**
- Confusing constants and variables
- Wrong function names (tf.multiply vs tf.matmul)
- Incorrect tensor dimension understanding

---

## 📊 GRADING RUBRIC

### **Grading Distribution:**
- **Accuracy:** 60% (correct answers)
- **Method:** 25% (showing work)
- **Understanding:** 15% (clear explanations)

### **Excellence Indicators:**
- All calculations shown step-by-step
- Clear explanations of concepts
- Correct use of mathematical notation
- Proper TensorFlow syntax knowledge
- Understanding of practical applications

### **Improvement Areas:**
- Memorize activation function formulas
- Practice matrix operations
- Understand gradient flow concepts
- Know when to use which activation

---

## 🚀 POST-QUIZ PREPARATION

### **Feedback Integration:**
- Review incorrect answers
- Understand solution methods
- Identify weak concept areas
- Plan Module 2 preparation

### **Module 2 Preview:**
- Optimization algorithms (next week)
- Gradient descent variants
- Regularization techniques
- Advanced training strategies

---

**Good luck with FT-I! You've got this! 💪**