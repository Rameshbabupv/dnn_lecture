# Day 3 Lecture Agenda - 2 Hours
**Week 3: Advanced Neural Network Fundamentals**  
**Date:** Aug 25, 2025 | **Duration:** 2 Hours | **Format:** Theory-Intensive Lecture

---

**© 2025 Prof. Ramesh Babu | SRM University | Data Science and Business Systems (DSBS)**  
*Course Materials for 21CSE558T - Deep Neural Network Architectures*

---

## Session Overview
**Primary Focus:** Activation Functions Deep Dive & Mathematical Foundations  
**Learning Style:** Theory → Mathematical Analysis → Conceptual Understanding  
**Preparation for:** Day 4 practical implementation

---

## Detailed Timeline

### **Hour 1: Activation Functions Mastery** (60 minutes)

#### **Opening & Context Setting** (10 minutes)
- Week 3 objectives and roadmap
- Connection to previous weeks (perceptron → MLP → activation functions)
- Why activation functions are critical for neural network expressiveness

#### **Classical Activation Functions** (25 minutes)
- **Sigmoid Function**
  - Mathematical definition: σ(x) = 1/(1 + e^(-x))
  - Properties: range (0,1), smooth, differentiable
  - Gradient: σ'(x) = σ(x)(1 - σ(x))
  - Problems: vanishing gradients, not zero-centered
  
- **Hyperbolic Tangent (Tanh)**
  - Mathematical definition: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
  - Properties: range (-1,1), zero-centered
  - Gradient: tanh'(x) = 1 - tanh²(x)
  - Advantages over sigmoid

#### **Modern Activation Functions** (25 minutes)
- **Rectified Linear Unit (ReLU)**
  - Definition: f(x) = max(0, x)
  - Properties: simple, computationally efficient
  - Gradient: f'(x) = 1 if x > 0, else 0
  - Dead neuron problem
  
- **Leaky ReLU & Variants**
  - Leaky ReLU: f(x) = max(αx, x) where α = 0.01
  - Parametric ReLU (PReLU): learnable α
  - Exponential Linear Unit (ELU): smooth for x < 0

**Break** (5 minutes)

---

### **Hour 2: Mathematical Foundations & Layer Architecture** (60 minutes)

#### **Neural Network Layer Mathematics** (30 minutes)
- **Dense/Fully Connected Layer Structure**
  - Mathematical representation: y = Wx + b
  - Weight matrix W dimensions and initialization
  - Bias vector b role and initialization strategies
  
- **Forward Pass Computation**
  - Layer-by-layer matrix operations
  - Activation function application
  - Output computation: a^(l) = f(W^(l)a^(l-1) + b^(l))

- **Parameter Initialization Strategies**
  - Xavier/Glorot initialization
  - He initialization for ReLU networks
  - Why proper initialization matters

#### **Backpropagation Mathematical Framework** (25 minutes)
- **Chain Rule Application**
  - Partial derivatives through composition
  - Error propagation through layers
  - Gradient computation: ∂L/∂W and ∂L/∂b

- **Loss Function Derivatives**
  - Mean Squared Error: ∂MSE/∂y
  - Cross-entropy: ∂CE/∂y  
  - How choice of activation affects gradients

#### **Putting It All Together** (5 minutes)
- Complete forward and backward pass overview
- How activation functions impact learning
- Preview of Day 4 implementation

---

## Learning Checkpoints

### **After Hour 1 - Students Should Understand:**
✅ Mathematical properties of each activation function  
✅ When to choose specific activation functions  
✅ Gradient computation for each function  
✅ Common problems (vanishing gradients, dead neurons)

### **After Hour 2 - Students Should Understand:**
✅ Matrix operations in neural network layers  
✅ Forward pass mathematical formulation  
✅ Backpropagation mathematical framework  
✅ Parameter initialization importance

---

## Interactive Elements

### **Mathematical Exercises** (Throughout)
- Calculate sigmoid gradient at x = 0, 2, -2
- Compare tanh vs sigmoid outputs for same inputs
- Determine ReLU gradient for positive/negative inputs
- Compute layer output for given weights and inputs

### **Conceptual Questions**
- Why is ReLU preferred over sigmoid in deep networks?
- When would you choose Leaky ReLU over standard ReLU?
- How does activation function choice affect gradient flow?

---

## Visual Aids & Demonstrations

1. **Activation Function Plots** - Visual comparison of function shapes
2. **Gradient Behavior Graphs** - Show vanishing gradient problem
3. **Layer Architecture Diagrams** - Matrix dimension visualization  
4. **Forward Pass Animation** - Step-by-step computation flow

---

## Assessment Integration

**Unit Test 1 Preparation Topics:**
- Activation function properties and derivatives
- Forward pass mathematical computation
- Layer parameter initialization concepts
- Basic backpropagation understanding

---

## Homework/Preparation for Day 4

1. **Review** activation function derivatives
2. **Install** required TensorFlow/NumPy environment
3. **Read** Tutorial T3 description
4. **Prepare** for hands-on tensor manipulation

---

---

## CO-PO Integration & Assessment

### **Course Outcomes Achievement**
- **CO-1** (Foundation Building): Understanding neural network mathematical components
- **CO-2** (Introduction Level): Learning activation function selection for multi-layer networks

### **Programme Outcomes Alignment**
- **PO-1**: Engineering knowledge of activation function mathematics *(Level 3)*
- **PO-2**: Problem analysis for activation function selection *(Level 2)*
- **PO-3**: Basic design decisions for neural network components *(Level 1)*

---

## Resources & References

### **Mandatory Reading**
- **Chollet, "Deep Learning with Python" (2018)**
  - **Chapter 2.3**: "The gears of neural networks" 
  - **Chapter 3.1**: "Anatomy of a neural network"

### **Mathematical Foundation**  
- **Goodfellow et al., "Deep Learning" (2017)**
  - **Chapter 6.3**: "Hidden units" (activation functions analysis)
  - **Chapter 6.4**: "Architecture design"

### **Supplementary Resources**
- **Aggarwal, "Neural Networks and Deep Learning" (2018)** - Chapter 1.3
- **TensorFlow Documentation:** Activation functions API reference
- **Online:** Neural network visualization tools