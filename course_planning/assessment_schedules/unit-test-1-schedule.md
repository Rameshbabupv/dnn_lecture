# Unit Test 1 Schedule - Modules 1 & 2

**Date:** September 19, 2025 (Week 6)  
**Duration:** 90 minutes  
**Coverage:** Module 1 (Introduction to Deep Learning) + Module 2 (Optimization & Regularization)  
**Weight:** 22.5% of total course grade (45% of CLA-1)

---

## 📋 Test Overview

### Scope and Coverage
**Module 1: Introduction to Deep Learning (40% of test)**
- Biological neurons and artificial neuron models
- Perceptron and multilayer perceptron architectures
- XOR problem and the need for hidden layers
- Activation functions and their properties
- TensorFlow fundamentals and basic operations

**Module 2: Optimization & Regularization (60% of test)**
- Gradient descent variants (batch, mini-batch, SGD)
- Loss functions and cost function optimization
- Backpropagation algorithm and computational graphs
- Regularization techniques (L1, L2, dropout)
- Learning rate scheduling and adaptive optimizers

---

## 📊 Question Distribution

### Theory Section (50 points - 55 minutes)
**Question Types:**
- **Multiple Choice (20 points):** 20 questions × 1 point each
- **Short Answer (15 points):** 5 questions × 3 points each  
- **Analytical Problems (15 points):** 3 questions × 5 points each

**Content Distribution:**
- Module 1 concepts: 20 points
- Module 2 concepts: 25 points
- Integration/application: 5 points

### Practical Section (50 points - 35 minutes)
**Coding Problems:**
- **TensorFlow Implementation (20 points):** Complete a neural network for classification
- **Algorithm Debugging (15 points):** Identify and fix errors in provided code
- **Hyperparameter Analysis (15 points):** Analyze model performance and suggest improvements

---

## 📚 Study Guide and Topics

### Module 1 - Key Topics for Review

#### Biological and Artificial Neurons
- [ ] Neuron structure and signal transmission
- [ ] Mathematical model of artificial neuron
- [ ] Threshold vs. activation function approaches
- [ ] Linear separability and perceptron limitations

#### Network Architectures
- [ ] Single-layer perceptron capabilities and limitations
- [ ] XOR problem: why single layer fails
- [ ] Multilayer perceptron architecture design
- [ ] Universal approximation theorem (conceptual)

#### Activation Functions
- [ ] Step function, sigmoid, tanh, ReLU properties
- [ ] Vanishing gradient problem with sigmoids
- [ ] Advantages of ReLU and variants (Leaky ReLU, ELU)
- [ ] Choosing appropriate activation functions

#### TensorFlow Fundamentals
- [ ] Tensor operations and shapes
- [ ] Building simple networks with Keras
- [ ] Model compilation and training basics
- [ ] Basic debugging and error interpretation

### Module 2 - Key Topics for Review

#### Optimization Algorithms
- [ ] Batch gradient descent: concept and implementation
- [ ] Stochastic gradient descent (SGD): advantages and challenges
- [ ] Mini-batch gradient descent: balancing efficiency and stability
- [ ] Learning rate selection and its impact

#### Advanced Optimization
- [ ] Momentum: concept and mathematical formulation
- [ ] Adaptive learning rates: AdaGrad, RMSprop, Adam
- [ ] Learning rate scheduling strategies
- [ ] Convergence analysis and stopping criteria

#### Regularization Techniques
- [ ] Overfitting vs. underfitting identification
- [ ] L1 regularization: sparsity and feature selection
- [ ] L2 regularization: weight decay and smoothness
- [ ] Dropout: concept, implementation, and best practices
- [ ] Early stopping and validation-based regularization

#### Loss Functions and Backpropagation
- [ ] Mean squared error for regression problems
- [ ] Cross-entropy loss for classification
- [ ] Computational graph representation
- [ ] Chain rule application in backpropagation
- [ ] Gradient flow and vanishing gradient issues

---

## 💻 Practical Preparation

### Coding Skills Assessment
Students should be prepared to:
- [ ] Implement a simple MLP using TensorFlow/Keras
- [ ] Configure optimizers with appropriate parameters
- [ ] Apply regularization techniques to prevent overfitting
- [ ] Debug common training issues (convergence, overfitting)
- [ ] Interpret training curves and model performance metrics

### Sample Code Scenarios
**Network Implementation:**
```python
# Be prepared to complete or debug similar code
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=?, activation='?'),
    tf.keras.layers.Dropout(?),
    tf.keras.layers.Dense(units=?, activation='?')
])

model.compile(optimizer=tf.keras.optimizers.?(learning_rate=?),
              loss='?',
              metrics=['accuracy'])
```

**Optimization Configuration:**
```python
# Understand parameter selection rationale
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999
)
```

---

## 📈 Assessment Criteria

### Theory Section Grading
**Multiple Choice (20 points):**
- Clear understanding of fundamental concepts
- Ability to distinguish between similar techniques
- Knowledge of appropriate applications

**Short Answer (15 points):**
- Accurate explanations of algorithms and concepts
- Clear reasoning and logical progression
- Proper use of technical terminology

**Analytical Problems (15 points):**
- Mathematical accuracy in calculations
- Logical problem-solving approach
- Integration of multiple concepts

### Practical Section Grading
**Implementation Quality (20 points):**
- Correct syntax and API usage
- Appropriate parameter selection
- Code efficiency and clarity

**Debugging Skills (15 points):**
- Accurate error identification
- Correct fixes with proper justification
- Understanding of common pitfalls

**Analysis and Optimization (15 points):**
- Correct interpretation of performance metrics
- Appropriate hyperparameter recommendations
- Clear reasoning for suggested improvements

---

## 🕐 Test Day Schedule

### Before the Test
- **1 Week Prior:** Review sessions and practice problems
- **3 Days Prior:** Final study guide distribution
- **1 Day Prior:** Q&A session for clarifications

### Test Day Timeline
- **Arrival:** 15 minutes before start time
- **Setup:** 10 minutes for system check and instructions
- **Theory Section:** 55 minutes (50 points)
- **Break:** 5 minutes (optional)
- **Practical Section:** 35 minutes (50 points)
- **Buffer:** 5 minutes for submission and technical issues

### Post-Test
- **Immediate:** Preliminary feedback on common issues
- **48 Hours:** Detailed results and individual feedback
- **1 Week:** Review session for missed concepts

---

## 🎯 Success Strategies

### Study Preparation
- [ ] **Practice Problems:** Complete all tutorial exercises (T1-T5)
- [ ] **Concept Mapping:** Create visual connections between topics
- [ ] **Code Review:** Understand every line in provided examples
- [ ] **Group Study:** Discuss concepts with classmates
- [ ] **Mock Tests:** Practice with time constraints

### Test-Taking Tips
- [ ] **Time Management:** Allocate appropriate time per section
- [ ] **Read Carefully:** Understand requirements before coding
- [ ] **Partial Credit:** Show work even if final answer is uncertain
- [ ] **Check Work:** Reserve time for review and debugging
- [ ] **Stay Calm:** Systematic approach to problem-solving

---

## 🔄 Remediation and Support

### For Students Needing Additional Support
**Pre-Test:**
- Individual tutoring sessions available
- Additional practice problems provided
- Concept clarification through office hours

**Post-Test:**
- Detailed performance analysis and feedback
- Targeted review sessions for missed concepts
- Additional assignments for skill reinforcement
- Preparation support for Unit Test 2

### Resources for Success
- **Study Groups:** Peer learning opportunities
- **TA Support:** Regular help sessions
- **Online Resources:** Supplementary materials and videos
- **Practice Platform:** Additional coding exercises

---

## 📋 Administrative Details

### Technical Requirements
- **Platform:** University computer lab or personal laptop
- **Software:** TensorFlow 2.x, Jupyter Notebook or Google Colab
- **Internet:** Required for accessing test platform
- **Backup:** USB with required software installations

### Academic Integrity
- **Individual Work:** No collaboration during test
- **Resources:** Only specified reference materials allowed
- **Code Originality:** Original implementation required
- **Documentation:** Proper citation of any referenced materials

### Accommodations
- **Extended Time:** Available for students with documented needs
- **Alternative Format:** Accessible versions for visual/motor impairments
- **Quiet Environment:** Separate testing room if required
- **Technical Support:** On-site assistance for platform issues