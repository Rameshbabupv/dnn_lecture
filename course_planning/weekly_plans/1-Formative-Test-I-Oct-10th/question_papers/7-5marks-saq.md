# 5-Mark Short Answer Questions (SAQs) - Selected Set
**Course Code**: 21CSE558T | **Course Title**: Deep Neural Network Architectures
**Test**: Formative Assessment I | **Coverage**: Module 1 & Module 2
**Total Questions**: 7 (5-mark explanatory questions)

---

## 📚 Question Bank Overview

This document contains 7 carefully selected questions from the original 20-question bank, maintaining balanced coverage across modules, difficulty levels, and core concepts.

**⚠️ FORMAT**: Questions ask "Why/What happens/Give reasons" and expect **flowing paragraph answers** with cause-effect reasoning.

---

# 5-MARK SAQ QUESTIONS (7 Selected Questions)

## Module 1: Introduction to Deep Learning (3 questions)

### Q1. You try to train a single perceptron to solve the XOR problem but it fails to converge even after many epochs. Explain why this happens and what fundamental limitation causes this failure.
**Module Coverage**: Module 1 | **Week Coverage**: Week 1-2 | **Lecture**: Perceptron and Boolean Logic
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI1 | **Marks**: 5 | **Difficulty**: Easy

**Expected Answer**:
The XOR problem requires separating points (0,1) and (1,0) from (0,0) and (1,1), which cannot be achieved with a single straight line. A single perceptron can only create linear decision boundaries, but XOR is not linearly separable and requires a non-linear boundary that a single perceptron cannot produce.

---

### Q2. You build a deep neural network but use only linear activation functions. Despite having multiple layers, the network performs like a simple linear model. Explain why this happens.
**Module Coverage**: Module 1 | **Week Coverage**: Week 3, Day 3 | **Lecture**: Activation Functions Deep Dive
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI3 | **Marks**: 5 | **Difficulty**: Easy

**Expected Answer**:
Linear activation functions cause each layer to perform only linear transformations, and the composition of linear functions is still linear. Without non-linear activation functions, no matter how many layers you stack, the entire network behaves as a single linear transformation, making it impossible to learn complex patterns like XOR.

---

### Q3. You replace sigmoid activations with ReLU in your deep network and observe faster training and better performance. Explain what causes this improvement.
**Module Coverage**: Module 1 | **Week Coverage**: Week 3, Day 3 | **Lecture**: Classical vs Modern Activation Functions
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL3 | **PI**: CO1-PI3 | **Marks**: 5 | **Difficulty**: Moderate

**Expected Answer**:
Sigmoid activations have a maximum derivative of 0.25, causing gradients to vanish exponentially in deep networks, which slows learning in early layers. ReLU has a derivative of 1 for positive inputs, allowing gradients to flow unchanged through many layers, enabling faster training and better performance in deep networks.

---

## Module 2: Optimization & Regularization (4 questions)

### Q4. Your model achieves 99% accuracy on training data but only 65% on test data. Explain what problem this indicates and why it occurs.
**Module Coverage**: Module 2 | **Week Coverage**: Week 6 | **Lecture**: Overfitting vs Underfitting Mastery
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI3 | **Marks**: 5 | **Difficulty**: Easy

**Expected Answer**:
This indicates overfitting, where the model has memorized the training data rather than learning generalizable patterns. The large gap between training and test accuracy occurs because the model has learned to fit noise and specific details in the training set that don't represent the underlying data distribution, causing poor performance on unseen data.

---

### Q5. You switch from batch gradient descent to stochastic gradient descent and observe that training becomes noisier but sometimes achieves better final results. Explain this trade-off.
**Module Coverage**: Module 2 | **Week Coverage**: Week 4 | **Lecture**: Gradient Descent Variants Implementation
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Marks**: 5 | **Difficulty**: Moderate

**Expected Answer**:
Stochastic gradient descent updates weights after each sample, creating noisy gradients that cause oscillations during training. However, this noise can be beneficial because it helps the optimizer escape local minima and explore the loss landscape more thoroughly, potentially finding better global solutions that batch gradient descent might miss due to its smoother but more constrained path.

---

### Q6. You apply L1 regularization to your network and find that many weights become exactly zero, while L2 regularization only makes weights smaller. Explain what causes this fundamental difference.
**Module Coverage**: Module 2 | **Week Coverage**: Week 6, Day 3 | **Lecture**: L1 vs L2 Regularization Comparison
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI4 | **Marks**: 5 | **Difficulty**: Moderate

**Expected Answer**:
L1 regularization adds the absolute value of weights to the loss (λ∑|w_i|), creating a diamond-shaped constraint that has sharp corners at the axes where weights are exactly zero. L2 regularization uses squared weights (λ∑w_i²), creating a circular constraint that smoothly shrinks weights toward zero but rarely reaches exactly zero, explaining why L1 produces sparsity while L2 produces weight decay.

---

### Q7. You observe that a simple model underfits while a complex model overfits the same dataset. Explain this bias-variance trade-off and how it relates to model complexity.
**Module Coverage**: Module 2 | **Week Coverage**: Week 6 | **Lecture**: Bias-Variance Tradeoff Deep Dive
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI3 | **Marks**: 5 | **Difficulty**: Difficult

**Expected Answer**:
Simple models have high bias (underfitting) because they lack the capacity to capture complex patterns, while complex models have high variance (overfitting) because they can memorize training data noise. The bias-variance trade-off shows that as model complexity increases, bias decreases but variance increases, requiring careful balance to achieve optimal generalization performance on unseen data.

---

## Summary Statistics

### Question Distribution:
- **Total Questions**: 7 (All 5-mark explanatory questions)
- **Module 1**: 3 questions (43%)
- **Module 2**: 4 questions (57%)

### Difficulty Distribution:
- **Easy**: 3 questions (43%)
- **Moderate**: 3 questions (43%)
- **Difficult**: 1 question (14%)

### Topic Coverage:
- **Perceptron limitations** (XOR problem)
- **Activation functions** (Linear vs non-linear, ReLU vs sigmoid)
- **Overfitting/Underfitting** (Training vs test performance)
- **Optimization** (SGD vs batch gradient descent)
- **Regularization** (L1 vs L2 differences)
- **Bias-variance trade-off** (Model complexity impact)

### Course Outcome Distribution:
- **CO-1**: 3 questions (43%)
- **CO-2**: 4 questions (57%)

### Week-wise Coverage:
✅ **Week 1-2**: XOR and perceptron limitations
✅ **Week 3**: Activation functions and their properties
✅ **Week 4**: Gradient descent optimization
✅ **Week 6**: Regularization and bias-variance concepts

---

**📧 Note**: Questions selected to provide balanced coverage of core concepts from Modules 1-2 with representative difficulty distribution.