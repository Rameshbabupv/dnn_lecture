# Unit Test 1 - Option 3B: Hybrid Assessment - Session 2 (Practical)
**Course**: 21CSE558T - Deep Neural Network Architectures
**Duration**: 60 minutes
**Total Marks**: 50 (out of 100 total)
**Date**: September 19, 2025
**Session**: 2 of 2

---

## Instructions
- This is Session 2 of the hybrid assessment
- Complete practical tasks using provided computer setup
- Save all your work with proper file naming
- Write comments explaining your code
- Partial credit given for working solutions

---

## PART C: CODE DEBUGGING (30 marks)
*Debug and fix the given problematic networks*

### Task C1: Fix Vanishing Gradient Network (15 marks)

**Given Code:**
```python
import tensorflow as tf
import numpy as np

# Problematic deep network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='sigmoid', input_shape=(10,)),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='sgd', loss='mse')

# Sample data
X = np.random.randn(100, 10)
y = np.random.randn(100, 1)
```

**Your Tasks:**
1. **Identify the problems** (5 marks)
   - List at least 3 issues with this network
   - Explain why each is problematic

2. **Fix the network** (10 marks)
   - Replace problematic activations
   - Use appropriate initialization
   - Choose better optimizer and loss function
   - Write the corrected code

**Expected Solution Structure:**
```python
# Your fixed network
improved_model = tf.keras.Sequential([
    # YOUR IMPROVEMENTS HERE
])

improved_model.compile(
    optimizer='your_choice',
    loss='your_choice',
    metrics=['your_metrics']
)

# Explain your choices in comments
```

### Task C2: Gradient Explosion Fix (15 marks)

**Scenario:** The following training loop suffers from exploding gradients.

**Given Code:**
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# Training step
@tf.function
def train_step(X_batch, y_batch):
    with tf.GradientTape() as tape:
        predictions = model(X_batch)
        loss = tf.reduce_mean(tf.square(predictions - y_batch))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

**Your Tasks:**
1. **Identify the cause** (5 marks)
   - What causes gradients to explode here?
   - Which parameters are problematic?

2. **Implement gradient clipping** (10 marks)
   - Add gradient clipping to the training step
   - Use appropriate clipping threshold
   - Test that it prevents explosion

**Expected Solution:**
```python
@tf.function
def improved_train_step(X_batch, y_batch):
    with tf.GradientTape() as tape:
        predictions = model(X_batch)
        loss = tf.reduce_mean(tf.square(predictions - y_batch))

    gradients = tape.gradient(loss, model.trainable_variables)

    # YOUR GRADIENT CLIPPING IMPLEMENTATION HERE

    optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
    return loss
```

---

## PART D: ARCHITECTURE DESIGN (20 marks)
*Design optimal network architectures for specific scenarios*

### Task D1: MNIST Classification Network (10 marks)

**Requirements:**
- Design a network for MNIST digit classification (28×28 images → 10 classes)
- Network must be at least 3 hidden layers deep
- Must avoid vanishing/exploding gradient problems
- Should achieve reasonable accuracy

**Your Tasks:**
1. **Design the architecture** (6 marks)
   ```python
   # Your MNIST classifier
   mnist_model = tf.keras.Sequential([
       # YOUR ARCHITECTURE HERE
       # Include comments explaining each choice
   ])
   ```

2. **Justify your choices** (4 marks)
   - Why did you choose these activations?
   - How does your design prevent gradient problems?
   - What initialization method did you use and why?

### Task D2: Optimizer Selection (10 marks)

**Scenario:** You have three different learning tasks:
1. **Task A**: Large dataset (1M samples), stable gradients
2. **Task B**: Small dataset (1K samples), noisy gradients
3. **Task C**: Complex loss landscape, needs adaptive learning rates

**Your Tasks:**
1. **Match optimizers to tasks** (6 marks)
   - For each task, choose: SGD, SGD+Momentum, Adam, or RMSprop
   - Provide specific learning rates

2. **Implementation** (4 marks)
   ```python
   # Task A optimizer
   optimizer_A = # YOUR CHOICE

   # Task B optimizer
   optimizer_B = # YOUR CHOICE

   # Task C optimizer
   optimizer_C = # YOUR CHOICE
   ```

---

## Practical Implementation Requirements

### Setup Instructions:
1. Open Python environment (Jupyter/Colab/IDE)
2. Import required libraries:
   ```python
   import tensorflow as tf
   import numpy as np
   import matplotlib.pyplot as plt
   tf.random.set_seed(42)
   np.random.seed(42)
   ```

### File Submission Format:
- **File Name**: `YourName_Session2_Practical.py` or `.ipynb`
- **Code Structure**: Organize by task (C1, C2, D1, D2)
- **Comments**: Explain your reasoning for each fix/choice
- **Output**: Include any plots or results

### Sample Template:
```python
# ========================================
# Unit Test 1 - Session 2: Practical
# Name: Your Name
# Roll Number: Your Roll Number
# ========================================

# Task C1: Vanishing Gradient Fix
print("=== TASK C1: VANISHING GRADIENT FIX ===")

# Problems identified:
# 1.
# 2.
# 3.

# Fixed network:
improved_model = tf.keras.Sequential([
    # Your improvements with comments
])

# Task C2: Gradient Explosion Fix
print("=== TASK C2: GRADIENT EXPLOSION FIX ===")

# Cause of explosion:
#

# Fixed training step:
@tf.function
def improved_train_step(X_batch, y_batch):
    # Your implementation
    pass

# Task D1: MNIST Architecture
print("=== TASK D1: MNIST ARCHITECTURE ===")

mnist_model = tf.keras.Sequential([
    # Your design with justifications
])

# Task D2: Optimizer Selection
print("=== TASK D2: OPTIMIZER SELECTION ===")

optimizer_A = # Task A choice
optimizer_B = # Task B choice
optimizer_C = # Task C choice

# Justifications:
# Task A:
# Task B:
# Task C:
```

---

## Evaluation Rubric

### Code Quality (20%)
- **Excellent**: Clean, well-commented, follows best practices
- **Good**: Mostly clean, adequate comments
- **Satisfactory**: Basic functionality, minimal comments
- **Poor**: Messy, hard to understand

### Problem Solving (40%)
- **Excellent**: Complete solutions, handles edge cases
- **Good**: Solutions work, minor issues
- **Satisfactory**: Basic solutions, some problems
- **Poor**: Incomplete or incorrect solutions

### Understanding (40%)
- **Excellent**: Deep insights, clear explanations
- **Good**: Good understanding, decent explanations
- **Satisfactory**: Basic understanding, surface explanations
- **Poor**: Little understanding, poor explanations

---

## Time Management Suggestions:
- **Task C1**: 20 minutes
- **Task C2**: 20 minutes
- **Task D1**: 10 minutes
- **Task D2**: 10 minutes

**Total Session 2**: 60 minutes

---

## Combined Assessment Summary:
- **Session 1 (Theory)**: 50 marks
- **Session 2 (Practical)**: 50 marks
- **Total Unit Test 1**: 100 marks

### Overall Grade Distribution:
- **90-100%**: Excellent understanding of theory and practical implementation
- **80-89%**: Good grasp of concepts with minor implementation issues
- **70-79%**: Satisfactory understanding, basic implementation skills
- **60-69%**: Basic concepts understood, needs improvement in application
- **Below 60%**: Significant gaps in understanding, requires additional support