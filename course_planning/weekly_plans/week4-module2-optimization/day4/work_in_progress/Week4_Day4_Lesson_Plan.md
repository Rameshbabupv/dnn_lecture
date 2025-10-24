# Week 4 - Day 4: Gradient Descent Variants - From Theory to Practice

**Course:** Deep Neural Network Architectures (21CSE558T)  
**Module:** 2 - Optimization and Regularization  
**Duration:** 60 minutes  
**Date:** Week 4, Day 4  
**Prerequisites:** Day 3 - Basic Optimization Theory

---

## 📋 Session Overview

### Learning Objectives
By the end of this session, students will be able to:
1. **Distinguish** between Batch, Mini-batch, and Stochastic Gradient Descent
2. **Implement** all three variants in Python with NumPy
3. **Analyze** convergence patterns and computational trade-offs
4. **Select** appropriate GD variant based on dataset characteristics
5. **Optimize** hyperparameters for different scenarios

### Course Outcome Alignment
- **CO-1:** Implement simple deep neural networks → **Level 3** (Algorithm implementation)
- **CO-2:** Apply multi-layer networks → **Level 2** (Optimization understanding)

### Program Outcome Mapping
- **PO-1:** Engineering Mathematics → **Level 3** (Applied optimization)
- **PO-2:** Problem Analysis → **Level 3** (Algorithm comparison)
- **PO-3:** Design/Development → **Level 2** (Implementation skills)

---

## ⏰ **Phase 1: Recap & Motivation (5 minutes)**

### Connection Bridge from Day 3
**Opening Question:** *"Yesterday we learned the optimization equation θ ← θ − α∇J(θ). What happens when we have 1 million training examples?"*

### The Central Challenge
```
Day 3 Approach: Use ALL examples to compute gradient
↓
Problem: Very slow for large datasets!
↓
Today's Solution: Smart gradient computation strategies
```

### Today's Core Question
**"How many examples should we use to compute each gradient update?"**

**Three Answers:**
- **All examples** → Batch Gradient Descent
- **One example** → Stochastic Gradient Descent  
- **Small groups** → Mini-batch Gradient Descent

---

## 📊 **Phase 2: The Three Gradient Descent Variants (15 minutes)**

### Variant 1: Batch Gradient Descent (BGD) - "The Perfectionist"

**Core Concept:** Use ALL training examples for each gradient computation

```python
# Conceptual Algorithm
for epoch in range(num_epochs):
    # Compute gradient using ENTIRE dataset
    gradient = compute_gradient_all_examples(X, y, weights)
    
    # Single parameter update per epoch
    weights = weights - learning_rate * gradient
```

**Mathematical Formulation:**
```
θ^(t+1) = θ^(t) - α · (1/m) Σ(i=1 to m) ∇J(f_θ(x_i), y_i)
```

**Characteristics:**
- ✅ **Stable convergence:** Smooth path to minimum
- ✅ **True gradient:** Uses complete dataset information
- ✅ **Deterministic:** Same result every run
- ❌ **Slow:** Only 1 update per epoch
- ❌ **Memory intensive:** Must load entire dataset
- ❌ **Poor scalability:** Impractical for large datasets

**When to Use:**
- Small datasets (< 10,000 examples)
- Research scenarios requiring reproducible results
- When computational resources are unlimited

---

### Variant 2: Stochastic Gradient Descent (SGD) - "The Speedster"

**Core Concept:** Use ONE example at a time for gradient computation

```python
# Conceptual Algorithm
for epoch in range(num_epochs):
    # Shuffle data each epoch
    shuffled_indices = random.permutation(dataset_size)
    
    for i in shuffled_indices:
        # Compute gradient using SINGLE example
        gradient = compute_gradient_single_example(x_i, y_i, weights)
        
        # Immediate parameter update
        weights = weights - learning_rate * gradient
```

**Mathematical Formulation:**
```
θ^(t+1) = θ^(t) - α · ∇J(f_θ(x_i), y_i)  [for randomly selected i]
```

**Characteristics:**
- ✅ **Frequent updates:** m updates per epoch (where m = dataset size)
- ✅ **Memory efficient:** Processes one example at a time
- ✅ **Online learning:** Can handle streaming data
- ✅ **Escape local minima:** Noise helps exploration
- ❌ **Noisy convergence:** Zigzag path, high variance
- ❌ **Unstable:** Oscillates around minimum
- ❌ **Sensitive:** Requires careful learning rate tuning

**When to Use:**
- Streaming/online data scenarios
- Severe memory constraints
- When intentional noise aids exploration

---

### Variant 3: Mini-batch Gradient Descent - "The Goldilocks Solution"

**Core Concept:** Use small batches of examples (typically 16-256)

```python
# Conceptual Algorithm
batch_size = 32  # Common choice
for epoch in range(num_epochs):
    # Shuffle data each epoch
    shuffled_data = shuffle(dataset)
    
    # Process in mini-batches
    for batch in create_batches(shuffled_data, batch_size):
        # Compute gradient using batch
        gradient = compute_gradient_batch(batch, weights)
        
        # Update after each batch
        weights = weights - learning_rate * gradient
```

**Mathematical Formulation:**
```
θ^(t+1) = θ^(t) - α · (1/|B|) Σ(i∈B) ∇J(f_θ(x_i), y_i)  [B = mini-batch]
```

**Characteristics:**
- ✅ **Balanced convergence:** Smoother than SGD, faster than BGD
- ✅ **Computational efficiency:** Vectorized operations
- ✅ **GPU-friendly:** Excellent hardware utilization
- ✅ **Manageable memory:** Reasonable resource requirements
- ✅ **Practical:** Industry standard approach
- ❌ **Hyperparameter choice:** Must select batch size

**Common Batch Sizes:**
- **16-32:** Small datasets or limited memory
- **64-128:** Most common choice for deep learning
- **256-512:** Large datasets with sufficient resources

**When to Use:**
- Most practical deep learning scenarios (95% of cases)
- When you have GPU acceleration available
- Datasets larger than 10,000 examples

---

## 💻 **Phase 3: Hands-on Implementation (25 minutes)**

### Dataset Setup and Problem Definition

**Problem:** Linear regression with synthetic data
**Goal:** Learn parameters for y = 2x + 1 + noise

```python
import numpy as np
import matplotlib.pyplot as plt

# Create synthetic dataset
np.random.seed(42)
X = np.random.randn(1000, 1)
y = 2*X.squeeze() + 1 + 0.1*np.random.randn(1000)

print(f"Dataset properties:")
print(f"  Examples: {len(X)}")
print(f"  Features: {X.shape[1]}")
print(f"  Target relationship: y = 2x + 1 + noise")
print(f"  X range: [{X.min():.2f}, {X.max():.2f}]")
print(f"  y range: [{y.min():.2f}, {y.max():.2f}]")
```

### Implementation 1: Batch Gradient Descent (7 minutes)

```python
def batch_gradient_descent(X, y, epochs=50, learning_rate=0.1, verbose=True):
    """
    Batch Gradient Descent implementation
    
    Parameters:
    - X: Input features (m, n)
    - y: Target values (m,)
    - epochs: Number of training epochs
    - learning_rate: Step size for parameter updates
    - verbose: Print progress updates
    
    Returns:
    - w: Learned weight
    - b: Learned bias
    - costs: List of costs per epoch
    """
    # Initialize parameters
    w = np.random.randn() * 0.1
    b = np.random.randn() * 0.1
    costs = []
    m = len(X)
    
    if verbose:
        print("=== Batch Gradient Descent ===")
        print(f"Initial parameters: w={w:.4f}, b={b:.4f}")
    
    for epoch in range(epochs):
        # Forward pass - compute predictions for ALL examples
        predictions = w * X.squeeze() + b
        
        # Compute cost (Mean Squared Error)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        costs.append(cost)
        
        # Compute gradients using ALL examples
        dw = (1/m) * np.sum((predictions - y) * X.squeeze())
        db = (1/m) * np.sum(predictions - y)
        
        # Parameter update (one update per epoch)
        w -= learning_rate * dw
        b -= learning_rate * db
        
        # Progress reporting
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:2d}: Cost = {cost:.6f}, w = {w:.4f}, b = {b:.4f}")
    
    if verbose:
        print(f"Final: w = {w:.4f}, b = {b:.4f}")
        print(f"Updates per epoch: 1")
        print(f"Total parameter updates: {epochs}")
    
    return w, b, costs

# Execute Batch GD
w_batch, b_batch, costs_batch = batch_gradient_descent(X, y)
```

### Implementation 2: Stochastic Gradient Descent (8 minutes)

```python
def stochastic_gradient_descent(X, y, epochs=50, learning_rate=0.01, verbose=True):
    """
    Stochastic Gradient Descent implementation
    
    Parameters:
    - X: Input features (m, n)
    - y: Target values (m,)
    - epochs: Number of training epochs
    - learning_rate: Step size (typically smaller than BGD)
    - verbose: Print progress updates
    
    Returns:
    - w: Learned weight  
    - b: Learned bias
    - costs: List of average costs per epoch
    """
    # Initialize parameters
    w = np.random.randn() * 0.1
    b = np.random.randn() * 0.1
    costs = []
    m = len(X)
    
    if verbose:
        print("\n=== Stochastic Gradient Descent ===")
        print(f"Initial parameters: w={w:.4f}, b={b:.4f}")
        print(f"Learning rate: {learning_rate} (smaller than BGD)")
    
    for epoch in range(epochs):
        epoch_cost = 0
        
        # Shuffle data each epoch for better convergence
        indices = np.random.permutation(m)
        
        # Process each example individually
        for i in indices:
            # Forward pass - SINGLE example
            x_i, y_i = X[i], y[i]
            prediction = w * x_i + b
            
            # Compute cost for this example
            cost = 0.5 * (prediction - y_i)**2
            epoch_cost += cost
            
            # Compute gradients for SINGLE example
            dw = (prediction - y_i) * x_i
            db = (prediction - y_i)
            
            # Immediate parameter update
            w -= learning_rate * dw
            b -= learning_rate * db
        
        # Average cost for the epoch
        avg_cost = epoch_cost / m
        costs.append(avg_cost)
        
        # Progress reporting
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:2d}: Cost = {avg_cost:.6f}, w = {w:.4f}, b = {b:.4f}")
    
    if verbose:
        print(f"Final: w = {w:.4f}, b = {b:.4f}")
        print(f"Updates per epoch: {m}")
        print(f"Total parameter updates: {epochs * m}")
    
    return w, b, costs

# Execute Stochastic GD (note the smaller learning rate!)
w_sgd, b_sgd, costs_sgd = stochastic_gradient_descent(X, y)
```

### Implementation 3: Mini-batch Gradient Descent (10 minutes)

```python
def mini_batch_gradient_descent(X, y, batch_size=32, epochs=50, learning_rate=0.05, verbose=True):
    """
    Mini-batch Gradient Descent implementation
    
    Parameters:
    - X: Input features (m, n)
    - y: Target values (m,)
    - batch_size: Number of examples per batch
    - epochs: Number of training epochs
    - learning_rate: Step size (between BGD and SGD rates)
    - verbose: Print progress updates
    
    Returns:
    - w: Learned weight
    - b: Learned bias
    - costs: List of average costs per epoch
    """
    # Initialize parameters
    w = np.random.randn() * 0.1
    b = np.random.randn() * 0.1
    costs = []
    m = len(X)
    
    # Calculate number of batches per epoch
    num_batches = int(np.ceil(m / batch_size))
    
    if verbose:
        print("\n=== Mini-batch Gradient Descent ===")
        print(f"Initial parameters: w={w:.4f}, b={b:.4f}")
        print(f"Batch size: {batch_size}")
        print(f"Batches per epoch: {num_batches}")
        print(f"Learning rate: {learning_rate}")
    
    for epoch in range(epochs):
        epoch_cost = 0
        
        # Shuffle data each epoch
        indices = np.random.permutation(m)
        
        # Process data in mini-batches
        for i in range(0, m, batch_size):
            # Get batch indices
            batch_indices = indices[i:i+batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            batch_m = len(X_batch)
            
            # Forward pass - BATCH of examples
            predictions = w * X_batch.squeeze() + b
            
            # Compute cost for this batch
            batch_cost = (1/(2*batch_m)) * np.sum((predictions - y_batch)**2)
            epoch_cost += batch_cost * batch_m
            
            # Compute gradients for BATCH
            dw = (1/batch_m) * np.sum((predictions - y_batch) * X_batch.squeeze())
            db = (1/batch_m) * np.sum(predictions - y_batch)
            
            # Parameter update after each batch
            w -= learning_rate * dw
            b -= learning_rate * db
        
        # Average cost for the epoch
        avg_cost = epoch_cost / m
        costs.append(avg_cost)
        
        # Progress reporting
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:2d}: Cost = {avg_cost:.6f}, w = {w:.4f}, b = {b:.4f}")
    
    if verbose:
        print(f"Final: w = {w:.4f}, b = {b:.4f}")
        print(f"Updates per epoch: {num_batches}")
        print(f"Total parameter updates: {epochs * num_batches}")
    
    return w, b, costs

# Execute Mini-batch GD
w_minibatch, b_minibatch, costs_minibatch = mini_batch_gradient_descent(X, y, batch_size=32)
```

---

## 📈 **Phase 4: Analysis and Comparison (10 minutes)**

### Comprehensive Results Visualization

```python
def compare_algorithms():
    """Compare all three gradient descent variants"""
    
    # Results summary
    print("\n" + "="*60)
    print("GRADIENT DESCENT VARIANTS COMPARISON")
    print("="*60)
    
    print(f"True parameters:     w = 2.000, b = 1.000")
    print(f"Batch GD:           w = {w_batch:.3f}, b = {b_batch:.3f}")
    print(f"Stochastic GD:      w = {w_sgd:.3f}, b = {b_sgd:.3f}")
    print(f"Mini-batch GD:      w = {w_minibatch:.3f}, b = {b_minibatch:.3f}")
    
    # Convergence patterns visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Batch GD
    plt.subplot(1, 3, 1)
    plt.plot(costs_batch, 'b-', linewidth=2, label='Batch GD')
    plt.title('Batch GD: Smooth Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Stochastic GD
    plt.subplot(1, 3, 2)
    plt.plot(costs_sgd, 'r-', alpha=0.7, label='Stochastic GD')
    plt.title('Stochastic GD: Noisy Path')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Mini-batch GD
    plt.subplot(1, 3, 3)
    plt.plot(costs_minibatch, 'g-', linewidth=2, label='Mini-batch GD')
    plt.title('Mini-batch GD: Balanced')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Comparative analysis
    print("\nConvergence Analysis:")
    print(f"Batch GD final cost:      {costs_batch[-1]:.6f}")
    print(f"Stochastic GD final cost: {costs_sgd[-1]:.6f}")
    print(f"Mini-batch GD final cost: {costs_minibatch[-1]:.6f}")

# Execute comparison
compare_algorithms()
```

### Trade-off Analysis Table

| Metric | Batch GD | Stochastic GD | Mini-batch GD |
|--------|----------|---------------|---------------|
| **Convergence** | Smooth, predictable | Noisy, oscillating | Moderately smooth |
| **Speed** | Slow (1 update/epoch) | Fast (1000 updates/epoch) | Medium (~31 updates/epoch) |
| **Memory** | High (entire dataset) | Low (single example) | Medium (batch size) |
| **Stability** | Very stable | Less stable | Good stability |
| **Final accuracy** | High | Moderate | High |
| **Computational efficiency** | Low | Variable | High |

---

## ✅ **Phase 5: Assessment and Wrap-up (5 minutes)**

### Quick Knowledge Check

**Question 1:** *"For a dataset with 10,000 examples and batch_size=50, how many parameter updates will mini-batch GD perform in 10 epochs?"*
- **Answer:** 200 batches/epoch × 10 epochs = 2,000 updates

**Question 2:** *"If your model is oscillating wildly during training, which two adjustments should you try first?"*
- **Answer:** Reduce learning rate OR increase batch size

**Question 3:** *"In production deep learning, which GD variant is most commonly used and why?"*
- **Answer:** Mini-batch GD - balances speed, stability, and computational efficiency

### Exit Ticket Challenge

```python
# Take-home experiment
def student_challenge():
    """
    Modify the mini-batch implementation to use:
    1. batch_size = 64 instead of 32
    2. learning_rate = 0.08 instead of 0.05
    
    Predict:
    - Will convergence be smoother or noisier?
    - Will it converge faster or slower?
    - Will the final accuracy be better or worse?
    
    Test your predictions and report your findings!
    """
    pass
```

### Next Session Preview

**Week 5 - Day 1: Advanced Optimization**
- **Why basic GD isn't enough for deep networks**
- **Momentum: Adding "memory" to gradient descent**
- **Adaptive learning rates: RMSprop and Adam**
- **The vanishing gradient problem**

---

## 🎯 Key Takeaways

### The Golden Rules of Gradient Descent

1. **Start with mini-batch GD** (batch_size = 32-64)
2. **Adjust learning rate with batch size** (larger batch → larger LR)
3. **Always shuffle your data** each epoch
4. **Monitor convergence patterns** to diagnose issues
5. **Consider your computational constraints** when choosing variant

### Decision Framework

```
Dataset Size < 1,000     → Batch GD
Memory Severely Limited  → Stochastic GD  
Streaming/Online Data    → Stochastic GD
Most Other Cases        → Mini-batch GD
```

### Learning Outcome Verification

Students can now:
- ✅ **Explain** the differences between GD variants
- ✅ **Implement** all three variants from scratch
- ✅ **Visualize** and analyze convergence patterns
- ✅ **Choose** appropriate variant for given scenarios
- ✅ **Tune** hyperparameters effectively

---

**End of Session**  
**Duration:** 60 minutes  
**Next:** Advanced optimization techniques and momentum