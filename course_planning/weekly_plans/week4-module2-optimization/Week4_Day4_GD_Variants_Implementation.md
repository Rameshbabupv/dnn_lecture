# Week 4 - Day 4: Gradient Descent Variants & Practical Implementation
**Course:** Deep Neural Network Architectures (21CSE558T)  
**Module:** 2 - Optimization and Regularization  
**Duration:** 2 hours  
**Date:** September 4, 2025

## 📋 Class Overview

### Learning Objectives
By the end of this session, students will be able to:
1. **Compare** batch, mini-batch, and stochastic gradient descent
2. **Implement** all three GD variants in Python/TensorFlow
3. **Analyze** convergence patterns and computational trade-offs
4. **Select** appropriate GD variant based on dataset characteristics
5. **Optimize** hyperparameters for real-world scenarios

### Session Structure (120 minutes)
- **Part A (45 min):** Batch vs Mini-batch vs Stochastic GD Theory
- **Break (10 min)**
- **Part B (45 min):** Hands-on Implementation and Comparison
- **Part C (20 min):** Hyperparameter Tuning and Best Practices

---

## 🎯 Part A: Gradient Descent Variants Theory (45 minutes)

### The Problem with Standard GD (10 minutes)

**🚨 Recall Yesterday's Challenge:**
Basic gradient descent works great for simple functions, but real problems have:
- **Millions of training examples** → Expensive to compute full gradient
- **Memory limitations** → Cannot load entire dataset
- **Time constraints** → Need faster convergence

**💡 Solution:** Modify how we compute and apply gradients!

### Batch Gradient Descent (BGD) (10 minutes)

**📊 What We Did Yesterday (Full Batch):**
```python
# Compute gradient using ALL training examples
for epoch in range(num_epochs):
    gradient = 0
    for i in range(m):  # All m examples
        gradient += compute_gradient(x_i, y_i)
    
    gradient = gradient / m  # Average gradient
    weights = weights - learning_rate * gradient
```

**✅ Advantages:**
- **Stable convergence:** Smooth path to minimum
- **True gradient:** Uses complete information
- **Guaranteed convergence:** For convex functions

**❌ Disadvantages:**
- **Slow:** One update per full dataset pass
- **Memory intensive:** Must load entire dataset
- **Computational cost:** Expensive for large datasets

**📈 Convergence Pattern:** Smooth, predictable curve toward minimum

### Stochastic Gradient Descent (SGD) (12 minutes)

**⚡ The Revolutionary Idea:**
"What if we update weights after EVERY single example?"

```python
# Update weights after each training example
for epoch in range(num_epochs):
    for i in range(m):  # Each example individually
        gradient = compute_gradient(x_i, y_i)  # Single example
        weights = weights - learning_rate * gradient
```

**🎯 Key Insight:**
Instead of computing exact gradient, use **noisy estimate** from single example

**✅ Advantages:**
- **Fast updates:** m times more weight updates per epoch
- **Memory efficient:** Only one example at a time
- **Online learning:** Can learn from streaming data
- **Escape local minima:** Noise helps jump out of bad spots

**❌ Disadvantages:**
- **Noisy convergence:** Zigzag path, never perfectly converges
- **Unstable:** High variance in gradient estimates
- **Hyperparameter sensitive:** Learning rate tuning critical

**📈 Convergence Pattern:** Noisy, oscillating around minimum

### Mini-batch Gradient Descent (13 minutes)

**🎯 The Best of Both Worlds:**
"Use small batches of examples - not too big, not too small!"

```python
# Update weights using small batches
batch_size = 32  # Typical: 16, 32, 64, 128, 256
for epoch in range(num_epochs):
    for batch in get_batches(dataset, batch_size):
        gradient = 0
        for example in batch:
            gradient += compute_gradient(example)
        
        gradient = gradient / batch_size  # Average over batch
        weights = weights - learning_rate * gradient
```

**⚖️ The Goldilocks Solution:**

**✅ Advantages:**
- **Balanced convergence:** Smoother than SGD, faster than BGD
- **Computational efficiency:** Vectorized operations on batches
- **Memory manageable:** Reasonable memory requirements
- **Parallel processing:** GPUs excel at batch operations
- **Stable learning:** Reduced variance compared to SGD

**❌ Disadvantages:**
- **Hyperparameter choice:** Must select batch size
- **Not as fast as pure SGD:** Fewer updates per epoch than SGD

**📈 Convergence Pattern:** Moderately smooth path with controlled noise

---

## 💻 Part B: Hands-on Implementation and Comparison (45 minutes)

### Dataset Setup (5 minutes)

**🎯 Demo Dataset:** Simple polynomial regression
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic dataset
np.random.seed(42)
X = np.random.randn(1000, 1)
y = 3*X.squeeze() + 2 + 0.1*np.random.randn(1000)  # y = 3x + 2 + noise

# Split into train/test
X_train, y_train = X[:800], y[:800]
X_test, y_test = X[800:], y[800:]

print(f"Training examples: {len(X_train)}")
print(f"Test examples: {len(X_test)}")
```

### Implementation 1: Batch Gradient Descent (12 minutes)

```python
def batch_gradient_descent(X, y, learning_rate=0.01, epochs=100):
    m = len(X)
    # Initialize parameters
    w = np.random.randn()
    b = np.random.randn()
    
    costs = []
    
    for epoch in range(epochs):
        # Forward pass - ALL examples
        predictions = w * X.squeeze() + b
        cost = np.mean((predictions - y)**2)
        costs.append(cost)
        
        # Compute gradients - ALL examples
        dw = (2/m) * np.sum((predictions - y) * X.squeeze())
        db = (2/m) * np.sum(predictions - y)
        
        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db
        
        if epoch % 10 == 0:
            print(f"BGD Epoch {epoch}: Cost = {cost:.4f}")
    
    return w, b, costs

# Run BGD
w_bgd, b_bgd, costs_bgd = batch_gradient_descent(X_train, y_train)
print(f"BGD Result: w = {w_bgd:.4f}, b = {b_bgd:.4f}")
```

**🔍 Key Observations:**
- One gradient computation per epoch
- Uses all 800 training examples each time
- Smooth cost reduction

### Implementation 2: Stochastic Gradient Descent (12 minutes)

```python
def stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=100):
    m = len(X)
    w = np.random.randn()
    b = np.random.randn()
    
    costs = []
    
    for epoch in range(epochs):
        epoch_cost = 0
        
        # Shuffle data each epoch
        indices = np.random.permutation(m)
        
        for i in indices:
            # Forward pass - SINGLE example
            x_i, y_i = X[i], y[i]
            prediction = w * x_i + b
            cost = (prediction - y_i)**2
            epoch_cost += cost
            
            # Compute gradients - SINGLE example
            dw = 2 * (prediction - y_i) * x_i
            db = 2 * (prediction - y_i)
            
            # Update parameters
            w -= learning_rate * dw
            b -= learning_rate * db
        
        costs.append(epoch_cost / m)
        
        if epoch % 10 == 0:
            print(f"SGD Epoch {epoch}: Cost = {costs[-1]:.4f}")
    
    return w, b, costs

# Run SGD
w_sgd, b_sgd, costs_sgd = stochastic_gradient_descent(X_train, y_train)
print(f"SGD Result: w = {w_sgd:.4f}, b = {b_sgd:.4f}")
```

**🔍 Key Observations:**
- 800 gradient computations per epoch (one per example)
- Much more parameter updates
- Noisy cost function

### Implementation 3: Mini-batch Gradient Descent (12 minutes)

```python
def mini_batch_gradient_descent(X, y, batch_size=32, learning_rate=0.01, epochs=100):
    m = len(X)
    w = np.random.randn()
    b = np.random.randn()
    
    costs = []
    
    for epoch in range(epochs):
        epoch_cost = 0
        
        # Shuffle data
        indices = np.random.permutation(m)
        
        # Create mini-batches
        for i in range(0, m, batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # Forward pass - BATCH
            predictions = w * X_batch.squeeze() + b
            cost = np.mean((predictions - y_batch)**2)
            epoch_cost += cost * len(X_batch)
            
            # Compute gradients - BATCH
            dw = (2/len(X_batch)) * np.sum((predictions - y_batch) * X_batch.squeeze())
            db = (2/len(X_batch)) * np.sum(predictions - y_batch)
            
            # Update parameters
            w -= learning_rate * dw
            b -= learning_rate * db
        
        costs.append(epoch_cost / m)
        
        if epoch % 10 == 0:
            print(f"Mini-batch GD Epoch {epoch}: Cost = {costs[-1]:.4f}")
    
    return w, b, costs

# Run Mini-batch GD
w_mb, b_mb, costs_mb = mini_batch_gradient_descent(X_train, y_train, batch_size=32)
print(f"Mini-batch GD Result: w = {w_mb:.4f}, b = {b_mb:.4f}")
```

**🔍 Key Observations:**
- ~25 gradient computations per epoch (800/32)
- Balance between BGD smoothness and SGD speed
- Moderate noise in cost function

### Visualization and Comparison (4 minutes)

```python
# Compare convergence
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(costs_bgd, 'b-', linewidth=2, label='Batch GD')
plt.title('Batch Gradient Descent')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(costs_sgd, 'r-', alpha=0.7, label='Stochastic GD')
plt.title('Stochastic Gradient Descent')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(costs_mb, 'g-', linewidth=2, label='Mini-batch GD')
plt.title('Mini-batch Gradient Descent')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.grid(True)

plt.tight_layout()
plt.show()

# Print final comparison
print("\n=== FINAL COMPARISON ===")
print(f"True parameters: w = 3.0, b = 2.0")
print(f"BGD: w = {w_bgd:.4f}, b = {b_bgd:.4f}")
print(f"SGD: w = {w_sgd:.4f}, b = {b_sgd:.4f}")
print(f"Mini-batch: w = {w_mb:.4f}, b = {b_mb:.4f}")
```

---

## ⚙️ Part C: Hyperparameter Tuning and Best Practices (20 minutes)

### Batch Size Selection (8 minutes)

**🎯 Batch Size Impact:**
```python
# Test different batch sizes
batch_sizes = [1, 8, 32, 128, 800]  # 1=SGD, 800=BGD
results = {}

for bs in batch_sizes:
    if bs == 800:  # Full batch
        w, b, costs = batch_gradient_descent(X_train, y_train, epochs=50)
    elif bs == 1:  # Stochastic
        w, b, costs = stochastic_gradient_descent(X_train, y_train, epochs=50)
    else:  # Mini-batch
        w, b, costs = mini_batch_gradient_descent(X_train, y_train, batch_size=bs, epochs=50)
    
    results[bs] = {'final_cost': costs[-1], 'convergence_speed': len([c for c in costs if c > 0.1])}

print("Batch Size Analysis:")
for bs, metrics in results.items():
    print(f"Batch size {bs}: Final cost = {metrics['final_cost']:.4f}")
```

**📊 General Guidelines:**
- **Small batches (8-32):** Good for noisy, diverse datasets
- **Medium batches (64-128):** Good balance for most problems  
- **Large batches (256-512):** Good for clean, consistent datasets
- **Rule of thumb:** Powers of 2 (8, 16, 32, 64, 128, 256)

### Learning Rate Tuning (7 minutes)

**⚖️ Learning Rate vs Batch Size Relationship:**
```python
# Optimal learning rates for different batch sizes
optimal_lr = {
    1: 0.001,      # SGD needs smaller LR
    32: 0.01,      # Mini-batch standard
    128: 0.03,     # Larger batches can use larger LR
    800: 0.1       # Full batch can be aggressive
}

# The reason: Larger batches → more stable gradients → can take bigger steps
```

**🎯 Learning Rate Guidelines:**
- **SGD:** Start with 0.001-0.01
- **Mini-batch:** Start with 0.01-0.1  
- **Batch GD:** Start with 0.1-1.0
- **Adaptive:** Use learning rate schedules (decay over time)

### Real-World Considerations (5 minutes)

**💾 Memory Constraints:**
```python
# Memory usage estimation
def estimate_memory(batch_size, features, data_type_bytes=4):
    """Estimate memory usage in MB"""
    memory_mb = (batch_size * features * data_type_bytes) / (1024**2)
    return memory_mb

# Examples
print("Memory Usage Estimates:")
print(f"Batch 32, 1000 features: {estimate_memory(32, 1000):.1f} MB")
print(f"Batch 128, 1000 features: {estimate_memory(128, 1000):.1f} MB")
print(f"Batch 512, 1000 features: {estimate_memory(512, 1000):.1f} MB")
```

**⚡ Computational Efficiency:**
- **GPU utilization:** Larger batches → better GPU usage
- **Vectorization:** Batch operations are faster than loops
- **Communication overhead:** Distributed training prefers larger batches

**🎯 Practical Decision Framework:**
1. **Start with mini-batch GD (batch_size=32)**
2. **If converging too slowly → increase batch size**
3. **If memory issues → decrease batch size**
4. **If oscillating too much → increase batch size**
5. **Monitor GPU utilization and adjust accordingly**

---

## 📊 Summary and Key Takeaways

### Algorithm Comparison Table

| Method | Batch Size | Updates/Epoch | Convergence | Memory | Speed | Best For |
|--------|------------|---------------|-------------|--------|-------|----------|
| Batch GD | Full (m) | 1 | Smooth | High | Slow | Small datasets |
| SGD | 1 | m | Noisy | Low | Fast | Online learning |
| Mini-batch | 16-256 | m/batch_size | Moderate | Medium | Fast | Most problems |

### When to Use What

**🎯 Use Batch GD when:**
- Small dataset (< 10,000 examples)
- Need deterministic results
- Have unlimited computational resources

**🎯 Use SGD when:**
- Online/streaming data
- Very limited memory
- Need to escape local minima

**🎯 Use Mini-batch GD when:**
- Most practical scenarios
- Have GPU acceleration
- Dataset size > 10,000 examples

---

## 🎯 Learning Outcomes Check

### Quick Assessment
1. **What's the main advantage of mini-batch over batch GD?**
   - *Answer: Faster convergence with manageable memory usage*

2. **Why might SGD escape local minima better than batch GD?**
   - *Answer: Noise in gradient estimates can help jump out of local minima*

3. **How do you choose batch size?**
   - *Answer: Balance between memory constraints, convergence stability, and computational efficiency*

### Hands-on Understanding
Students should now be able to:
- ✅ Implement all three GD variants
- ✅ Visualize convergence differences  
- ✅ Select appropriate hyperparameters
- ✅ Understand computational trade-offs

---

## 📚 Instructor Notes

### Demo Tips
1. **Start with visualization:** Show the different convergence patterns first
2. **Interactive coding:** Let students modify batch sizes and learning rates
3. **Real-time comparison:** Run all three algorithms simultaneously
4. **Emphasize trade-offs:** Every choice has pros and cons

### Common Student Questions
- **"Why not always use the fastest method?"** → Trade-offs explanation
- **"How do I know if my batch size is good?"** → Monitor convergence and resource usage
- **"Can I change batch size during training?"** → Advanced topic, mention briefly

### Next Session Preparation
- **Tutorial T4:** Students will implement these algorithms
- **Week 5 Preview:** Vanishing gradients and advanced optimization
- **Practical assignment:** Compare GD variants on real dataset

---

## 🔗 Connections

### This Session Links To:
- **Tutorial T4:** Building neural network with gradient descent
- **Tutorial T5:** Keras implementation (uses mini-batch by default)
- **Week 5:** Advanced optimization (momentum, Adam)

### Real-World Applications:
- **TensorFlow/PyTorch:** Default mini-batch GD with batch_size=32
- **Production systems:** Auto-scaling batch size based on hardware
- **Research:** Experimenting with batch sizes for different problems

---

**End of Day 4 Content**
**Total Duration:** 2 hours (120 minutes)  
**Key Deliverable:** Complete understanding and implementation of all GD variants
**Student Outcome:** Practical optimization skills for neural network training