# Week 4 - Day 4: Assessment and Exercises - Gradient Descent Variants

**Course:** Deep Neural Network Architectures (21CSE558T)  
**Module:** 2 - Optimization and Regularization  
**Topic:** Gradient Descent Variants Comparison

---

## 🎯 Learning Assessment Framework

### Pre-Class Quick Check (2 minutes)
**Before starting implementation, check student understanding from Day 3:**

**Q1:** What is the basic gradient descent update equation?  
**Expected Answer:** θ ← θ - α∇J(θ)

**Q2:** What does the learning rate α control?  
**Expected Answer:** Step size of parameter updates

**Q3:** What happens if α is too large?  
**Expected Answer:** Algorithm may overshoot and diverge

---

## 📝 In-Class Assessment Activities

### Assessment 1: Concept Check (5 minutes)
**After explaining the three variants, test understanding:**

#### Multiple Choice Questions

**1. For a dataset with 10,000 examples, how many parameter updates will Stochastic GD perform in one epoch?**
- a) 1
- b) 10
- c) 1,000
- d) 10,000 ✓

**2. Which gradient descent variant typically requires the smallest learning rate?**
- a) Batch GD
- b) Stochastic GD ✓
- c) Mini-batch GD
- d) All require the same learning rate

**3. If your model's cost function is oscillating wildly during training, which adjustment should you try first?**
- a) Increase learning rate
- b) Decrease learning rate ✓
- c) Increase batch size ✓
- d) Both b and c ✓

**4. For most practical deep learning applications, which variant is preferred?**
- a) Batch GD
- b) Stochastic GD
- c) Mini-batch GD ✓
- d) It doesn't matter

---

### Assessment 2: Implementation Debugging (10 minutes)
**Present buggy code and ask students to identify issues:**

```python
# BUGGY CODE - Find the errors!
def buggy_mini_batch_gd(X, y, batch_size=32, epochs=100, lr=0.5):
    w, b = 1.0, 1.0  # Error 1: Poor initialization
    
    for epoch in range(epochs):
        # Error 2: Not shuffling data
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            
            predictions = w * X_batch + b
            
            # Error 3: Wrong gradient calculation
            dw = (predictions - y_batch).mean()
            db = (predictions - y_batch).mean()
            
            # Error 4: Learning rate too large
            w -= lr * dw
            b -= lr * db
    
    return w, b
```

**Expected Student Answers:**
1. **Poor initialization:** Should use small random values, not 1.0
2. **No data shuffling:** Should shuffle each epoch for better convergence
3. **Wrong gradient for dw:** Should multiply by X_batch: `dw = ((predictions - y_batch) * X_batch).mean()`
4. **Learning rate too large:** 0.5 is typically too large for mini-batch GD

---

### Assessment 3: Prediction Exercise (8 minutes)
**Before running code, ask students to predict outcomes:**

```python
# Given these three setups, predict which will converge fastest and most stably:

# Setup A: Batch GD
batch_gd(X, y, learning_rate=0.1, epochs=50)

# Setup B: SGD  
stochastic_gd(X, y, learning_rate=0.01, epochs=50)

# Setup C: Mini-batch GD
mini_batch_gd(X, y, batch_size=32, learning_rate=0.05, epochs=50)
```

**Prediction Questions:**
1. **Which will have the smoothest convergence curve?** *(Answer: A - Batch GD)*
2. **Which will make the most parameter updates?** *(Answer: B - SGD)*
3. **Which is most practical for real applications?** *(Answer: C - Mini-batch)*
4. **Which might find a better final solution?** *(Answer: B - SGD, due to noise helping escape local minima)*

---

## 🧪 Hands-on Exercises

### Exercise 1: Parameter Sensitivity Analysis (Individual - 10 minutes)

```python
def exercise_1_learning_rates():
    """
    Test different learning rates with mini-batch GD
    Students should complete this function
    """
    X, y = create_dataset()  # Provided
    
    # TODO: Test these learning rates: [0.001, 0.01, 0.1, 0.5, 1.0]
    learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
    results = {}
    
    for lr in learning_rates:
        # TODO: Run mini-batch GD with this learning rate
        # Record final cost and whether it converged
        pass
    
    # TODO: Plot results and identify best learning rate
    return results

# Expected observations:
# - 0.001: Very slow convergence
# - 0.01: Good convergence
# - 0.1: Fast convergence
# - 0.5: May oscillate
# - 1.0: Likely diverges
```

**Assessment Criteria:**
- ✅ **Correct implementation** (3 points)
- ✅ **Proper learning rate testing** (2 points)  
- ✅ **Correct identification of best LR** (2 points)
- ✅ **Good visualization** (1 point)
- ✅ **Insightful analysis** (2 points)

---

### Exercise 2: Batch Size Optimization (Pairs - 15 minutes)

```python
def exercise_2_batch_sizes():
    """
    Compare different batch sizes and their effects
    Work in pairs - one codes, one analyzes
    """
    X, y = create_large_dataset(m=5000)  # Larger dataset
    
    # TODO: Test batch sizes: [1, 16, 32, 64, 128, 256, 1000]
    batch_sizes = [1, 16, 32, 64, 128, 256, 1000]
    
    results = {}
    for batch_size in batch_sizes:
        # TODO: Measure:
        # 1. Final cost
        # 2. Training time  
        # 3. Convergence smoothness (std of last 10 costs)
        # 4. Memory usage (estimate)
        pass
    
    # TODO: Create comparison plots
    # TODO: Identify optimal batch size
    return results

# Expected observations:
# - batch_size=1 (SGD): Fast but noisy
# - batch_size=32-64: Good balance
# - batch_size=1000: Smooth but slower
```

**Assessment Questions:**
1. **Which batch size converged fastest?** 
2. **Which was most stable?**
3. **What's the trade-off between batch size and convergence smoothness?**
4. **If you had limited GPU memory, which would you choose?**

---

### Exercise 3: Real Dataset Challenge (Groups of 3 - 20 minutes)

```python
def exercise_3_real_challenge():
    """
    Apply all three GD variants to Boston Housing dataset
    Groups compete to achieve best performance
    """
    from sklearn.datasets import load_boston
    from sklearn.preprocessing import StandardScaler
    
    # Load real dataset
    boston = load_boston()
    X = StandardScaler().fit_transform(boston.data)
    y = boston.target
    
    # Challenge: Implement all three variants and compare
    # Goals:
    # 1. Achieve lowest final cost
    # 2. Fastest convergence
    # 3. Most stable training
    
    results = {
        'batch_gd': None,      # TODO: Implement
        'sgd': None,           # TODO: Implement  
        'mini_batch_gd': None  # TODO: Implement
    }
    
    # Bonus: Try different hyperparameters
    return results
```

**Competition Criteria:**
- **Lowest validation error** (5 points)
- **Best hyperparameter tuning** (3 points)
- **Clearest analysis and presentation** (2 points)

---

## 📊 Exit Assessment

### Quick Exit Ticket (3 minutes)
**Students must answer before leaving:**

**1. Fill in the blank:**
"For a dataset with 1,000 examples and batch_size=50, mini-batch GD will perform _____ parameter updates per epoch."

**Answer:** 20 updates (1000/50 = 20 batches)

**2. Scenario question:**
"You're training on a dataset with 1 million examples on a machine with limited memory. Which GD variant would you choose and why?"

**Sample Answer:** "Stochastic GD or mini-batch GD with small batch size (16-32) because they use less memory per update compared to batch GD which would need to load all 1M examples."

**3. Troubleshooting:**
"Your loss curve shows wild oscillations. List two things you would try to fix this:"

**Sample Answers:** 
- Reduce learning rate
- Increase batch size
- Check for data preprocessing issues
- Try gradient clipping

---

### Take-Home Assignment (Due next class)

```python
"""
TAKE-HOME CHALLENGE: GD Variants on Your Own Dataset

Instructions:
1. Find a regression dataset (sklearn, Kaggle, or create synthetic)
2. Implement all three GD variants
3. Create comprehensive comparison including:
   - Convergence plots
   - Final accuracy comparison
   - Training time analysis
   - Memory usage estimates
   - Hyperparameter sensitivity analysis

Deliverable: Jupyter notebook with analysis and conclusions

Grading Criteria:
- Implementation correctness (40%)
- Experimental design (25%)
- Analysis depth (25%)
- Presentation quality (10%)

Bonus: Compare with sklearn's SGD implementation
"""
```

---

## 🎯 Assessment Rubric

### Knowledge Check Rubric

| Level | Criteria | Points |
|-------|----------|---------|
| **Excellent (9-10)** | Perfect understanding of all three variants, can predict behavior, debug code, and select appropriate method for scenarios | 9-10 |
| **Proficient (7-8)** | Good understanding of variants, minor errors in predictions, can implement with guidance | 7-8 |
| **Developing (5-6)** | Basic understanding, needs help with implementation, some conceptual gaps | 5-6 |
| **Beginning (3-4)** | Limited understanding, major conceptual errors, cannot implement independently | 3-4 |
| **Inadequate (0-2)** | Little to no understanding, cannot distinguish between variants | 0-2 |

### Practical Implementation Rubric

| Aspect | Excellent (4) | Good (3) | Fair (2) | Poor (1) |
|---------|---------------|----------|----------|----------|
| **Code Quality** | Clean, well-commented, follows best practices | Mostly clean, some comments | Functional but messy | Many bugs, hard to read |
| **Algorithm Understanding** | Perfect implementation of all variants | Minor errors in 1-2 variants | Major errors but shows understanding | Fundamental misunderstanding |
| **Analysis** | Insightful conclusions, identifies trade-offs | Good analysis, most conclusions correct | Basic analysis, some insights | Limited analysis |
| **Visualization** | Clear, informative plots with proper labels | Good plots, minor issues | Basic plots, some missing elements | Poor or missing visualizations |

---

## 🔍 Common Student Errors and Solutions

### Error 1: Incorrect Gradient Calculation for Mini-batch
**Student Code:**
```python
dw = (predictions - y_batch).mean()  # WRONG!
```
**Correction:**
```python
dw = ((predictions - y_batch) * X_batch).mean()  # CORRECT
```

### Error 2: Using Same Learning Rate for All Variants
**Issue:** Students use lr=0.1 for all methods
**Solution:** Explain that SGD typically needs smaller learning rate (0.01), mini-batch needs medium (0.05), batch can use larger (0.1)

### Error 3: Not Shuffling Data
**Issue:** Students process data in same order each epoch
**Solution:** Emphasize importance of `np.random.permutation()` each epoch

### Error 4: Batch Size Larger Than Dataset
**Issue:** Setting batch_size=1000 for dataset with 500 examples
**Solution:** Show that this automatically becomes batch GD, explain with `min(batch_size, dataset_size)`

---

## 📈 Performance Tracking

### Individual Student Progress
Track each student's performance on:
- [ ] Conceptual understanding (quiz scores)
- [ ] Implementation accuracy (code review)
- [ ] Analysis quality (exercise reports)
- [ ] Participation in discussions

### Class-Level Metrics
Monitor:
- [ ] Average quiz score (target: >80%)
- [ ] Percentage completing exercises (target: >90%)
- [ ] Quality of take-home submissions
- [ ] Time spent on debugging vs. new concepts

---

## 🎯 Learning Outcome Verification

By end of class, students should demonstrate:

### CO-1 Alignment: Simple Deep Neural Networks
- ✅ **Knowledge:** Understand gradient-based optimization
- ✅ **Comprehension:** Explain trade-offs between GD variants
- ✅ **Application:** Implement all three variants correctly

### CO-2 Alignment: Multi-layer Networks  
- ✅ **Analysis:** Compare convergence patterns
- ✅ **Synthesis:** Select appropriate variant for scenarios
- ✅ **Evaluation:** Assess performance and make improvements

### Success Indicators:
- **90%** of students can implement basic variants
- **80%** can explain trade-offs clearly  
- **70%** can select appropriate method for new scenarios
- **60%** can debug and optimize hyperparameters

---

**Assessment Summary:**  
Total class activities: ~45 minutes of assessment  
Take-home work: 3-4 hours estimated  
Assessment weight: 15% of total course grade (typical lab assessment)