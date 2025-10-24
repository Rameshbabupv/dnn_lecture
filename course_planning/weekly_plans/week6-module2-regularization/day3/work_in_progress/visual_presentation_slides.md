# Week 6 Day 3: Visual Presentation Slides
## Overfitting, Underfitting & Classical Regularization

**Course:** 21CSE558T - Deep Neural Network Architectures
**Duration:** 2 Hours
**Date:** September 15, 2025
**Instructor:** Prof. Ramesh Babu

---

# Slide 1: Welcome & Session Overview

## 🎯 Today's Learning Journey

**What We'll Master:**
- **WHY** models fail in the real world
- **WHAT** bias-variance tradeoff really means
- **HOW** to use L1 & L2 regularization effectively

**Our Learning Philosophy:**
> *"Learn through stories, understand through analogies, master through practice"*

**Session Structure:**
- **Hour 1:** The Problem (Overfitting & Underfitting)
- **Hour 2:** The Solution (Classical Regularization)

---

# Slide 2: The Restaurant Chef Dilemma

## 🍽️ Why Do Smart Models Fail?

### Three Chefs, Three Approaches:

**Chef A - The Simplifier** 🥪
- *"Salt + Pepper + Heat = Good Food"*
- Simple but limited (High Bias)

**Chef B - The Balanced** 👨‍🍳
- Learns principles, adapts to situations
- The sweet spot (Good Fit)

**Chef C - The Perfectionist** 🤖
- Memorizes every detail perfectly
- Fails with new customers (High Variance)

### 💡 The Big Question:
*Which chef would you hire for your restaurant?*

---

# Slide 3: The Archer's Target

## 🏹 Understanding Bias & Variance

```
Total Error = Bias² + Variance + Irreducible Error
```

### Visual Guide:

| Scenario | Bias | Variance | Result |
|----------|------|----------|---------|
| **Skilled Archer** ✅ | Low | Low | Clustered at bullseye |
| **Systematic Error** 📍 | High | Low | Clustered away from target |
| **Inconsistent Archer** ⚠️ | Low | High | Scattered around bullseye |
| **Worst Case** ❌ | High | High | Scattered away from target |

### 🎯 Key Insight:
*Perfect practice doesn't mean perfect performance!*

---

# Slide 4: Mathematical Foundation

## 📊 The Bias-Variance Decomposition

### **Bias** - Systematic Error
```
Bias = E[f̂(x)] - f(x)
```
- Model's tendency to consistently miss the true value
- Like always shooting left of the target

### **Variance** - Inconsistency
```
Variance = E[(f̂(x) - E[f̂(x)])²]
```
- Model's sensitivity to training data changes
- Like arrows scattered everywhere

### **The Fundamental Tradeoff:**
- ↑ Model Complexity → ↓ Bias, ↑ Variance
- ↓ Model Complexity → ↑ Bias, ↓ Variance

---

# Slide 5: Student Learning Patterns

## 📚 The Cramming vs Understanding Analogy

### **Student A - The Crammer** (Overfitting) ⚠️
- Memorizes exact practice questions
- Perfect practice scores → Fails real exam
- **ML Translation:** High training acc, low validation acc

### **Student B - The Understander** (Good Fit) ✅
- Learns underlying principles
- Good practice scores → Adapts to real exam
- **ML Translation:** Balanced train-validation performance

### **Student C - The Struggling** (Underfitting) 📈
- Poor performance on both practice and exam
- **ML Translation:** Low training and validation accuracy

---

# Slide 6: Overfitting Detection

## 🚨 Early Warning Signs

### **The Relationship Red Flags:**
1. **Too Perfect** → 100% training accuracy
2. **No Flexibility** → Poor on new data
3. **Memorization** → Knows details, misses big picture
4. **Social Anxiety** → Terrible with test data

### **Mathematical Symptoms:**
```
⚠️ Training Accuracy: 98%+
🚨 Validation Accuracy: <80%
💥 Gap: >15% difference
📈📉 Learning Curves: Diverging trends
```

### **Diagnostic Questions:**
- Is your model *too good* on training data?
- Are training and validation curves diverging?

---

# Slide 7: Hour 1 Summary & Transition

## ✅ What We've Learned So Far

### **Diagnosed the Problems:**
- **Bias-Variance Tradeoff** → Restaurant chef learning
- **Overfitting** → Student cramming behavior
- **Detection Methods** → Early warning systems

### **Now We Need Treatment:**
> *"We've learned to diagnose the disease.
> Now let's learn the medicine!"*

---

# HOUR 2: THE MEDICINE

---

# Slide 8: Marie Kondo for Neural Networks

## ✨ L1 Regularization - The Decluttering Expert

### **The Philosophy:**
> *"Keep only features that spark joy (improve prediction)"*

### **Before L1** (Cluttered House) 🏠
```
Kitchen: [coffee_maker, toaster, 47_unused_gadgets, moldy_cheese]
Features: [useful_1, useful_2, 9,998_random_features]
```

### **After L1** (Minimalist Paradise) ✨
```
Kitchen: [coffee_maker, toaster]  # Only joy-sparking items
Features: [useful_1, useful_2]  # Only predictive features
```

### **The L1 Process:**
- Does this feature improve prediction? → **Keep**
- Does this feature add noise? → **Remove (weight = 0)**

---

# Slide 9: L1 Mathematical Foundation

## 💰 The Budget Allocation Problem

### **Mathematical Form:**
```
Loss_total = Loss_original + λ∑|wᵢ|
```

### **The Company Budget Meeting:**
- **Normal Budget:** Marketing $400K, R&D $300K, Sales $250K, HR $50K
- **L1 Constraint:** Σ|department_budget| ≤ $1M
- **L1 Result:** HR gets $0 (eliminated!)

### **Geometric Interpretation:**
```
|w₁| + |w₂| ≤ λ
```
- **Diamond-shaped constraint**
- **Sharp corners** → Force weights to exactly zero
- **Automatic feature selection**

---

# Slide 10: Equal Opportunity Employer

## 🤝 L2 Regularization - The Fair Workplace

### **The Philosophy:**
> *"No single feature should dominate the model"*

### **Company Culture A** (No Regularization)
- Star employee gets 90% of credit
- Others feel undervalued
- **ML:** One feature dominates

### **Company Culture B** (L2 Regularization)
- Equal opportunity for all employees
- Balanced workload distribution
- **ML:** All features contribute proportionally

### **Real-World Benefits:**
- **Stability** → No single feature can break model
- **Fairness** → All relevant features get a voice
- **Robustness** → Works even if features missing

---

# Slide 11: L2 Mathematical Foundation

## 📈 Investment Portfolio Theory

### **Mathematical Form:**
```
Loss_total = Loss_original + λ∑wᵢ²
```

### **Portfolio Strategy:**
- **Risky Portfolio:** 80% Tesla, 20% others → High risk
- **Diversified Portfolio:** Balanced across sectors → Lower risk
- **L2 Translation:** Spread importance across features

### **Geometric Interpretation:**
```
w₁² + w₂² ≤ λ
```
- **Circular constraint**
- **Smooth boundaries** → Weights shrink gradually
- **No sparsity** → All features remain active

---

# Slide 12: L1 vs L2 Comparison

## ⚖️ The Complete Comparison

| Aspect | L1 (LASSO) | L2 (Ridge) |
|--------|------------|------------|
| **Penalty** | Σ\|wᵢ\| | Σwᵢ² |
| **Geometry** | 💎 Diamond | ⭕ Circle |
| **Effect** | Feature Selection | Weight Smoothing |
| **Sparsity** | ✅ Yes (weights → 0) | ❌ No (weights → small) |
| **Use Case** | Remove irrelevant features | Handle correlated features |
| **Analogy** | 🧹 Marie Kondo | 🤝 Equal Opportunity |
| **Interpretation** | Easy (fewer features) | Moderate (all features) |

### **When to Use Which?**
- **Many irrelevant features** → L1
- **Correlated features** → L2
- **Need interpretability** → L1
- **Need stability** → L2

---

# Slide 13: Hyperparameter Tuning

## 🎛️ Finding the Sweet Spot

### **The Goldilocks Principle:**

| λ Value | Effect | Result |
|---------|--------|---------|
| **Too Small** (< 0.001) | No effect | Still overfitting |
| **Just Right** (0.001-0.1) | Balanced | Good generalization |
| **Too Large** (> 1.0) | Over-regularization | Underfitting |

### **Tuning Strategy:**
1. **Start broad:** Try [0.001, 0.01, 0.1, 1.0]
2. **Cross-validate:** Use k-fold validation
3. **Monitor gap:** Watch train-validation difference
4. **Refine:** Narrow down around best performer

### **Visual Guide:**
```
λ ↑ → Bias ↑, Variance ↓
λ ↓ → Bias ↓, Variance ↑
```

---

# Slide 14: The Model Doctor

## 🏥 Complete Diagnostic System

### **Patient Examination:**
```python
def diagnose_model_health(train_acc, val_acc):
    gap = abs(train_acc - val_acc)

    if train_acc > 0.95 and val_acc < 0.8:
        return "Overfitting - Apply regularization"
    elif train_acc < 0.8 and val_acc < 0.8:
        return "Underfitting - Increase complexity"
    else:
        return "Healthy - Continue approach"
```

### **Treatment Prescriptions:**
- **Overfitting** → L1/L2 regularization, early stopping
- **Underfitting** → Reduce regularization, add complexity
- **Healthy** → Monitor and deploy

---

# Slide 15: Real-World Case Studies

## 🌍 Where This Matters

### **Medical Diagnosis:**
- **Problem:** Model memorizes hospital A, fails at hospital B
- **Solution:** L2 regularization for robust feature weights

### **Finance:**
- **Problem:** Trading model perfect in backtest, loses money live
- **Solution:** L1 for feature selection, L2 for stability

### **Image Recognition:**
- **Problem:** 100% accuracy on training photos, poor on new images
- **Solution:** Combined L1/L2 + data augmentation

### **Text Classification:**
- **Problem:** Memorizes training documents, poor generalization
- **Solution:** L1 for keyword selection, L2 for balanced weights

---

# Slide 16: Implementation in TensorFlow

## 💻 From Theory to Practice

### **L1 Implementation:**
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l1(0.01)),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### **L2 Implementation:**
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### **Combined L1 + L2:**
```python
tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)
```

---

# Slide 17: Assessment Preparation

## 📝 Unit Test 1 Ready Checklist (Sep 19)

### **Mathematical Mastery:**
- [ ] Can derive bias-variance decomposition
- [ ] Understands L1 vs L2 penalty differences
- [ ] Can calculate regularization penalties manually
- [ ] Knows geometric interpretations

### **Conceptual Understanding:**
- [ ] Can explain overfitting using analogies
- [ ] Identifies problems from learning curves
- [ ] Chooses appropriate regularization technique
- [ ] Understands hyperparameter tuning

### **Practical Skills:**
- [ ] Implements L1/L2 in TensorFlow
- [ ] Sets up overfitting detection
- [ ] Tunes λ parameters effectively

---

# Slide 18: Tonight's Homework

## 🏠 Your Mission Until Day 4

### **Must Complete:**
1. **Tutorial T6** → Implement both L1 and L2 regularization
2. **Practice Session** → Identify overfitting from learning curves
3. **Math Review** → Work through penalty calculations
4. **Teach Back** → Explain concepts using our analogies
5. **Read Ahead** → Advanced regularization preview

### **Success Criteria:**
- Can implement regularization without looking at notes
- Can diagnose model problems from visual patterns
- Can explain trade-offs to a non-technical friend

---

# Slide 19: Day 4 Preview

## 🔮 Advanced Regularization Techniques

### **Coming Up:**
- **Dropout** → "The Random Absence Policy"
- **Batch Normalization** → "The Team Coordination System"
- **Data Augmentation** → "The Experience Multiplier"
- **Early Stopping** → "The Perfect Timing Strategy"

### **The Journey Continues:**
> *"We've learned the classical medicine.
> Next, we'll explore the modern treatments!"*

---

# Slide 20: Key Takeaways

## 🎯 Remember These Core Concepts

### **The Analogies:**
1. **Bias-Variance** = Restaurant chef learning strategies
2. **Overfitting** = Student cramming + relationship red flags
3. **L1** = Marie Kondo decluttering
4. **L2** = Equal opportunity + portfolio diversification

### **The Mathematics:**
- **Total Error** = Bias² + Variance + Irreducible Error
- **L1 Penalty** = λ∑|wᵢ| (Diamond constraint)
- **L2 Penalty** = λ∑wᵢ² (Circle constraint)

### **The Wisdom:**
> *"The art of machine learning is not in building perfect models, but in building models that fail gracefully and generalize beautifully."*

---

# Slide 21: Interactive Q&A

## 💬 Let's Test Your Understanding

### **Quick Fire Round:**
1. When would you choose L1 over L2?
2. What's the danger of λ = 0?
3. How do you know if λ is too large?
4. Which regularization creates sparse models?
5. What's the geometric shape of L2 constraint?

### **Think-Pair-Share:**
*"Explain overfitting to someone who's never heard of machine learning using one of today's analogies"*

### **Practical Challenge:**
*"Given a model with 95% training accuracy and 70% validation accuracy, what's your diagnosis and treatment plan?"*

---

# Slide 22: Course Context

## 📍 Where We Are in the Journey

### **Module 2 Progress:**
- ✅ **Week 4** → Gradient Descent Fundamentals
- ✅ **Week 5** → Gradient Problems (Vanishing/Exploding)
- ✅ **Week 6 Day 3** → Classical Regularization ← **WE ARE HERE**
- 🔜 **Week 6 Day 4** → Advanced Regularization

### **Building Towards:**
- **Week 7-9** → Image Processing & CNNs
- **Week 10-12** → Transfer Learning
- **Week 13-15** → Object Detection

### **Learning Outcomes Progress:**
- **CO-1** ✅ → Create simple neural networks
- **CO-2** 🚧 → Build multi-layer networks (in progress)

---

# Slide 23: Thank You

## 🙏 Session Complete!

### **What We Accomplished Today:**
- ✅ Mastered bias-variance tradeoff through analogies
- ✅ Learned to detect overfitting patterns
- ✅ Implemented L1 and L2 regularization
- ✅ Built practical diagnostic tools
- ✅ Prepared for Unit Test 1

### **Your Next Steps:**
1. Complete tonight's homework
2. Practice with Tutorial T6
3. Prepare questions for Day 4
4. Review for Unit Test 1

### **Remember:**
> *"Every expert was once a beginner.
> Every pro was once an amateur.
> Every icon was once an unknown."*

**See you for Advanced Regularization! 🚀**

---

## 📚 Slide References

### **Required Reading:**
- Goodfellow et al., "Deep Learning" Chapter 7
- Chollet, "Deep Learning with Python" Chapter 4.4

### **Online Resources:**
- TensorFlow Regularizers Documentation
- Interactive Bias-Variance Tools
- Course Materials Repository

---

*© 2025 Prof. Ramesh Babu | SRM University | Deep Neural Network Architectures*
*"Making AI Accessible Through Analogies"*