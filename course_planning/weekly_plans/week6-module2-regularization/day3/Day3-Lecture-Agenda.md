# Day 3 Lecture Agenda - 2 Hours
**Week 6: Regularization Fundamentals**
**Date:** Sep 15, 2025 | **Duration:** 2 Hours | **Format:** Theory + Hands-on Implementation

---

**© 2025 Prof. Ramesh Babu | SRM University | Data Science and Business Systems (DSBS)**
*Course Materials for 21CSE558T - Deep Neural Network Architectures*

---

## Session Overview
**Primary Focus:** Overfitting vs Underfitting + Classical Regularization Techniques
**Learning Style:** Theory → Mathematical Analysis → Practical Implementation
**Preparation for:** Unit Test 1 (Sep 19) & Tutorial T6 completion

---

## Detailed Timeline

### **Hour 1: Overfitting vs Underfitting Mastery** (60 minutes)

#### **Opening & Context Setting** (10 minutes)
- Week 6 objectives and Unit Test 1 preparation roadmap
- Connection to Week 5: Gradient problems → Now solving overfitting
- Why regularization is critical for model generalization

#### **The Bias-Variance Tradeoff Deep Dive** (25 minutes)
- **Mathematical Foundation**
  - Total Error = Bias² + Variance + Irreducible Error
  - High Bias: Underfitting scenarios and identification
  - High Variance: Overfitting patterns and detection

- **Visual Analysis & Real-World Analogies**
  - The "Goldilocks Principle" in machine learning
  - Model complexity curves: Training vs Validation error
  - The Restaurant Training Analogy: Memorizing vs Understanding recipes

#### **Overfitting Detection & Analysis** (20 minutes)
- **Early Warning Signs**
  - Training accuracy ↑↑ but validation accuracy ↓↓
  - Loss divergence patterns in learning curves
  - Model sensitivity to small data changes

- **Hands-on Visualization**
```python
import matplotlib.pyplot as plt
import numpy as np

def plot_overfitting_example():
    """Demonstrate overfitting vs good fit vs underfitting"""
    # Generate sample data with noise
    np.random.seed(42)
    X = np.linspace(0, 1, 20)
    y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.1, X.shape)

    # Three models with different complexity
    X_plot = np.linspace(0, 1, 100)

    # Underfitting: linear model
    z1 = np.polyfit(X, y, 1)
    p1 = np.poly1d(z1)

    # Good fit: moderate polynomial
    z2 = np.polyfit(X, y, 3)
    p2 = np.poly1d(z2)

    # Overfitting: high-degree polynomial
    z3 = np.polyfit(X, y, 15)
    p3 = np.poly1d(z3)

    plt.figure(figsize=(15, 5))

    # Plot all three scenarios
    titles = ['Underfitting (High Bias)', 'Good Fit', 'Overfitting (High Variance)']
    models = [p1, p2, p3]

    for i, (model, title) in enumerate(zip(models, titles)):
        plt.subplot(1, 3, i+1)
        plt.scatter(X, y, alpha=0.8, color='red', label='Training Data')
        plt.plot(X_plot, model(X_plot), 'b-', linewidth=2, label='Model')
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Demo during lecture
plot_overfitting_example()
```

#### **Practical Implementation Break** (5 minutes)

---

### **Hour 2: Classical Regularization Techniques** (60 minutes)

#### **L1 Regularization (LASSO) - The Feature Selector** (25 minutes)
- **Mathematical Foundation**
  - Loss Function: `L_total = L_original + λ∑|w_i|`
  - Geometric interpretation: Diamond-shaped constraint
  - Why L1 drives weights to exactly zero

- **The Sparse Solution Analogy**
  - "Marie Kondo Approach": Keep only essential features
  - Feature selection as automatic byproduct

- **TensorFlow Implementation**
```python
import tensorflow as tf

# L1 Regularization Implementation
model_l1 = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l1(0.01),
                         input_shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l1(0.01)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compare weights before and after training
def analyze_l1_sparsity(model, X_train, y_train):
    """Analyze sparsity induced by L1 regularization"""
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

    # Check initial weights
    initial_weights = model.layers[0].get_weights()[0]
    print(f"Initial non-zero weights: {np.count_nonzero(initial_weights)}/{initial_weights.size}")

    # Train the model
    model.fit(X_train, y_train, epochs=10, verbose=0)

    # Check final weights
    final_weights = model.layers[0].get_weights()[0]
    print(f"Final non-zero weights: {np.count_nonzero(final_weights)}/{final_weights.size}")

    # Visualize sparsity
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(initial_weights.flatten(), bins=50, alpha=0.7, label='Initial')
    plt.title('Initial Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(final_weights.flatten(), bins=50, alpha=0.7, label='Final', color='orange')
    plt.title('Final Weight Distribution (L1 Regularized)')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
```

#### **L2 Regularization (Ridge) - The Weight Smoother** (25 minutes)
- **Mathematical Foundation**
  - Loss Function: `L_total = L_original + λ∑w_i²`
  - Geometric interpretation: Circular constraint
  - Why L2 shrinks weights towards zero but doesn't eliminate them

- **The Weight Distribution Analogy**
  - "Equal Opportunity Employer": Distributes importance across features
  - Prevents any single weight from dominating

- **Comparative Analysis: L1 vs L2**
```python
# Direct comparison implementation
def compare_l1_vs_l2_regularization():
    """Compare L1 and L2 regularization effects"""

    # Create synthetic dataset with correlated features
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

    # Models with different regularization
    models = {
        'No Regularization': tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(20,))
        ]),
        'L1 Regularization': tf.keras.Sequential([
            tf.keras.layers.Dense(1,
                                kernel_regularizer=tf.keras.regularizers.l1(0.1),
                                input_shape=(20,))
        ]),
        'L2 Regularization': tf.keras.Sequential([
            tf.keras.layers.Dense(1,
                                kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                input_shape=(20,))
        ])
    }

    # Train and compare
    results = {}
    for name, model in models.items():
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=100, verbose=0)
        weights = model.layers[0].get_weights()[0].flatten()
        results[name] = weights

    # Visualization
    plt.figure(figsize=(15, 5))
    for i, (name, weights) in enumerate(results.items()):
        plt.subplot(1, 3, i+1)
        plt.bar(range(len(weights)), weights)
        plt.title(f'{name}\nSparsity: {np.sum(np.abs(weights) < 0.01)/len(weights):.2%}')
        plt.xlabel('Feature Index')
        plt.ylabel('Weight Value')

    plt.tight_layout()
    plt.show()
```

#### **Hyperparameter Selection & Best Practices** (10 minutes)
- **Lambda (λ) Selection Strategies**
  - Cross-validation for optimal λ values
  - Typical ranges: L1 (0.001-0.1), L2 (0.001-0.01)
  - Learning rate interaction considerations

---

## Learning Checkpoints

### **After Hour 1 - Students Should Understand:**
✅ Bias-variance tradeoff mathematical foundation
✅ Visual identification of overfitting patterns
✅ Early detection strategies for model problems
✅ Real-world implications of poor generalization

### **After Hour 2 - Students Should Understand:**
✅ L1 vs L2 regularization mathematical differences
✅ When to choose each regularization technique
✅ TensorFlow implementation for both methods
✅ Hyperparameter tuning strategies

---

## Interactive Elements

### **Mathematical Exercises** (Throughout)
- Calculate L1 penalty for weight vector [2, -1, 0.5, -0.3]
- Compare L2 penalty for same weight vector
- Determine optimal λ for given training/validation curves
- Identify overfitting from provided learning curves

### **Conceptual Questions**
- Why does L1 regularization lead to feature selection?
- When would you choose L2 over L1 regularization?
- How does regularization strength affect model capacity?

---

## Visual Aids & Demonstrations

1. **Bias-Variance Decomposition Plots** - Mathematical relationship visualization
2. **Regularization Constraint Geometry** - L1 diamond vs L2 circle
3. **Weight Evolution During Training** - Live regularization effects
4. **Feature Selection Demonstration** - L1 sparsity in action

---

## Assessment Integration

**Unit Test 1 Preparation Topics (Sep 19):**
- Regularization mathematical formulations
- Overfitting vs underfitting identification
- L1 vs L2 comparison and selection criteria
- Hyperparameter tuning concepts

**Tutorial T6 Integration:**
- Implement both L1 and L2 in gradient descent optimizer
- Compare convergence behavior with different λ values

---

## Homework/Preparation for Day 4

1. **Complete** Tutorial T6 L1/L2 implementation sections
2. **Practice** identifying overfitting in provided datasets
3. **Review** mathematical derivations for L1/L2 penalties
4. **Prepare** for advanced regularization techniques (Day 4)

---

## CO-PO Integration & Assessment

### **Course Outcomes Achievement**
- **CO-1** (Network Creation): Building regularized neural networks
- **CO-2** (Multi-layer Networks): Appropriate regularization selection

### **Programme Outcomes Alignment**
- **PO-1**: Engineering knowledge of regularization mathematics *(Level 3)*
- **PO-2**: Problem analysis for overfitting detection *(Level 2)*
- **PO-3**: Design solutions with regularization techniques *(Level 1)*

---

## Resources & References

### **Mandatory Reading**
- **Goodfellow et al., "Deep Learning" (2017)**
  - **Chapter 7**: "Regularization for Deep Learning"
  - **Chapter 5.2**: "Capacity, Overfitting and Underfitting"

### **Mathematical Foundation**
- **Chollet, "Deep Learning with Python" (2018)**
  - **Chapter 4.4**: "Overfitting and Underfitting"
  - **Chapter 3.6**: "Evaluating machine learning models"

### **Supplementary Resources**
- **TensorFlow Documentation:** Regularizers API reference
- **Online:** Interactive bias-variance decomposition tools
- **Papers:** "Understanding the Bias-Variance Tradeoff" (Fortmann-Roe, 2012)