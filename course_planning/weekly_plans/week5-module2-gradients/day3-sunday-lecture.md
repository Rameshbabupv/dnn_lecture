# Week 5 - Day 3: Vanishing & Exploding Gradients
**Date:** Sunday, September 14, 2025
**Time:** 10:30 PM - 12:10 AM (100 minutes)
**Module:** 2 - Optimization and Regularization
**Topic:** Unit Saturation & Gradient Problems

## Learning Objectives
By the end of this session, students will be able to:
1. Understand the mathematical basis of vanishing and exploding gradients
2. Identify situations where gradient problems occur
3. Implement solutions to mitigate gradient issues
4. Analyze gradient flow in deep networks

## Session Structure

### Part 1: Introduction to Gradient Problems (10:30 PM - 10:50 PM)
**20 minutes**

#### Opening Discussion (5 min)
- Review of backpropagation chain rule
- Why deep networks face unique challenges

#### The Gradient Flow Problem (15 min)
- Mathematical foundation: ∂L/∂w₁ = ∂L/∂y × ∂y/∂h_n × ... × ∂h₂/∂h₁ × ∂h₁/∂w₁
- Multiplication of many small/large numbers
- Visual demonstration with 10-layer network

### Part 2: Vanishing Gradients (10:50 PM - 11:20 PM)
**30 minutes**

#### The Vanishing Gradient Problem (15 min)
- Sigmoid activation: σ(x) = 1/(1 + e^(-x))
- Derivative: σ'(x) = σ(x)(1 - σ(x)) ≤ 0.25
- Compounding effect in deep networks
- Impact on learning in early layers

#### Practical Demonstration (15 min)
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Create a deep network with sigmoid activations
def create_deep_sigmoid_network(depth=10):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='sigmoid', input_shape=(10,)))

    for _ in range(depth-2):
        model.add(tf.keras.layers.Dense(64, activation='sigmoid'))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

# Analyze gradient magnitudes
def analyze_gradients(model, x_sample):
    with tf.GradientTape() as tape:
        y_pred = model(x_sample)
        loss = tf.reduce_mean(tf.square(y_pred - 1.0))

    gradients = tape.gradient(loss, model.trainable_variables)

    gradient_norms = []
    for i, grad in enumerate(gradients):
        if grad is not None and i % 2 == 0:  # Weights only, skip biases
            norm = tf.norm(grad).numpy()
            gradient_norms.append(norm)
            print(f"Layer {i//2}: Gradient norm = {norm:.6f}")

    return gradient_norms

# Demonstration
model = create_deep_sigmoid_network(10)
x_sample = tf.random.normal((32, 10))
gradient_norms = analyze_gradients(model, x_sample)

# Visualization
plt.figure(figsize=(10, 6))
plt.semilogy(range(len(gradient_norms)), gradient_norms, 'bo-')
plt.xlabel('Layer (from input to output)')
plt.ylabel('Gradient Norm (log scale)')
plt.title('Vanishing Gradient Effect in Deep Sigmoid Network')
plt.grid(True)
plt.show()
```

### Part 3: Exploding Gradients (11:20 PM - 11:40 PM)
**20 minutes**

#### The Exploding Gradient Problem (10 min)
- When weights are initialized too large
- Unbounded activation functions
- Gradient accumulation in RNNs
- NaN values and training instability

#### Detection and Visualization (10 min)

**What We're Detecting:**
- Gradient explosion: When gradient norms exceed safe thresholds (>100)
- NaN/Inf values: Complete training failure
- Training instability: High variance in gradient norms

**What We're Visualizing:**
- Gradient norm evolution across epochs
- Explosion events marked on timeline
- Distribution of gradient magnitudes
- Layer-wise gradient analysis

```python
# SEE COMPLETE INTERACTIVE NOTEBOOK: week5_gradient_problems_colab.ipynb

# Quick demonstration code for lecture:
def detect_and_visualize_explosion():
    """Detect gradient explosion and visualize the problem"""

    # Create unstable network (poor initialization)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='linear',
                            kernel_initializer=tf.random_normal_initializer(stddev=2.0),
                            input_shape=(10,)),
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_initializer=tf.random_normal_initializer(stddev=2.0)),
        tf.keras.layers.Dense(1)
    ])

    # Training setup
    X = tf.random.normal((100, 10))
    y = tf.random.normal((100, 1))
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)  # High LR

    # Detection variables
    explosion_threshold = 100.0
    gradient_history = []
    explosion_events = []

    print("Monitoring for gradient explosion...")

    for epoch in range(20):
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.reduce_mean(tf.square(y_pred - y))

        gradients = tape.gradient(loss, model.trainable_variables)

        # DETECTION: Calculate total gradient norm
        total_norm = tf.reduce_sum([tf.norm(g) for g in gradients if g is not None])
        gradient_history.append(total_norm.numpy())

        # DETECTION: Check if explosion occurred
        if total_norm > explosion_threshold:
            explosion_events.append(epoch)
            print(f"⚠️ EXPLOSION at epoch {epoch}: norm = {total_norm:.2f}")

        # DETECTION: Check for NaN/Inf
        if tf.math.is_nan(total_norm) or tf.math.is_inf(total_norm):
            print(f"💥 CRITICAL: NaN/Inf detected at epoch {epoch}!")
            break

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # VISUALIZATION
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Gradient norm over time
    epochs = range(len(gradient_history))
    ax1.plot(epochs, gradient_history, 'b-', linewidth=2)
    ax1.axhline(y=explosion_threshold, color='r', linestyle='--',
                label=f'Explosion Threshold ({explosion_threshold})')

    # Mark explosion events
    for event in explosion_events:
        ax1.scatter(event, gradient_history[event], color='red',
                   s=100, marker='x', linewidths=3, zorder=5)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Gradient Norm')
    ax1.set_title('Gradient Explosion Detection')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Gradient distribution
    ax2.hist(gradient_history, bins=20, color='purple', alpha=0.7, edgecolor='black')
    ax2.axvline(x=explosion_threshold, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Gradient Norm')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Gradient Norms')

    plt.tight_layout()
    plt.show()

    return gradient_history, explosion_events

# Run detection and visualization
grad_history, explosions = detect_and_visualize_explosion()
print(f"\nSummary: {len(explosions)} explosions detected in {len(grad_history)} epochs")
```

**Key Detection Metrics:**
- Gradient norm threshold: Typically 100-1000
- Explosion frequency: Number of events per training
- Maximum gradient norm reached
- Epochs until first explosion

### Part 4: Solutions and Mitigation Strategies (11:40 PM - 12:00 AM)
**20 minutes**

#### Solutions for Vanishing Gradients (10 min)
1. **ReLU Activation Functions**
   - f(x) = max(0, x)
   - Derivative = 1 for x > 0
   - No saturation in positive region

2. **He/Xavier Initialization**
   - Xavier: Var(W) = 2/(n_in + n_out)
   - He: Var(W) = 2/n_in

3. **Batch Normalization**
   - Normalizes inputs to each layer
   - Maintains stable distributions

4. **Residual Connections (Skip Connections)**
   - Allows gradient to flow directly

#### Solutions for Exploding Gradients (10 min)
1. **Gradient Clipping**
```python
# Implement gradient clipping
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

# Or manual clipping
def clip_gradients(gradients, max_norm=1.0):
    clipped_grads = []
    for grad in gradients:
        if grad is not None:
            clipped_grads.append(tf.clip_by_norm(grad, max_norm))
        else:
            clipped_grads.append(grad)
    return clipped_grads
```

2. **Weight Regularization**
   - L2 regularization prevents weights from growing too large

3. **Proper Weight Initialization**
   - Use appropriate initialization schemes

4. **Learning Rate Scheduling**
   - Reduce learning rate when gradients are large

### Part 5: Hands-on Implementation (12:00 AM - 12:10 AM)
**10 minutes**

#### Quick Exercise: Fix a Problematic Network
```python
# Students will fix this network
def create_problematic_network():
    """This network has gradient problems - fix it!"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='sigmoid', input_shape=(784,)),
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Solution
def create_improved_network():
    """Fixed version with better gradient flow"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu',
                            kernel_initializer='he_normal',
                            input_shape=(784,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(128, activation='relu',
                            kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(128, activation='relu',
                            kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile with gradient clipping
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
```

## Assignment for Next Session
1. Implement gradient monitoring for your T6 neural network
2. Compare training with and without batch normalization
3. Read paper: "Understanding the difficulty of training deep feedforward neural networks" (Glorot & Bengio, 2010)

## Resources
- TensorFlow Gradient Documentation
- Deep Learning Book Chapter 8 (Goodfellow et al.)
- Batch Normalization Paper (Ioffe & Szegedy, 2015)

## Assessment Questions
1. Why do sigmoid activations cause vanishing gradients?
2. How does gradient clipping prevent explosion?
3. What is the intuition behind He initialization?
4. How do residual connections help gradient flow?