# Unit Test 1 - Option 2: Hands-On Programming Assessment
**Course**: 21CSE558T - Deep Neural Network Architectures
**Duration**: 2 hours (120 minutes)
**Total Marks**: 100
**Date**: September 19, 2025
**Platform**: Google Colab

---

## Instructions
- Complete all programming tasks in Google Colab
- Submit your notebook with all outputs visible
- Include comments explaining your approach
- Test your code before submission
- Partial credit will be given for working solutions

---

## TASK 1: Vanishing Gradient Diagnosis and Fix (25 marks)

### Problem Statement
You are given a deep neural network that suffers from vanishing gradients. Your task is to diagnose the problem and implement solutions.

### Given Code (Starter Template)
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Problematic network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='sigmoid', input_shape=(10,)),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Sample data
X = tf.random.normal((100, 10))
y = tf.random.uniform((100, 1))
```

### Your Tasks:
**A)** Implement gradient monitoring (8 marks)
- Compute gradients using GradientTape
- Calculate gradient norms for each layer
- Print gradient magnitudes with status indicators

**B)** Create gradient visualization (7 marks)
- Plot gradient magnitudes by layer
- Use log scale to show the vanishing effect
- Add appropriate labels and legend

**C)** Fix the vanishing gradient problem (10 marks)
- Replace sigmoid activations with appropriate alternatives
- Implement proper weight initialization
- Re-run gradient analysis to show improvement
- Document the improvements achieved

### Expected Output:
- Gradient analysis before fix showing vanishing gradients
- Visualization plots comparing before/after
- Fixed network with healthy gradient flow

---

## TASK 2: Optimization Algorithm Comparison (30 marks)

### Problem Statement
Compare the performance of different optimization algorithms on a neural network training task.

### Your Tasks:
**A)** Implement three optimizers (15 marks)
```python
# Create identical networks for fair comparison
def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

# Implement and train with:
# 1. SGD
# 2. Adam
# 3. RMSprop
```

**B)** Training comparison (10 marks)
- Train each model for 10 epochs on MNIST data
- Record training history (loss and accuracy)
- Use same batch size and initial conditions

**C)** Performance analysis (5 marks)
- Create comparison plots for loss curves
- Calculate final accuracies
- Provide written analysis of results

### Expected Output:
- Three trained models with different optimizers
- Comparative loss/accuracy plots
- Analysis explaining optimizer differences

---

## TASK 3: Gradient Clipping Implementation (20 marks)

### Problem Statement
Implement gradient clipping to prevent exploding gradients and demonstrate its effectiveness.

### Your Tasks:
**A)** Create exploding gradient scenario (8 marks)
```python
# Network prone to exploding gradients
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(100,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Large learning rate to cause explosion
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
```

**B)** Implement gradient clipping (8 marks)
- Create custom training loop with GradientTape
- Implement both norm clipping and value clipping
- Apply clipping before optimizer.apply_gradients()

**C)** Demonstrate effectiveness (4 marks)
- Train with and without clipping
- Plot gradient norms over training steps
- Show how clipping prevents explosion

### Expected Code Structure:
```python
@tf.function
def train_step_with_clipping(X_batch, y_batch, clip_norm=1.0):
    with tf.GradientTape() as tape:
        predictions = model(X_batch)
        loss = tf.reduce_mean(tf.square(predictions - y_batch))

    gradients = tape.gradient(loss, model.trainable_variables)
    # YOUR CLIPPING IMPLEMENTATION HERE
    clipped_gradients = # YOUR CODE

    optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
    return loss
```

---

## TASK 4: Normalization Techniques (15 marks)

### Problem Statement
Compare the effects of different normalization techniques on network performance.

### Your Tasks:
**A)** Implement three versions (9 marks)
```python
# Version 1: No normalization
model_none = tf.keras.Sequential([...])

# Version 2: Batch normalization
model_batch = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation=None),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    # Continue pattern...
])

# Version 3: Layer normalization
model_layer = tf.keras.Sequential([...])  # Your implementation
```

**B)** Training comparison (6 marks)
- Train all three models on same dataset
- Compare convergence speed and final accuracy
- Create comparison plots

### Expected Output:
- Three different normalization implementations
- Training curves showing normalization benefits
- Written summary of observations

---

## TASK 5: Architecture Design Challenge (10 marks)

### Problem Statement
Design an optimal network architecture for a specific problem while avoiding gradient issues.

### Scenario:
"Design a network for CIFAR-10 classification (32×32×3 images, 10 classes) that:
- Has at least 5 layers
- Uses modern techniques to avoid gradient problems
- Achieves reasonable performance"

### Your Tasks:
**A)** Architecture design (6 marks)
- Use appropriate activations, initialization, normalization
- Include regularization techniques
- Justify your design choices in comments

**B)** Implementation and testing (4 marks)
- Implement your designed architecture
- Train for a few epochs to verify it works
- Report key metrics and gradient health

### Expected Output:
```python
# Your optimized architecture
optimal_model = tf.keras.Sequential([
    # YOUR DESIGN HERE with justification comments
])

# Training and evaluation
# YOUR IMPLEMENTATION
```

---

## Evaluation Rubric

| Component | Excellent (90-100%) | Good (80-89%) | Satisfactory (70-79%) | Needs Improvement (<70%) |
|-----------|-------------------|---------------|----------------------|---------------------------|
| **Code Quality** | Clean, well-commented, efficient | Mostly clean, some comments | Basic functionality, minimal comments | Poor structure, hard to understand |
| **Problem Solving** | Complete solutions, handles edge cases | Solutions work, minor issues | Basic solutions, some problems | Incomplete or incorrect solutions |
| **Analysis** | Deep insights, clear explanations | Good understanding, decent analysis | Basic analysis, surface-level | Little to no analysis |
| **Visualization** | Professional plots, clear labels | Good plots, mostly clear | Basic plots, adequate | Poor or missing visualizations |

---

## Submission Requirements
1. **Google Colab Notebook**: Complete with all outputs visible
2. **File Naming**: `StudentName_UnitTest1_Option2.ipynb`
3. **Code Comments**: Explain your approach for each task
4. **Output Requirements**: All plots and results must be displayed
5. **Deadline**: Submit before the end of exam time

---

## Technical Setup Instructions
1. Open Google Colab
2. Import required libraries:
   ```python
   import tensorflow as tf
   import numpy as np
   import matplotlib.pyplot as plt
   from tensorflow.keras import layers, models
   ```
3. Load MNIST dataset:
   ```python
   (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
   X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
   y_train = tf.keras.utils.to_categorical(y_train, 10)
   ```

**Note**: Starter code templates will be provided in shared Colab notebook before the exam.