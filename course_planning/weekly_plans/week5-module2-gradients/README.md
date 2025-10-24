# Week 5: Gradient Problems & Regularization
**Module 2: Optimization and Regularization**
**Dates: September 8-12, 2025**

## 📚 Week Overview
This week focuses on understanding and solving critical issues in deep neural network training:
- Unit saturation and gradient flow problems
- Vanishing and exploding gradients
- Overfitting and regularization techniques

## 🎯 Learning Objectives
By the end of Week 5, students will be able to:
1. Identify and diagnose gradient problems in deep networks
2. Implement solutions for vanishing/exploding gradients
3. Apply regularization techniques to prevent overfitting
4. Optimize neural networks using various gradient descent algorithms

## 📅 Session Schedule

### Day 3 - Sunday Evening Session
**Time:** 10:30 PM - 12:10 AM (100 minutes)
**File:** `day3-sunday-lecture.md`
**Topics:**
- Vanishing gradient problem
- Exploding gradient problem
- Solutions: ReLU, proper initialization, batch normalization
- Gradient clipping and monitoring

### Day 4 - Tuesday Morning Tutorial
**Time:** 6:30 AM - 7:20 AM (50 minutes)
**File:** `day4-tuesday-tutorial.md`
**Topics:**
- Hands-on overfitting detection
- L2 regularization implementation
- Dropout layers
- Early stopping
- Combining regularization techniques

## 💻 Lab Exercises

### Tutorial Task T6: Gradient Descent Optimization
**File:** `lab-exercise-t6.py`

This comprehensive lab exercise includes:
- **GradientDescentVisualizer**: Compare SGD, Momentum, Adam, RMSprop, Adagrad
- **GradientProblemsDemonstrator**: Visualize vanishing/exploding gradients
- Practical implementation of gradient clipping
- Performance comparison across optimizers

### Running the Lab Exercise
```bash
# Activate the virtual environment
source ../../../labs/srnenv/bin/activate

# Install required dependencies (if needed)
pip install tensorflow numpy matplotlib scikit-learn

# Run the exercise
python lab-exercise-t6.py
```

## 📊 Key Concepts Covered

### 1. Gradient Problems
- **Vanishing Gradients**: Gradients become exponentially small in deep networks
- **Exploding Gradients**: Gradients grow exponentially large
- **Root Causes**: Poor activation functions, weight initialization

### 2. Solutions Implemented
- **Activation Functions**: ReLU vs Sigmoid comparison
- **Weight Initialization**: He/Xavier initialization
- **Batch Normalization**: Maintaining stable distributions
- **Gradient Clipping**: Preventing explosion

### 3. Regularization Techniques
- **L1/L2 Regularization**: Weight decay
- **Dropout**: Random neuron deactivation
- **Early Stopping**: Prevent overtraining
- **Data Augmentation**: Increase training diversity

## 🔧 Implementation Examples

### Quick Reference: Fixing Gradient Problems
```python
# Problem: Vanishing gradients with sigmoid
model_bad = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='sigmoid'),  # Bad
    tf.keras.layers.Dense(128, activation='sigmoid'),  # Bad
])

# Solution: ReLU + Batch Norm + Proper Init
model_good = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu',
                         kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
])
```

### Quick Reference: Regularization
```python
# Combine multiple regularization techniques
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.BatchNormalization(),
])

# Compile with gradient clipping
model.compile(
    optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),
    loss='binary_crossentropy'
)

# Train with early stopping
early_stop = tf.keras.callbacks.EarlyStopping(patience=10)
model.fit(X_train, y_train, callbacks=[early_stop])
```

## 📝 Assignments

### For Next Session
1. Complete Tutorial Task T6 implementation
2. Compare at least 3 different optimizers on your own dataset
3. Implement gradient monitoring for your neural network
4. Document which regularization technique works best for your problem

### Reading Materials
- Deep Learning Book (Goodfellow et al.) - Chapter 8: Optimization
- "Understanding the difficulty of training deep feedforward neural networks" (Glorot & Bengio, 2010)
- "Batch Normalization: Accelerating Deep Network Training" (Ioffe & Szegedy, 2015)

## 🎓 Assessment Preparation
This week's content is crucial for:
- **Unit Test II** (October 31): Optimization and regularization questions
- **Lab Assessment**: Implementation of T6
- **Final Exam**: Understanding gradient flow and regularization

## 💡 Tips for Success
1. **Visualize Everything**: Plot gradient norms, loss curves, accuracy trends
2. **Start Simple**: Begin with shallow networks, add depth gradually
3. **Monitor Training**: Watch for NaN values, loss explosions
4. **Experiment**: Try different combinations of techniques
5. **Document Results**: Keep track of what works for your specific problem

## 🔗 Additional Resources
- [TensorFlow Gradient Tape Guide](https://www.tensorflow.org/guide/autodiff)
- [Keras Optimizers Documentation](https://keras.io/api/optimizers/)
- [Regularization in Neural Networks](https://keras.io/api/layers/regularization_layers/)

## ⚠️ Common Pitfalls to Avoid
1. Using sigmoid/tanh in deep networks without batch norm
2. Initializing weights with wrong variance
3. Setting learning rate too high with gradient problems
4. Applying dropout during inference
5. Not monitoring validation metrics

## 📧 Support
For questions or clarifications:
- Post in course forum
- Office hours: Tuesday/Thursday 2-4 PM
- Email: instructor@srm.edu.in

---
**Note:** All code examples are compatible with TensorFlow 2.x and Python 3.8+