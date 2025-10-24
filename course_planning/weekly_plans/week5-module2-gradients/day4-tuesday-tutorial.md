# Week 5 - Day 4: Overfitting & Regularization Tutorial
**Date:** Tuesday, September 16, 2025
**Time:** 6:30 AM - 7:20 AM (50 minutes)
**Module:** 2 - Optimization and Regularization
**Type:** Tutorial Session (T6 Preparation)
**Topic:** Hands-on: Detecting and Preventing Overfitting

## Learning Objectives
By the end of this tutorial, students will be able to:
1. Detect overfitting in neural networks
2. Implement regularization techniques
3. Use early stopping effectively
4. Apply dropout layers appropriately

## Tutorial Structure

### Part 1: Quick Recap & Setup (6:30 AM - 6:35 AM)
**5 minutes**

#### Environment Setup
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, n_clusters_per_class=2,
                          random_state=42)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")
```

### Part 2: Demonstrating Overfitting (6:35 AM - 6:45 AM)
**10 minutes**

#### Build an Overfitting Model
```python
def create_overfitting_model():
    """Create a model that will overfit"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(20,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

# Train and observe overfitting
model_overfit = create_overfitting_model()

history_overfit = model_overfit.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=0
)

# Visualization function
def plot_training_history(history, title="Training History"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title(f'{title} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Print final metrics
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(f"Overfitting Gap: {final_val_loss - final_train_loss:.4f}")

plot_training_history(history_overfit, "Overfitting Model")
```

### Part 3: Implementing Regularization Techniques (6:45 AM - 7:00 AM)
**15 minutes**

#### Technique 1: L2 Regularization
```python
def create_l2_regularized_model(l2_lambda=0.01):
    """Model with L2 regularization"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(20,),
                            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
        tf.keras.layers.Dense(256, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
        tf.keras.layers.Dense(128, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

# Train L2 model
model_l2 = create_l2_regularized_model(0.01)
history_l2 = model_l2.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=0
)
```

#### Technique 2: Dropout
```python
def create_dropout_model(dropout_rate=0.3):
    """Model with Dropout layers"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(20,)),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

# Train Dropout model
model_dropout = create_dropout_model(0.3)
history_dropout = model_dropout.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=0
)
```

#### Technique 3: Early Stopping
```python
def create_early_stopping_model():
    """Model with early stopping callback"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(20,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Train with early stopping
model_early = create_early_stopping_model()
history_early = model_early.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=0
)

print(f"Training stopped at epoch: {len(history_early.history['loss'])}")
```

### Part 4: Comparison and Best Practices (7:00 AM - 7:15 AM)
**15 minutes**

#### Compare All Techniques
```python
def compare_models(models_dict, X_test, y_test):
    """Compare performance of different models"""
    results = {}

    for name, model in models_dict.items():
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)

        results[name] = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'generalization_gap': test_loss - train_loss
        }

    # Display results
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)

    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Training Accuracy: {metrics['train_acc']:.4f}")
        print(f"  Test Accuracy: {metrics['test_acc']:.4f}")
        print(f"  Generalization Gap: {metrics['generalization_gap']:.4f}")

    return results

# Compare all models
models = {
    'Overfitting Model': model_overfit,
    'L2 Regularized': model_l2,
    'Dropout Model': model_dropout,
    'Early Stopping': model_early
}

results = compare_models(models, X_test, y_test)
```

#### Best Practices Summary
```python
# Create the optimal model combining techniques
def create_optimal_model():
    """Combine multiple regularization techniques"""
    model = tf.keras.Sequential([
        # Input layer with moderate size
        tf.keras.layers.Dense(128, activation='relu', input_shape=(20,),
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        # Hidden layers with decreasing size
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),

        # Output layer
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# Key Takeaways
print("""
REGULARIZATION BEST PRACTICES:
1. Start with a simpler model architecture
2. Use dropout (0.2-0.5) after dense layers
3. Apply L2 regularization (0.001-0.01) to kernel weights
4. Implement early stopping with patience=10-20
5. Use batch normalization for deep networks
6. Monitor validation metrics during training
7. Combine multiple techniques for best results
""")
```

### Part 5: Quick Exercise & Assignment (7:15 AM - 7:20 AM)
**5 minutes**

#### Student Exercise
```python
# TODO: Students implement this function
def diagnose_overfitting(history):
    """
    Analyze training history and return diagnosis
    Returns: 'overfitting', 'underfitting', or 'good fit'
    """
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]

    # Calculate average losses for last 10 epochs
    avg_train_loss = np.mean(history.history['loss'][-10:])
    avg_val_loss = np.mean(history.history['val_loss'][-10:])

    gap = avg_val_loss - avg_train_loss

    if gap > 0.1:
        return "Overfitting detected!"
    elif avg_train_loss > 0.5:
        return "Underfitting detected!"
    else:
        return "Good fit!"

# Test the function
diagnosis = diagnose_overfitting(history_dropout)
print(f"Model diagnosis: {diagnosis}")
```

## Assignment for T6 Implementation
1. Implement gradient descent optimization with momentum
2. Add regularization to your neural network from T5
3. Create visualization of loss landscape
4. Compare SGD, Adam, and RMSprop optimizers
5. Implement learning rate scheduling

## Quick Reference Card
```python
# Regularization Techniques Cheat Sheet

# L2 Regularization
kernel_regularizer=tf.keras.regularizers.l2(0.01)

# L1 Regularization
kernel_regularizer=tf.keras.regularizers.l1(0.01)

# Dropout
tf.keras.layers.Dropout(rate=0.3)

# Early Stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Batch Normalization
tf.keras.layers.BatchNormalization()

# Learning Rate Reduction
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001
)
```

## Resources for Further Study
- Deep Learning Book Chapter 7: Regularization
- TensorFlow Regularization Tutorial
- Dropout Paper (Srivastava et al., 2014)
- Batch Normalization Paper (Ioffe & Szegedy, 2015)