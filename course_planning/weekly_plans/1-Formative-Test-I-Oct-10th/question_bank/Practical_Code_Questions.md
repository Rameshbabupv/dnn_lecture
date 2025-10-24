# Practical Code Questions for Deep Neural Networks
**Course**: 21CSE558T - Deep Neural Network Architectures
**Focus**: TensorFlow/Keras Implementation Questions
**Coverage**: Modules 1-2 Practical Applications

---

# SECTION 1: TENSORFLOW BASICS (Module 1)

## Q1. Tensor Creation and Manipulation (2 marks)
**PLO**: PLO-2 | **CO**: CO-1 | **BL**: BL3 | **PI**: CO1-PI2 | **Difficulty**: Easy

Write TensorFlow code to:
1. Create a 3×4 matrix filled with random values from normal distribution
2. Convert it to a 1D tensor
3. Reshape it back to 2×6 matrix

**Expected Solution**:
```python
import tensorflow as tf

# Create 3x4 matrix with random values
matrix_3x4 = tf.random.normal(shape=(3, 4), mean=0, stddev=1)
print("Original 3x4 matrix:")
print(matrix_3x4)

# Convert to 1D tensor
tensor_1d = tf.reshape(matrix_3x4, [-1])
print("\n1D tensor:")
print(tensor_1d)

# Reshape to 2x6 matrix
matrix_2x6 = tf.reshape(tensor_1d, (2, 6))
print("\n2x6 matrix:")
print(matrix_2x6)
```

---

## Q2. Basic Neural Network Implementation (5 marks)
**PLO**: PLO-2 | **CO**: CO-1 | **BL**: BL6 | **PI**: CO1-PI2 | **Difficulty**: Moderate

Create a simple neural network using TensorFlow/Keras for binary classification with the following specifications:
- Input layer: 4 features
- Hidden layer: 8 neurons with ReLU activation
- Output layer: 1 neuron with sigmoid activation
- Compile with Adam optimizer and binary crossentropy loss

**Expected Solution**:
```python
import tensorflow as tf
from tensorflow import keras

# Create the model
model = keras.Sequential([
    # Input layer (implicitly defined by first hidden layer)
    keras.layers.Dense(
        units=8,
        activation='relu',
        input_shape=(4,),
        name='hidden_layer'
    ),

    # Output layer
    keras.layers.Dense(
        units=1,
        activation='sigmoid',
        name='output_layer'
    )
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

# Display model summary
print("Model Architecture:")
model.summary()

# Display model configuration
print("\nModel Configuration:")
print(f"Optimizer: {model.optimizer.__class__.__name__}")
print(f"Loss Function: {model.loss}")
print(f"Metrics: {model.metrics_names}")
```

---

## Q3. Activation Function Comparison (3 marks)
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL4 | **PI**: CO1-PI3 | **Difficulty**: Moderate

Write code to compare three activation functions (sigmoid, tanh, ReLU) by plotting their outputs for input range [-5, 5].

**Expected Solution**:
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate input values
x = np.linspace(-5, 5, 100)
x_tensor = tf.constant(x, dtype=tf.float32)

# Apply different activation functions
sigmoid_output = tf.nn.sigmoid(x_tensor)
tanh_output = tf.nn.tanh(x_tensor)
relu_output = tf.nn.relu(x_tensor)

# Plotting
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x, sigmoid_output.numpy())
plt.title('Sigmoid Activation')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x, tanh_output.numpy())
plt.title('Tanh Activation')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x, relu_output.numpy())
plt.title('ReLU Activation')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)

plt.tight_layout()
plt.show()

# Print value ranges
print(f"Sigmoid range: [{tf.reduce_min(sigmoid_output):.3f}, {tf.reduce_max(sigmoid_output):.3f}]")
print(f"Tanh range: [{tf.reduce_min(tanh_output):.3f}, {tf.reduce_max(tanh_output):.3f}]")
print(f"ReLU range: [{tf.reduce_min(relu_output):.3f}, {tf.reduce_max(relu_output):.3f}]")
```

---

# SECTION 2: OPTIMIZATION & TRAINING (Module 2)

## Q4. Gradient Descent Implementation (5 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL6 | **PI**: CO2-PI1 | **Difficulty**: Difficult

Implement gradient descent from scratch using TensorFlow to minimize the function f(x) = x² + 2x + 1. Include learning rate scheduling.

**Expected Solution**:
```python
import tensorflow as tf
import matplotlib.pyplot as plt

# Define the function f(x) = x^2 + 2x + 1
def objective_function(x):
    return x**2 + 2*x + 1

# Define the gradient (derivative) f'(x) = 2x + 2
def gradient_function(x):
    return 2*x + 2

# Initialize parameters
x = tf.Variable(5.0, dtype=tf.float32)  # Starting point
learning_rate = 0.1
num_iterations = 50

# Storage for plotting
x_history = []
loss_history = []

print("Iteration | x | Loss | Gradient")
print("-" * 40)

# Gradient descent loop
for i in range(num_iterations):
    with tf.GradientTape() as tape:
        loss = objective_function(x)

    # Compute gradient
    gradient = tape.gradient(loss, x)

    # Update x
    x.assign_sub(learning_rate * gradient)

    # Store history
    x_history.append(x.numpy())
    loss_history.append(loss.numpy())

    # Print progress every 10 iterations
    if i % 10 == 0:
        print(f"{i:9d} | {x.numpy():5.3f} | {loss.numpy():8.3f} | {gradient.numpy():8.3f}")

print(f"\nFinal result:")
print(f"Optimal x: {x.numpy():.6f}")
print(f"Minimum value: {objective_function(x).numpy():.6f}")
print(f"Expected optimal x: -1.0")
print(f"Expected minimum: 0.0")

# Plot convergence
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(x_history)
plt.title('Parameter Convergence')
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.axhline(y=-1, color='r', linestyle='--', label='Optimal x')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.title('Loss Convergence')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.axhline(y=0, color='r', linestyle='--', label='Minimum loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## Q5. Overfitting Detection and Prevention (5 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL6 | **PI**: CO2-PI3 | **Difficulty**: Moderate

Create a neural network that demonstrates overfitting and then apply regularization techniques to prevent it.

**Expected Solution**:
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic dataset prone to overfitting
np.random.seed(42)
tf.random.set_seed(42)

# Small dataset to encourage overfitting
X_train = np.random.randn(100, 10).astype(np.float32)
y_train = (np.sum(X_train[:, :3], axis=1) > 0).astype(np.float32)

X_val = np.random.randn(50, 10).astype(np.float32)
y_val = (np.sum(X_val[:, :3], axis=1) > 0).astype(np.float32)

# Model without regularization (prone to overfitting)
def create_overfitting_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(50, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Model with regularization
def create_regularized_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(50, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dropout(0.3),  # Dropout regularization
        tf.keras.layers.Dense(50, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # L2 regularization
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train both models
print("Training model without regularization...")
model_overfit = create_overfitting_model()
history_overfit = model_overfit.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    verbose=0
)

print("Training model with regularization...")
model_regularized = create_regularized_model()
history_regularized = model_regularized.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    verbose=0
)

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss plots
axes[0, 0].plot(history_overfit.history['loss'], label='Training Loss')
axes[0, 0].plot(history_overfit.history['val_loss'], label='Validation Loss')
axes[0, 0].set_title('Without Regularization - Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(history_regularized.history['loss'], label='Training Loss')
axes[0, 1].plot(history_regularized.history['val_loss'], label='Validation Loss')
axes[0, 1].set_title('With Regularization - Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Accuracy plots
axes[1, 0].plot(history_overfit.history['accuracy'], label='Training Accuracy')
axes[1, 0].plot(history_overfit.history['val_accuracy'], label='Validation Accuracy')
axes[1, 0].set_title('Without Regularization - Accuracy')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(history_regularized.history['accuracy'], label='Training Accuracy')
axes[1, 1].plot(history_regularized.history['val_accuracy'], label='Validation Accuracy')
axes[1, 1].set_title('With Regularization - Accuracy')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Final performance comparison
print("\nFinal Performance Comparison:")
print("=" * 50)
overfit_train_acc = history_overfit.history['accuracy'][-1]
overfit_val_acc = history_overfit.history['val_accuracy'][-1]
reg_train_acc = history_regularized.history['accuracy'][-1]
reg_val_acc = history_regularized.history['val_accuracy'][-1]

print(f"Without Regularization:")
print(f"  Training Accuracy: {overfit_train_acc:.3f}")
print(f"  Validation Accuracy: {overfit_val_acc:.3f}")
print(f"  Gap: {overfit_train_acc - overfit_val_acc:.3f}")

print(f"\nWith Regularization:")
print(f"  Training Accuracy: {reg_train_acc:.3f}")
print(f"  Validation Accuracy: {reg_val_acc:.3f}")
print(f"  Gap: {reg_train_acc - reg_val_acc:.3f}")
```

---

## Q6. Custom Optimizer Implementation (5 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL6 | **PI**: CO2-PI1 | **Difficulty**: Difficult

Implement a simple version of the momentum optimizer from scratch using TensorFlow.

**Expected Solution**:
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class MomentumOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}

    def apply_gradients(self, gradients, variables):
        for grad, var in zip(gradients, variables):
            if var.name not in self.velocities:
                self.velocities[var.name] = tf.Variable(
                    tf.zeros_like(var), trainable=False
                )

            # Momentum update: v = β*v + (1-β)*grad
            velocity = self.velocities[var.name]
            velocity.assign(self.momentum * velocity + (1 - self.momentum) * grad)

            # Parameter update: θ = θ - α*v
            var.assign_sub(self.learning_rate * velocity)

# Test function: f(x, y) = x^2 + y^2 (minimum at (0, 0))
def objective_function(x, y):
    return x**2 + y**2

# Initialize parameters
x = tf.Variable(3.0, dtype=tf.float32)
y = tf.Variable(4.0, dtype=tf.float32)

# Initialize optimizers
custom_momentum = MomentumOptimizer(learning_rate=0.1, momentum=0.9)
builtin_momentum = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

# Storage for comparison
custom_path = []
builtin_path = []

# Parameters for builtin optimizer (need to reset)
x_builtin = tf.Variable(3.0, dtype=tf.float32)
y_builtin = tf.Variable(4.0, dtype=tf.float32)

print("Comparing Custom vs Built-in Momentum Optimizer")
print("=" * 50)
print("Iteration | Custom (x, y) | Built-in (x, y) | Custom Loss | Built-in Loss")
print("-" * 80)

# Training loop
for i in range(50):
    # Custom momentum optimizer
    with tf.GradientTape() as tape:
        loss_custom = objective_function(x, y)
    gradients_custom = tape.gradient(loss_custom, [x, y])
    custom_momentum.apply_gradients(gradients_custom, [x, y])
    custom_path.append([x.numpy(), y.numpy()])

    # Built-in momentum optimizer
    with tf.GradientTape() as tape:
        loss_builtin = objective_function(x_builtin, y_builtin)
    gradients_builtin = tape.gradient(loss_builtin, [x_builtin, y_builtin])
    builtin_momentum.apply_gradients(zip(gradients_builtin, [x_builtin, y_builtin]))
    builtin_path.append([x_builtin.numpy(), y_builtin.numpy()])

    # Print progress every 10 iterations
    if i % 10 == 0:
        print(f"{i:9d} | ({x.numpy():5.2f}, {y.numpy():5.2f}) | "
              f"({x_builtin.numpy():5.2f}, {y_builtin.numpy():5.2f}) | "
              f"{loss_custom.numpy():8.3f} | {loss_builtin.numpy():8.3f}")

# Convert to numpy arrays for plotting
custom_path = np.array(custom_path)
builtin_path = np.array(builtin_path)

# Plot optimization paths
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(custom_path[:, 0], custom_path[:, 1], 'b-o', markersize=3, label='Custom Momentum')
plt.plot(builtin_path[:, 0], builtin_path[:, 1], 'r-s', markersize=3, label='Built-in Momentum')
plt.plot(0, 0, 'g*', markersize=15, label='Optimum (0, 0)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Optimization Paths Comparison')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
custom_losses = [objective_function(point[0], point[1]).numpy() for point in custom_path]
builtin_losses = [objective_function(point[0], point[1]).numpy() for point in builtin_path]

plt.plot(custom_losses, 'b-', label='Custom Momentum')
plt.plot(builtin_losses, 'r-', label='Built-in Momentum')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Convergence Comparison')
plt.legend()
plt.grid(True)
plt.yscale('log')

plt.tight_layout()
plt.show()

print(f"\nFinal Results:")
print(f"Custom Momentum: ({x.numpy():.6f}, {y.numpy():.6f})")
print(f"Built-in Momentum: ({x_builtin.numpy():.6f}, {y_builtin.numpy():.6f})")
print(f"Target: (0.0, 0.0)")
```

---

# SECTION 3: DEBUGGING & DIAGNOSTICS

## Q7. Gradient Monitoring and Debugging (3 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI2 | **Difficulty**: Moderate

Write code to monitor gradients during training and detect vanishing/exploding gradient problems.

**Expected Solution**:
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Create a deep network prone to gradient problems
def create_deep_network(activation='sigmoid'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(50, activation=activation, input_shape=(10,)))

    # Add many hidden layers
    for i in range(8):
        model.add(tf.keras.layers.Dense(50, activation=activation))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

# Generate sample data
X_train = np.random.randn(1000, 10).astype(np.float32)
y_train = np.random.randint(0, 2, (1000, 1)).astype(np.float32)

# Create models with different activations
model_sigmoid = create_deep_network('sigmoid')
model_relu = create_deep_network('relu')

models = [
    ('Sigmoid Network', model_sigmoid),
    ('ReLU Network', model_relu)
]

# Function to monitor gradients
def monitor_gradients(model, X, y, epochs=5):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    gradient_norms = []
    layer_names = []

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            predictions = model(X, training=True)
            loss = tf.keras.losses.binary_crossentropy(y, predictions)
            loss = tf.reduce_mean(loss)

        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)

        # Calculate gradient norms for each layer
        epoch_norms = []
        for i, (grad, var) in enumerate(zip(gradients, model.trainable_variables)):
            if grad is not None:
                grad_norm = tf.norm(grad).numpy()
                epoch_norms.append(grad_norm)

                if epoch == 0:  # Store layer names only once
                    layer_names.append(f"Layer_{i//2}")  # Each layer has weights and biases

        gradient_norms.append(epoch_norms)

        # Apply gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return np.array(gradient_norms), layer_names

# Monitor gradients for both models
plt.figure(figsize=(15, 10))

for idx, (name, model) in enumerate(models):
    print(f"Monitoring gradients for {name}...")

    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # Monitor gradients
    grad_norms, layer_names = monitor_gradients(model, X_train[:100], y_train[:100])

    # Plot gradient norms
    plt.subplot(2, 2, idx*2 + 1)
    for i in range(0, len(layer_names), 2):  # Only plot weight gradients (skip biases)
        plt.plot(grad_norms[:, i], label=f'Layer {i//2}')

    plt.title(f'{name} - Gradient Norms Over Training')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)

    # Plot gradient distribution by layer
    plt.subplot(2, 2, idx*2 + 2)
    final_grad_norms = grad_norms[-1][::2]  # Last epoch, weights only
    layer_indices = range(len(final_grad_norms))

    plt.bar(layer_indices, final_grad_norms)
    plt.title(f'{name} - Final Gradient Norms by Layer')
    plt.xlabel('Layer Index')
    plt.ylabel('Gradient Norm')
    plt.yscale('log')
    plt.grid(True)

    # Diagnosis
    print(f"\n{name} Gradient Analysis:")
    print(f"  Average gradient norm: {np.mean(final_grad_norms):.6f}")
    print(f"  Min gradient norm: {np.min(final_grad_norms):.6f}")
    print(f"  Max gradient norm: {np.max(final_grad_norms):.6f}")
    print(f"  Gradient range: {np.max(final_grad_norms) / np.min(final_grad_norms):.2f}")

    if np.min(final_grad_norms) < 1e-7:
        print("  ⚠️  WARNING: Vanishing gradients detected!")
    elif np.max(final_grad_norms) > 10:
        print("  ⚠️  WARNING: Exploding gradients detected!")
    else:
        print("  ✅ Gradients appear healthy")

plt.tight_layout()
plt.show()
```

---

# SECTION 4: COMPREHENSIVE IMPLEMENTATION

## Q8. Complete Neural Network Pipeline (8 marks)
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL6 | **PI**: CO2-PI1 | **Difficulty**: Difficult

Implement a complete neural network training pipeline with data preprocessing, model creation, training with callbacks, and evaluation.

**Expected Solution**:
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# 1. Data Generation and Preprocessing
print("1. Generating and preprocessing data...")

# Generate synthetic dataset
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Data shapes: Train {X_train_scaled.shape}, Val {X_val_scaled.shape}, Test {X_test_scaled.shape}")

# 2. Model Architecture
print("\n2. Creating model architecture...")

def create_model(input_dim, dropout_rate=0.3, l2_reg=0.01):
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Dense(
            64,
            activation='relu',
            input_shape=(input_dim,),
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            name='dense_1'
        ),
        tf.keras.layers.BatchNormalization(name='batch_norm_1'),
        tf.keras.layers.Dropout(dropout_rate, name='dropout_1'),

        # Hidden layer 2
        tf.keras.layers.Dense(
            32,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            name='dense_2'
        ),
        tf.keras.layers.BatchNormalization(name='batch_norm_2'),
        tf.keras.layers.Dropout(dropout_rate, name='dropout_2'),

        # Hidden layer 3
        tf.keras.layers.Dense(
            16,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            name='dense_3'
        ),
        tf.keras.layers.Dropout(dropout_rate, name='dropout_3'),

        # Output layer
        tf.keras.layers.Dense(1, activation='sigmoid', name='output')
    ])

    return model

model = create_model(X_train_scaled.shape[1])

# 3. Model Compilation
print("\n3. Compiling model...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

model.summary()

# 4. Callbacks Setup
print("\n4. Setting up callbacks...")

callbacks = [
    # Early stopping
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),

    # Learning rate reduction
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1
    ),

    # Model checkpointing
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# 5. Model Training
print("\n5. Training model...")

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# 6. Training Visualization
print("\n6. Visualizing training progress...")

def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

plot_training_history(history)

# 7. Model Evaluation
print("\n7. Evaluating model...")

# Load best model
model.load_weights('best_model.h5')

# Evaluate on test set
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
    X_test_scaled, y_test, verbose=0
)

# Calculate F1 score
test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)

print(f"\nFinal Test Results:")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy:.4f}")
print(f"  Test Precision: {test_precision:.4f}")
print(f"  Test Recall: {test_recall:.4f}")
print(f"  Test F1 Score: {test_f1:.4f}")

# 8. Prediction Analysis
print("\n8. Analyzing predictions...")

# Get predictions
y_pred_proba = model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int)

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot prediction distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(y_pred_proba[y_test == 0], alpha=0.7, label='Class 0', bins=20)
plt.hist(y_pred_proba[y_test == 1], alpha=0.7, label='Class 1', bins=20)
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Prediction Probability Distribution')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(range(len(y_test)), y_pred_proba,
           c=y_test, cmap='viridis', alpha=0.6)
plt.axhline(y=0.5, color='red', linestyle='--', label='Decision Threshold')
plt.xlabel('Sample Index')
plt.ylabel('Predicted Probability')
plt.title('Predictions vs True Labels')
plt.colorbar(label='True Label')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n9. Pipeline completed successfully! ✅")
```

---

# ASSESSMENT RUBRIC FOR PRACTICAL QUESTIONS

## Coding Standards (20% of marks):
- **Proper imports and setup** (4 points)
- **Code organization and comments** (4 points)
- **Variable naming and readability** (4 points)
- **Error handling and edge cases** (4 points)
- **Output formatting and presentation** (4 points)

## Technical Implementation (60% of marks):
- **Correct TensorFlow/Keras usage** (15 points)
- **Proper model architecture** (15 points)
- **Appropriate optimization techniques** (15 points)
- **Accurate mathematical implementation** (15 points)

## Analysis and Interpretation (20% of marks):
- **Results analysis** (5 points)
- **Performance interpretation** (5 points)
- **Problem diagnosis** (5 points)
- **Recommendations and insights** (5 points)

## Bonus Points (Up to 10% extra):
- **Creative visualization** (2 points)
- **Advanced techniques** (3 points)
- **Comprehensive testing** (3 points)
- **Documentation quality** (2 points)

---

# PRACTICAL SKILLS ASSESSED

## Technical Competencies:
1. ✅ **TensorFlow/Keras proficiency**
2. ✅ **Neural network implementation**
3. ✅ **Optimization algorithm understanding**
4. ✅ **Debugging and diagnostics**
5. ✅ **Performance analysis**
6. ✅ **Code optimization**

## Problem-Solving Skills:
1. ✅ **Algorithm design**
2. ✅ **Parameter tuning**
3. ✅ **Error diagnosis**
4. ✅ **Performance optimization**
5. ✅ **Experimental design**

These practical questions ensure students can implement theoretical concepts in real-world scenarios using industry-standard tools and frameworks.