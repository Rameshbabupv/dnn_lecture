"""
Tutorial T10: Building CNN for Fashion-MNIST Classification - SOLUTION
Week 10, Day 4 - October 29, 2025
Deep Neural Network Architectures (21CSE558T)

This is the COMPLETE SOLUTION for Tutorial T10.
Students should attempt tutorial_t10_starter.py first.

Expected Results:
- Training Accuracy: ~95% (after 10 epochs)
- Validation Accuracy: ~90-92%
- Test Accuracy: ~90%
- Training Time: ~2-3 minutes (CPU), ~30 seconds (GPU)

Author: Course Instructor
Last Updated: October 2025
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

print("="*60)
print("TUTORIAL T10: CNN FOR FASHION-MNIST CLASSIFICATION")
print("="*60)
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print("="*60 + "\n")


# ============================================================================
# PART 1: LOAD FASHION-MNIST DATASET
# ============================================================================

print("PART 1: Loading Fashion-MNIST Dataset...")
print("-" * 60)

# Load the dataset (automatically downloads if not present)
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Define class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")
print(f"Number of classes: {len(class_names)}")
print(f"Pixel value range: [{X_train.min()}, {X_train.max()}]")

# Visualize sample images
plt.figure(figsize=(12, 3))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(class_names[y_train[i]], fontsize=9)
    plt.axis('off')
plt.suptitle('Fashion-MNIST Sample Images (10 Classes)', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('output/sample_images.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Sample images saved to output/sample_images.png\n")


# ============================================================================
# PART 2: DATA PREPROCESSING
# ============================================================================

print("PART 2: Preprocessing Data...")
print("-" * 60)

# Normalize pixel values to [0, 1] range
# This helps with training stability and convergence
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print(f"After normalization: [{X_train.min():.2f}, {X_train.max():.2f}]")

# Reshape data for CNN input
# CNN expects 4D input: (samples, height, width, channels)
# Add channel dimension for grayscale images (channels = 1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print(f"Reshaped X_train: {X_train.shape}")
print(f"Reshaped X_test: {X_test.shape}")

# Convert labels to one-hot encoding
# Example: 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train_categorical = keras.utils.to_categorical(y_train, 10)
y_test_categorical = keras.utils.to_categorical(y_test, 10)

print(f"One-hot encoded y_train: {y_train_categorical.shape}")
print(f"One-hot encoded y_test: {y_test_categorical.shape}")
print(f"Example: Label {y_train[0]} → {y_train_categorical[0]}")
print("✓ Preprocessing completed\n")


# ============================================================================
# PART 3: BUILD CNN ARCHITECTURE
# ============================================================================

print("PART 3: Building CNN Architecture...")
print("-" * 60)

# Create Sequential model
model = keras.Sequential([
    # First convolutional block
    # Conv2D: Applies 32 filters of size 3x3 to input
    # ReLU: Introduces non-linearity
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'),
    # MaxPooling2D: Reduces spatial dimensions by half
    layers.MaxPooling2D((2, 2), name='pool1'),

    # Second convolutional block
    # More filters (64) to learn more complex features
    layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
    layers.MaxPooling2D((2, 2), name='pool2'),

    # Flatten and fully connected layers
    # Flatten: Converts 2D feature maps to 1D vector
    layers.Flatten(name='flatten'),
    # Dense: Fully connected layer for classification
    layers.Dense(64, activation='relu', name='dense1'),
    # Output: 10 units with softmax for multi-class classification
    layers.Dense(10, activation='softmax', name='output')
], name='Fashion_MNIST_CNN')

# Display model architecture
print("\nModel Architecture:")
model.summary()

# Calculate output dimensions manually
print("\n" + "="*60)
print("OUTPUT DIMENSION CALCULATIONS:")
print("="*60)
print("Input: 28x28x1 (height × width × channels)")
print("\nLayer 1 - Conv1 (32 filters, 3x3, stride=1, padding='valid'):")
print("  Formula: (W - F + 2P) / S + 1")
print("  Output = (28 - 3 + 0) / 1 + 1 = 26x26x32")
print("  Parameters: (3×3×1 + 1) × 32 = 320")
print("\nLayer 2 - Pool1 (2x2, stride=2):")
print("  Formula: W / S")
print("  Output = 26 / 2 = 13x13x32")
print("  Parameters: 0 (no trainable parameters)")
print("\nLayer 3 - Conv2 (64 filters, 3x3, stride=1, padding='valid'):")
print("  Formula: (W - F + 2P) / S + 1")
print("  Output = (13 - 3 + 0) / 1 + 1 = 11x11x64")
print("  Parameters: (3×3×32 + 1) × 64 = 18,496")
print("\nLayer 4 - Pool2 (2x2, stride=2):")
print("  Formula: W / S")
print("  Output = 11 / 2 = 5.5 → 5x5x64 (floor division)")
print("  Parameters: 0")
print("\nLayer 5 - Flatten:")
print("  Output = 5×5×64 = 1,600 features")
print("\nLayer 6 - Dense1 (64 units):")
print("  Parameters: (1,600 + 1) × 64 = 102,464")
print("\nLayer 7 - Output (10 units):")
print("  Parameters: (64 + 1) × 10 = 650")
print("\nTotal Trainable Parameters: 121,930")
print("="*60 + "\n")


# ============================================================================
# PART 4: COMPILE MODEL
# ============================================================================

print("PART 4: Compiling Model...")
print("-" * 60)

model.compile(
    optimizer='adam',  # Adaptive learning rate optimizer
    loss='categorical_crossentropy',  # For multi-class classification
    metrics=['accuracy']  # Track accuracy during training
)

print("✓ Model compiled successfully!")
print("  Optimizer: Adam")
print("  Loss: Categorical Crossentropy")
print("  Metrics: Accuracy\n")


# ============================================================================
# PART 5: TRAIN MODEL
# ============================================================================

print("PART 5: Training Model...")
print("-" * 60)
print("Training for 10 epochs with 20% validation split")
print("(This will take 2-3 minutes on CPU, ~30 seconds on GPU)\n")

# Train the model
history = model.fit(
    X_train, y_train_categorical,
    epochs=10,
    batch_size=128,
    validation_split=0.2,  # Use 20% of training data for validation
    verbose=1  # Show progress bar
)

print("\n✓ Training completed!\n")


# ============================================================================
# PART 6: PLOT TRAINING HISTORY
# ============================================================================

print("PART 6: Visualizing Training History...")
print("-" * 60)

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot accuracy
axes[0].plot(history.history['accuracy'], label='Training Accuracy',
             marker='o', linewidth=2, markersize=6)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy',
             marker='s', linewidth=2, markersize=6)
axes[0].set_title('Model Accuracy Over Epochs', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Accuracy', fontsize=11)
axes[0].legend(loc='lower right', fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0.7, 1.0])

# Plot loss
axes[1].plot(history.history['loss'], label='Training Loss',
             marker='o', linewidth=2, markersize=6)
axes[1].plot(history.history['val_loss'], label='Validation Loss',
             marker='s', linewidth=2, markersize=6)
axes[1].set_title('Model Loss Over Epochs', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Loss', fontsize=11)
axes[1].legend(loc='upper right', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/training_history.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Training history plots saved to output/training_history.png\n")


# ============================================================================
# PART 7: EVALUATE ON TEST SET
# ============================================================================

print("PART 7: Evaluating Model on Test Set...")
print("-" * 60)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)

print(f"\n{'='*60}")
print("FINAL TEST RESULTS:")
print(f"{'='*60}")
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"{'='*60}\n")

# Compare with validation accuracy
final_val_accuracy = history.history['val_accuracy'][-1]
print(f"Final Validation Accuracy: {final_val_accuracy:.4f} ({final_val_accuracy*100:.2f}%)")
print(f"Difference (Test - Val):    {(test_accuracy - final_val_accuracy):.4f}")
if abs(test_accuracy - final_val_accuracy) < 0.02:
    print("✓ Good generalization (test ≈ validation)")
else:
    print("⚠ Possible overfitting or distribution shift\n")


# ============================================================================
# PART 8: VISUALIZE PREDICTIONS
# ============================================================================

print("\nPART 8: Visualizing Model Predictions...")
print("-" * 60)

# Make predictions on first 10 test images
predictions = model.predict(X_test[:10], verbose=0)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = y_test[:10]

# Visualize predictions with confidence scores
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for i in range(10):
    axes[i].imshow(X_test[i].reshape(28, 28), cmap='gray')

    pred_class = predicted_classes[i]
    true_class = true_classes[i]
    confidence = predictions[i][pred_class] * 100

    # Color code: green if correct, red if wrong
    color = 'green' if pred_class == true_class else 'red'

    axes[i].set_title(
        f'Pred: {class_names[pred_class]} ({confidence:.1f}%)\n'
        f'True: {class_names[true_class]}',
        color=color, fontsize=9
    )
    axes[i].axis('off')

plt.suptitle('Model Predictions on Test Set (Green=Correct, Red=Wrong)',
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('output/predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Prediction visualizations saved to output/predictions.png\n")


# ============================================================================
# PART 9: VISUALIZE LEARNED FILTERS
# ============================================================================

print("PART 9: Visualizing Learned Filters...")
print("-" * 60)

# Extract weights from first convolutional layer
filters, biases = model.layers[0].get_weights()
print(f"First Conv Layer Filters Shape: {filters.shape}")
print(f"(kernel_height, kernel_width, input_channels, output_filters)")
print(f"Number of filters: {filters.shape[-1]}")
print(f"Kernel size: {filters.shape[0]}×{filters.shape[1]}")

# Normalize filters to [0, 1] for visualization
f_min, f_max = filters.min(), filters.max()
filters_normalized = (filters - f_min) / (f_max - f_min)

# Visualize first 16 filters (4x4 grid)
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.flatten()

for i in range(16):
    # Extract filter for i-th output channel
    filter_img = filters_normalized[:, :, 0, i]
    axes[i].imshow(filter_img, cmap='viridis')
    axes[i].set_title(f'Filter {i+1}', fontsize=9)
    axes[i].axis('off')

plt.suptitle('Learned 3×3 Filters in First Conv Layer (Conv1)',
             fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('output/learned_filters.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Learned filters saved to output/learned_filters.png\n")

print("FILTER INTERPRETATION:")
print("- Edge detectors (horizontal, vertical, diagonal)")
print("- Texture detectors (patterns, gradients)")
print("- These are learned automatically from data!")
print("- Different from hand-crafted filters (Sobel, Gabor, etc.)\n")


# ============================================================================
# PART 10: VISUALIZE FEATURE MAPS
# ============================================================================

print("PART 10: Visualizing Feature Maps...")
print("-" * 60)

# Create a model that outputs intermediate layer activations
layer_names = ['conv1', 'pool1', 'conv2', 'pool2']
layer_outputs = [model.get_layer(name).output for name in layer_names]
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)

# Get feature maps for a sample image (first test image)
sample_idx = 0
sample_image = X_test[sample_idx:sample_idx+1]
sample_label = class_names[y_test[sample_idx]]

print(f"Analyzing feature maps for: {sample_label}")
activations = activation_model.predict(sample_image, verbose=0)

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Show original image
axes[0, 0].imshow(sample_image[0, :, :, 0], cmap='gray')
axes[0, 0].set_title(f'Original Image\n({sample_label})', fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

# Visualize feature maps for each layer
positions = [(0, 1), (0, 2), (1, 1), (1, 2)]
for idx, (layer_name, activation) in enumerate(zip(layer_names, activations)):
    row, col = positions[idx]

    n_features = activation.shape[-1]
    size_h, size_w = activation.shape[1], activation.shape[2]

    # Create grid to display multiple feature maps
    n_cols = 8
    n_rows = min(4, n_features // n_cols)  # Show up to 32 feature maps
    display_grid = np.zeros((size_h * n_rows, size_w * n_cols))

    # Fill the grid with feature maps
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            channel_idx = row_idx * n_cols + col_idx
            if channel_idx < n_features:
                channel_image = activation[0, :, :, channel_idx]
                # Normalize for display
                channel_image -= channel_image.mean()
                if channel_image.std() > 0:
                    channel_image /= channel_image.std()
                channel_image = np.clip(channel_image, 0, 1)
                display_grid[row_idx * size_h:(row_idx + 1) * size_h,
                             col_idx * size_w:(col_idx + 1) * size_w] = channel_image

    axes[row, col].imshow(display_grid, cmap='viridis')
    axes[row, col].set_title(f'{layer_name}\nShape: {activation.shape[1:]}',
                             fontsize=10, fontweight='bold')
    axes[row, col].axis('off')

# Hide unused subplot
axes[1, 0].axis('off')

plt.suptitle('Feature Maps at Different CNN Layers (Hierarchical Learning)',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('output/feature_maps.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Feature maps saved to output/feature_maps.png\n")

print("FEATURE MAP INTERPRETATION:")
print("- Conv1: Simple features (edges, textures, basic patterns)")
print("- Pool1: Downsampled version of Conv1 (reduced resolution)")
print("- Conv2: Complex features (shapes, object parts)")
print("- Pool2: Downsampled version of Conv2")
print("- Each layer builds on features from previous layer")
print("- Hierarchical feature learning in action!\n")


# ============================================================================
# PART 11: COMPARISON WITH MLP (BONUS)
# ============================================================================

print("PART 11: Comparing CNN vs MLP...")
print("-" * 60)

# Build MLP with similar number of parameters
mlp_model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28, 1), name='flatten_input'),
    layers.Dense(128, activation='relu', name='dense1'),
    layers.Dense(64, activation='relu', name='dense2'),
    layers.Dense(10, activation='softmax', name='output')
], name='Fashion_MNIST_MLP')

mlp_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nMLP Architecture:")
mlp_model.summary()

print("\nTraining MLP (this will take 1-2 minutes)...")
mlp_history = mlp_model.fit(
    X_train, y_train_categorical,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    verbose=0  # Silent training
)

mlp_test_loss, mlp_test_accuracy = mlp_model.evaluate(
    X_test, y_test_categorical, verbose=0
)

print(f"\n{'='*60}")
print("CNN vs MLP COMPARISON:")
print(f"{'='*60}")
print(f"{'Metric':<30} {'CNN':<15} {'MLP':<15}")
print("-" * 60)
print(f"{'Total Parameters':<30} {model.count_params():<15,} {mlp_model.count_params():<15,}")
print(f"{'Test Accuracy':<30} {test_accuracy:.4f} ({test_accuracy*100:.2f}%)  {mlp_test_accuracy:.4f} ({mlp_test_accuracy*100:.2f}%)")
print(f"{'Test Loss':<30} {test_loss:.4f}          {mlp_test_loss:.4f}")
print(f"{'='*60}\n")

if test_accuracy > mlp_test_accuracy:
    diff = (test_accuracy - mlp_test_accuracy) * 100
    print(f"✓ CNN outperforms MLP by {diff:.2f}% with fewer parameters!")
else:
    print("⚠ MLP performed better (unusual - check training)")

print("\nWHY CNNs WIN:")
print("1. Parameter Efficiency: Weight sharing across spatial locations")
print("2. Translation Equivariance: Same features detected anywhere in image")
print("3. Hierarchical Learning: Build complex from simple features")
print("4. Spatial Structure: Preserve 2D relationships in images\n")


# ============================================================================
# PART 12: SAVE MODEL
# ============================================================================

print("PART 12: Saving Model...")
print("-" * 60)

# Save the trained model
model.save('output/fashion_mnist_cnn_model.keras')
print("✓ Model saved to output/fashion_mnist_cnn_model.keras")

# Also save in HDF5 format (for compatibility)
model.save('output/fashion_mnist_cnn_model.h5')
print("✓ Model saved to output/fashion_mnist_cnn_model.h5\n")

print("To load the model later:")
print("  loaded_model = keras.models.load_model('output/fashion_mnist_cnn_model.keras')\n")


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*60)
print("TUTORIAL T10 COMPLETED SUCCESSFULLY!")
print("="*60)
print("✓ Loaded Fashion-MNIST dataset (60,000 train + 10,000 test)")
print("✓ Built CNN with 2 conv blocks and 121,930 parameters")
print("✓ Trained model for 10 epochs (~2-3 minutes)")
print(f"✓ Achieved {test_accuracy*100:.2f}% accuracy on test set")
print("✓ Visualized training history, predictions, filters, feature maps")
print("✓ Compared CNN vs MLP performance")
print("✓ Saved trained model for future use")
print("="*60)

print("\n📚 KEY TAKEAWAYS:")
print("1. CNNs excel at image classification tasks")
print("2. Convolutional layers learn spatial features automatically")
print("3. Pooling layers add translation invariance and reduce dimensions")
print("4. Early layers learn simple features, deeper layers learn complex patterns")
print("5. CNNs achieve high accuracy with fewer parameters than MLPs")
print("6. Fashion-MNIST is harder than MNIST (~90% vs ~98% accuracy)")

print("\n🏠 HOMEWORK ASSIGNMENT (Due: Before Week 11):")
print("Task 1: Manual Convolution Calculation")
print("  - Calculate 2D convolution for 6×6 image, 3×3 kernel")
print("  - Show all intermediate steps")
print("\nTask 2: Architecture Design")
print("  - Design CNN for MNIST digit classification")
print("  - Specify layers, filters, kernel sizes")
print("  - Calculate output dimensions and total parameters")
print("  - Justify design choices")
print("\nTask 3: Code Experimentation")
print("  - Modify this code: add layer, change kernel sizes")
print("  - Compare training time and accuracy")
print("  - Document observations (2-3 paragraphs)")

print("\n📖 NEXT WEEK (Week 11):")
print("- Famous CNN architectures (LeNet, AlexNet, VGG, ResNet)")
print("- Advanced techniques (Dropout, Batch Normalization)")
print("- Designing deeper networks")
print("- Architecture patterns and best practices")

print("\n🎯 UNIT TEST 2 PREPARATION (October 31):")
print("- Review convolution calculations and formulas")
print("- Practice output dimension calculations")
print("- Understand parameter counting in CNNs")
print("- Be ready to design and justify CNN architectures")
print("- Study differences between CNN and MLP")

print("\n📂 OUTPUT FILES GENERATED:")
print("  output/sample_images.png         - Dataset visualization")
print("  output/training_history.png      - Accuracy and loss curves")
print("  output/predictions.png           - Model predictions on test set")
print("  output/learned_filters.png       - First layer filters (3×3)")
print("  output/feature_maps.png          - Intermediate layer activations")
print("  output/fashion_mnist_cnn_model.keras  - Saved model")
print("  output/fashion_mnist_cnn_model.h5     - Saved model (HDF5)")

print("\n" + "="*60)
print("🎓 GREAT WORK! You've built your first CNN!")
print("="*60 + "\n")
