"""
Tutorial T10: Building CNN for Fashion-MNIST Classification
Week 10, Day 4 - October 29, 2025
Deep Neural Network Architectures (21CSE558T)

Learning Objectives:
1. Load and preprocess Fashion-MNIST dataset
2. Build first CNN using Keras Sequential API
3. Train CNN and visualize training history
4. Evaluate model performance
5. Visualize learned filters and feature maps

Duration: 50 minutes
"""

# ============================================================================
# PART 1: SETUP AND IMPORTS (5 minutes)
# ============================================================================

# TODO: Import NumPy for numerical operations
# import numpy as np

# TODO: Import TensorFlow and Keras
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

# TODO: Import Matplotlib for visualization
# import matplotlib.pyplot as plt

# Set random seed for reproducibility
# np.random.seed(42)
# tf.random.set_seed(42)


# ============================================================================
# PART 2: LOAD FASHION-MNIST DATASET (5 minutes)
# ============================================================================

"""
Fashion-MNIST Dataset:
- 70,000 grayscale images (60,000 train + 10,000 test)
- 10 clothing categories
- 28x28 pixels per image
- Pixel values: 0-255 (grayscale intensity)

Classes:
0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat
5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot
"""

# TODO: Load Fashion-MNIST dataset
# (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# TODO: Print dataset shapes
# print("Training data shape:", X_train.shape)
# print("Training labels shape:", y_train.shape)
# print("Test data shape:", X_test.shape)
# print("Test labels shape:", y_test.shape)

# TODO: Visualize sample images (first 10 from training set)
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#
# plt.figure(figsize=(12, 3))
# for i in range(10):
#     plt.subplot(2, 5, i + 1)
#     plt.imshow(X_train[i], cmap='gray')
#     plt.title(class_names[y_train[i]])
#     plt.axis('off')
# plt.tight_layout()
# plt.savefig('output/sample_images.png', dpi=150, bbox_inches='tight')
# plt.show()
# print("✓ Sample images saved to output/sample_images.png")


# ============================================================================
# PART 3: DATA PREPROCESSING (5 minutes)
# ============================================================================

# TODO: Normalize pixel values to [0, 1] range
# Hint: Divide by 255.0 to scale from [0, 255] to [0, 1]
# X_train = X_train / 255.0
# X_test = X_test / 255.0

# TODO: Reshape data for CNN input
# CNN expects shape: (samples, height, width, channels)
# Current shape: (samples, height, width)
# Add channel dimension for grayscale images (channels = 1)
# X_train = X_train.reshape(-1, 28, 28, 1)
# X_test = X_test.reshape(-1, 28, 28, 1)

# TODO: Print preprocessed shapes
# print("\nPreprocessed shapes:")
# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)

# TODO: Convert labels to one-hot encoding
# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_categorical(y_test, 10)
# print("y_train shape:", y_train.shape)
# print("y_test shape:", y_test.shape)


# ============================================================================
# PART 4: BUILD CNN ARCHITECTURE (15 minutes)
# ============================================================================

"""
CNN Architecture Design:
Input: 28x28x1 (grayscale image)
↓
Conv Layer 1: 32 filters, 3x3 kernel → Output: 26x26x32
ReLU activation
MaxPooling: 2x2 → Output: 13x13x32
↓
Conv Layer 2: 64 filters, 3x3 kernel → Output: 11x11x64
ReLU activation
MaxPooling: 2x2 → Output: 5x5x64
↓
Flatten: 5x5x64 = 1600 features
↓
Dense Layer 1: 64 units, ReLU
↓
Output Layer: 10 units (classes), Softmax
"""

# TODO: Create Sequential model
# model = keras.Sequential([
#     # First convolutional block
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'),
#     layers.MaxPooling2D((2, 2), name='pool1'),
#
#     # Second convolutional block
#     layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
#     layers.MaxPooling2D((2, 2), name='pool2'),
#
#     # Flatten and fully connected layers
#     layers.Flatten(name='flatten'),
#     layers.Dense(64, activation='relu', name='dense1'),
#     layers.Dense(10, activation='softmax', name='output')
# ], name='Fashion_MNIST_CNN')

# TODO: Display model summary
# model.summary()

# TODO: Calculate and display output dimensions
# print("\n" + "="*60)
# print("OUTPUT DIMENSION CALCULATIONS:")
# print("="*60)
# print("Input: 28x28x1")
# print("\nConv1 (32 filters, 3x3, stride=1, padding=valid):")
# print("  Output = (28-3)/1 + 1 = 26x26x32")
# print("Pool1 (2x2, stride=2):")
# print("  Output = 26/2 = 13x13x32")
# print("\nConv2 (64 filters, 3x3, stride=1, padding=valid):")
# print("  Output = (13-3)/1 + 1 = 11x11x64")
# print("Pool2 (2x2, stride=2):")
# print("  Output = 11/2 = 5.5 → 5x5x64 (floor)")
# print("\nFlatten: 5x5x64 = 1,600 features")
# print("Dense: 1,600 → 64 → 10")
# print("="*60)


# ============================================================================
# PART 5: COMPILE MODEL (2 minutes)
# ============================================================================

# TODO: Compile the model
# Optimizer: Adam (adaptive learning rate)
# Loss: Categorical crossentropy (multi-class classification)
# Metrics: Accuracy
# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )
# print("\n✓ Model compiled successfully!")


# ============================================================================
# PART 6: TRAIN MODEL (8 minutes)
# ============================================================================

# TODO: Train the model
# Use 20% of training data for validation
# Train for 10 epochs
# Batch size: 128
# print("\nStarting training...")
# print("(This will take 2-3 minutes)\n")

# history = model.fit(
#     X_train, y_train,
#     epochs=10,
#     batch_size=128,
#     validation_split=0.2,
#     verbose=1
# )

# print("\n✓ Training completed!")


# ============================================================================
# PART 7: PLOT TRAINING HISTORY (3 minutes)
# ============================================================================

# TODO: Plot training and validation accuracy
# plt.figure(figsize=(12, 4))
#
# # Accuracy subplot
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
# plt.title('Model Accuracy Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True, alpha=0.3)
#
# # Loss subplot
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Training Loss', marker='o')
# plt.plot(history.history['val_loss'], label='Validation Loss', marker='s')
# plt.title('Model Loss Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.savefig('output/training_history.png', dpi=150, bbox_inches='tight')
# plt.show()
# print("✓ Training history plots saved to output/training_history.png")


# ============================================================================
# PART 8: EVALUATE ON TEST SET (2 minutes)
# ============================================================================

# TODO: Evaluate model on test data
# test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
# print(f"\n{'='*60}")
# print("FINAL TEST RESULTS:")
# print(f"{'='*60}")
# print(f"Test Loss: {test_loss:.4f}")
# print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
# print(f"{'='*60}")


# ============================================================================
# PART 9: VISUALIZE PREDICTIONS (3 minutes)
# ============================================================================

# TODO: Make predictions on test set
# predictions = model.predict(X_test[:10])
# predicted_classes = np.argmax(predictions, axis=1)
# true_classes = np.argmax(y_test[:10], axis=1)

# TODO: Visualize predictions
# plt.figure(figsize=(15, 3))
# for i in range(10):
#     plt.subplot(2, 5, i + 1)
#     plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
#     pred_label = class_names[predicted_classes[i]]
#     true_label = class_names[true_classes[i]]
#     color = 'green' if predicted_classes[i] == true_classes[i] else 'red'
#     plt.title(f'Pred: {pred_label}\nTrue: {true_label}', color=color, fontsize=8)
#     plt.axis('off')
# plt.tight_layout()
# plt.savefig('output/predictions.png', dpi=150, bbox_inches='tight')
# plt.show()
# print("✓ Prediction visualizations saved to output/predictions.png")


# ============================================================================
# PART 10: VISUALIZE LEARNED FILTERS (BONUS - 7 minutes)
# ============================================================================

# TODO: Extract first convolutional layer filters
# filters, biases = model.layers[0].get_weights()
# print(f"\nFirst Conv Layer Filters Shape: {filters.shape}")
# print(f"(kernel_height, kernel_width, input_channels, output_filters)")

# TODO: Normalize filters to [0, 1] for visualization
# f_min, f_max = filters.min(), filters.max()
# filters_normalized = (filters - f_min) / (f_max - f_min)

# TODO: Visualize first 8 filters
# plt.figure(figsize=(12, 3))
# for i in range(8):
#     plt.subplot(2, 4, i + 1)
#     # Get filter for i-th output channel, squeeze to remove input channel dimension
#     filter_img = filters_normalized[:, :, 0, i]
#     plt.imshow(filter_img, cmap='viridis')
#     plt.title(f'Filter {i+1}')
#     plt.axis('off')
# plt.suptitle('Learned Filters in First Conv Layer (3x3 kernels)', fontsize=14, y=1.02)
# plt.tight_layout()
# plt.savefig('output/learned_filters.png', dpi=150, bbox_inches='tight')
# plt.show()
# print("✓ Learned filters saved to output/learned_filters.png")


# ============================================================================
# PART 11: VISUALIZE FEATURE MAPS (BONUS - 7 minutes)
# ============================================================================

# TODO: Create a model that outputs feature maps from intermediate layers
# layer_names = ['conv1', 'pool1', 'conv2', 'pool2']
# layer_outputs = [model.get_layer(name).output for name in layer_names]
# activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)

# TODO: Get feature maps for a sample image
# sample_image = X_test[0:1]  # First test image
# activations = activation_model.predict(sample_image)

# TODO: Visualize feature maps
# fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# axes = axes.flatten()
#
# for i, (layer_name, activation) in enumerate(zip(layer_names, activations)):
#     n_features = activation.shape[-1]  # Number of feature maps
#     size_h, size_w = activation.shape[1], activation.shape[2]
#
#     # Create grid to display feature maps
#     n_cols = 8
#     n_rows = n_features // n_cols
#     display_grid = np.zeros((size_h * n_rows, size_w * n_cols))
#
#     # Fill the grid with feature maps
#     for row in range(n_rows):
#         for col in range(n_cols):
#             channel_idx = row * n_cols + col
#             if channel_idx < n_features:
#                 channel_image = activation[0, :, :, channel_idx]
#                 # Normalize for display
#                 channel_image -= channel_image.mean()
#                 if channel_image.std() > 0:
#                     channel_image /= channel_image.std()
#                 channel_image = np.clip(channel_image, 0, 1)
#                 display_grid[row * size_h:(row + 1) * size_h,
#                              col * size_w:(col + 1) * size_w] = channel_image
#
#     axes[i].imshow(display_grid, cmap='viridis')
#     axes[i].set_title(f'{layer_name} - Shape: {activation.shape[1:]}')
#     axes[i].axis('off')
#
# plt.suptitle('Feature Maps at Different CNN Layers', fontsize=14, y=0.995)
# plt.tight_layout()
# plt.savefig('output/feature_maps.png', dpi=150, bbox_inches='tight')
# plt.show()
# print("✓ Feature maps saved to output/feature_maps.png")


# ============================================================================
# PART 12: SUMMARY AND NEXT STEPS
# ============================================================================

print("\n" + "="*60)
print("TUTORIAL T10 COMPLETED!")
print("="*60)
print("✓ Loaded Fashion-MNIST dataset (60,000 train + 10,000 test)")
print("✓ Built CNN with 2 conv layers and 2 pooling layers")
print("✓ Trained model for 10 epochs")
print("✓ Achieved ~90% accuracy on test set")
print("✓ Visualized training history, predictions, filters, and feature maps")
print("="*60)

print("\n📚 KEY TAKEAWAYS:")
print("1. CNNs use convolutional layers to learn spatial features")
print("2. Pooling layers reduce dimensions and add translation invariance")
print("3. Early layers learn simple features (edges, textures)")
print("4. Deeper layers learn complex features (shapes, objects)")
print("5. CNNs achieve high accuracy with fewer parameters than MLPs")

print("\n🏠 HOMEWORK (Due: Before Week 11):")
print("Task 1: Manual convolution calculation (6x6 image, 3x3 kernel)")
print("Task 2: Design CNN architecture for MNIST (with justification)")
print("Task 3: Modify this code - add layer, experiment with kernel sizes")

print("\n📖 NEXT WEEK (Week 11):")
print("- Famous CNN architectures (LeNet, AlexNet, VGG)")
print("- Advanced CNN techniques (Dropout, Batch Normalization)")
print("- Designing deeper networks")

print("\n🎯 UNIT TEST 2 PREP (October 31):")
print("- Review convolution calculations")
print("- Practice output dimension formulas")
print("- Understand parameter counting")
print("- Be ready to design and justify CNN architectures")
print("="*60)


# ============================================================================
# EXTENSION EXERCISES (For Fast Finishers)
# ============================================================================

"""
EXTENSION 1: Add Dropout Regularization
- Add Dropout(0.25) after each pooling layer
- Add Dropout(0.5) after dense layer
- Compare training history with and without dropout

EXTENSION 2: Compare with MLP
- Build an MLP with same number of parameters
- Compare training time and accuracy
- Discuss advantages of CNNs

EXTENSION 3: Experiment with Architecture
- Try different numbers of filters (16, 32, 64, 128)
- Try different kernel sizes (3x3, 5x5, 7x7)
- Try different numbers of layers (1, 2, 3 conv layers)
- Document which configuration works best

EXTENSION 4: Confusion Matrix
- Generate predictions for entire test set
- Create confusion matrix using sklearn
- Identify which classes are most confused
- Hypothesize why certain classes are hard to distinguish

EXTENSION 5: Data Augmentation
- Use ImageDataGenerator for random rotations, shifts, flips
- Train model with augmented data
- Compare accuracy with and without augmentation
"""
