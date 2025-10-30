"""
Tutorial T11: CIFAR-10 Multiclass Classification with Modern CNN
Course: 21CSE558T - Deep Neural Network Architectures
Module 4: CNNs (Week 2 of 3)
Date: Monday, November 3, 2025
Duration: 1 hour

STARTER CODE - Complete the TODOs
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation,
    MaxPooling2D, GlobalAveragePooling2D,
    Dropout, Dense, Flatten
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# =============================================================================
# PART 1: LOAD AND EXPLORE CIFAR-10 DATASET
# =============================================================================

print("\n" + "="*70)
print("PART 1: LOADING CIFAR-10 DATASET")
print("="*70)

# TODO 1.1: Load CIFAR-10 dataset
# Hint: Use cifar10.load_data()
(x_train, y_train), (x_test, y_test) = None  # TODO: Replace None

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Training samples: {x_train.shape[0] if x_train is not None else 'TODO'}")
print(f"Test samples: {x_test.shape[0] if x_test is not None else 'TODO'}")
print(f"Image shape: {x_train.shape[1:] if x_train is not None else 'TODO'}")
print(f"Number of classes: {len(class_names)}")

# TODO 1.2: Visualize sample images (9 images in 3×3 grid)
# Hint: Use plt.subplots() and plt.imshow()
def visualize_samples(images, labels, class_names, n=9):
    """Visualize n sample images with labels"""
    # TODO: Complete this function
    pass

# visualize_samples(x_train, y_train, class_names)

# =============================================================================
# PART 2: DATA PREPROCESSING
# =============================================================================

print("\n" + "="*70)
print("PART 2: DATA PREPROCESSING")
print("="*70)

# TODO 2.1: Normalize images to [0, 1]
# Hint: Divide by 255.0
x_train_norm = None  # TODO
x_test_norm = None   # TODO

# TODO 2.2: Convert labels to one-hot encoding
# Hint: Use to_categorical(labels, num_classes)
y_train_cat = None  # TODO
y_test_cat = None   # TODO

# TODO 2.3: Create validation split (10% of training data)
# Hint: Use train_test_split or slice the data
val_split = 0.1
val_samples = int(len(x_train_norm) * val_split)

x_val = None    # TODO: Last val_samples of x_train_norm
y_val = None    # TODO: Last val_samples of y_train_cat
x_train_final = None  # TODO: Remaining training samples
y_train_final = None  # TODO: Remaining training labels

print(f"Final training samples: {x_train_final.shape[0] if x_train_final is not None else 'TODO'}")
print(f"Validation samples: {x_val.shape[0] if x_val is not None else 'TODO'}")
print(f"Test samples: {x_test_norm.shape[0] if x_test_norm is not None else 'TODO'}")

# =============================================================================
# PART 3: BUILD BASELINE MODEL (Week 10 Style)
# =============================================================================

print("\n" + "="*70)
print("PART 3: BUILDING BASELINE MODEL")
print("="*70)

def build_baseline_model():
    """
    Baseline CNN (Week 10 style):
    - Simple Conv → Pool → Conv → Pool → Flatten → Dense
    - NO BatchNorm, NO Dropout, NO modern techniques
    """
    model = Sequential([
        # TODO 3.1: Add Conv2D(32 filters, 3×3, activation='relu', padding='same')
        # Hint: Don't forget input_shape=(32, 32, 3)

        # TODO 3.2: Add MaxPooling2D(2×2)

        # TODO 3.3: Add Conv2D(64 filters, 3×3, activation='relu', padding='same')

        # TODO 3.4: Add MaxPooling2D(2×2)

        # TODO 3.5: Add Flatten()

        # TODO 3.6: Add Dense(128, activation='relu')

        # TODO 3.7: Add Dense(10, activation='softmax')

    ], name='Baseline_CNN')

    return model

# Create and compile baseline model
model_baseline = build_baseline_model()

# TODO 3.8: Compile model with optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']
# model_baseline.compile(...)

# TODO 3.9: Print model summary
# model_baseline.summary()

# =============================================================================
# PART 4: BUILD MODERN MODEL (Week 11 Style with Regularization)
# =============================================================================

print("\n" + "="*70)
print("PART 4: BUILDING MODERN MODEL (WITH REGULARIZATION)")
print("="*70)

def build_modern_model():
    """
    Modern CNN (Week 11 style):
    - BatchNormalization after Conv, before activation
    - Dropout after pooling
    - GlobalAveragePooling instead of Flatten
    - Progressive filter growth: 32 → 64 → 128
    """
    model = Sequential([
        # Block 1: 32 filters
        # TODO 4.1: Conv2D(32, 3×3, padding='same', input_shape=(32,32,3)) - NO activation!
        # TODO 4.2: BatchNormalization()
        # TODO 4.3: Activation('relu')
        # TODO 4.4: Conv2D(32, 3×3, padding='same') - NO activation!
        # TODO 4.5: BatchNormalization()
        # TODO 4.6: Activation('relu')
        # TODO 4.7: MaxPooling2D(2×2)
        # TODO 4.8: Dropout(0.2)

        # Block 2: 64 filters
        # TODO 4.9: Conv2D(64, 3×3, padding='same')
        # TODO 4.10: BatchNormalization()
        # TODO 4.11: Activation('relu')
        # TODO 4.12: Conv2D(64, 3×3, padding='same')
        # TODO 4.13: BatchNormalization()
        # TODO 4.14: Activation('relu')
        # TODO 4.15: MaxPooling2D(2×2)
        # TODO 4.16: Dropout(0.3)

        # Block 3: 128 filters
        # TODO 4.17: Conv2D(128, 3×3, padding='same')
        # TODO 4.18: BatchNormalization()
        # TODO 4.19: Activation('relu')
        # TODO 4.20: GlobalAveragePooling2D() - Modern approach!

        # Classification head
        # TODO 4.21: Dropout(0.5)
        # TODO 4.22: Dense(10, activation='softmax')

    ], name='Modern_CNN')

    return model

# Create and compile modern model
model_modern = build_modern_model()

# TODO 4.23: Compile model
# model_modern.compile(...)

# TODO 4.24: Print model summary and compare parameters with baseline
# model_modern.summary()

# =============================================================================
# PART 5: DATA AUGMENTATION
# =============================================================================

print("\n" + "="*70)
print("PART 5: DATA AUGMENTATION")
print("="*70)

# TODO 5.1: Create ImageDataGenerator for training with augmentation
# Hint: rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
#       horizontal_flip=True, zoom_range=0.1
train_datagen = None  # TODO: Create ImageDataGenerator

# TODO 5.2: Create ImageDataGenerator for validation (NO augmentation!)
# Hint: Only rescaling, no transformations
val_datagen = None  # TODO: Create ImageDataGenerator (no augmentation)

# TODO 5.3: Visualize augmented images
def visualize_augmentation(datagen, image, n=9):
    """Show n augmented versions of the same image"""
    # TODO: Complete this function
    # Hint: Use datagen.flow() and next() to generate augmented images
    pass

# Pick one sample image and visualize augmentations
# sample_image = x_train_norm[0:1]  # Need batch dimension
# visualize_augmentation(train_datagen, sample_image)

# =============================================================================
# PART 6: TRAINING
# =============================================================================

print("\n" + "="*70)
print("PART 6: TRAINING MODELS")
print("="*70)

# TODO 6.1: Create Early Stopping callback
# Hint: monitor='val_loss', patience=5, restore_best_weights=True
early_stop = None  # TODO

# Training configuration
EPOCHS = 30
BATCH_SIZE = 64

# TODO 6.2: Train baseline model (without augmentation)
print("Training Baseline Model (No Regularization)...")
# history_baseline = model_baseline.fit(...)

# TODO 6.3: Train modern model (with augmentation)
print("\nTraining Modern Model (With Regularization + Augmentation)...")
# Hint: Use train_datagen.flow(x_train_final, y_train_final, batch_size=BATCH_SIZE)
# history_modern = model_modern.fit(...)

# =============================================================================
# PART 7: EVALUATION AND COMPARISON
# =============================================================================

print("\n" + "="*70)
print("PART 7: EVALUATION AND COMPARISON")
print("="*70)

# TODO 7.1: Evaluate both models on test set
# test_loss_baseline, test_acc_baseline = model_baseline.evaluate(...)
# test_loss_modern, test_acc_modern = model_modern.evaluate(...)

# TODO 7.2: Plot training curves comparison
def plot_comparison(history1, history2, metric='accuracy'):
    """Plot training curves for two models"""
    # TODO: Complete this function
    # Plot both training and validation curves
    # Use different colors for each model
    pass

# plot_comparison(history_baseline, history_modern, 'accuracy')
# plot_comparison(history_baseline, history_modern, 'loss')

# TODO 7.3: Print final comparison
def print_final_results(baseline_acc, modern_acc):
    """Print formatted comparison of results"""
    print("="*70)
    print("FINAL TEST RESULTS")
    print("="*70)
    print(f"Baseline Model (No Regularization): {baseline_acc:.2%}")
    print(f"Modern Model (Full Regularization):  {modern_acc:.2%}")
    print(f"Improvement:                         +{(modern_acc - baseline_acc):.2%}")
    print("="*70)

# print_final_results(test_acc_baseline, test_acc_modern)

# TODO 7.4: Create confusion matrix for modern model
# Hint: Use sklearn.metrics.confusion_matrix and seaborn.heatmap
def plot_confusion_matrix(model, x_test, y_test, class_names):
    """Plot confusion matrix"""
    # TODO: Complete this function
    pass

# plot_confusion_matrix(model_modern, x_test_norm, y_test, class_names)

# =============================================================================
# PART 8: ANALYSIS QUESTIONS (Answer in comments)
# =============================================================================

"""
ANALYSIS QUESTIONS:

1. Compare parameter counts:
   Baseline parameters: _______
   Modern parameters: _______
   Why the difference? _______________________

2. Training speed:
   Baseline: _____ seconds/epoch
   Modern: _____ seconds/epoch
   Why is modern slower? _____________________

3. Overfitting analysis:
   Baseline train-val gap: ______%
   Modern train-val gap: ______%
   Which technique reduced overfitting most? __________

4. Which augmentation helps most?
   Try disabling each and see impact: _______________

5. What happens if you remove:
   - BatchNormalization? _______________
   - Dropout? _______________
   - GlobalAveragePooling? _______________

6. Which classes are most confused?
   From confusion matrix: _______________
   Why? _______________
"""

# =============================================================================
# BONUS: EXPERIMENT WITH HYPERPARAMETERS
# =============================================================================

"""
BONUS TASKS (Optional):

1. Try different dropout rates:
   - 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
   - Which works best? _______

2. Try different augmentation strategies:
   - Only horizontal flip
   - Only rotation
   - All augmentations
   - Which combination is best? _______

3. Try adding L2 regularization:
   - kernel_regularizer=l2(0.001)
   - Does it help? _______

4. Try different architectures:
   - Deeper (4-5 blocks)
   - Wider (more filters per layer)
   - Which is better? _______
"""

print("\n" + "="*70)
print("TUTORIAL T11 STARTER CODE - COMPLETE THE TODOs!")
print("="*70)
print("\nKey Learning Objectives:")
print("1. Build baseline vs modern CNN architectures")
print("2. Implement BatchNorm, Dropout, GlobalAveragePooling")
print("3. Apply data augmentation")
print("4. Compare and analyze performance")
print("5. Understand regularization impact")
print("\nGood luck! 🚀")
print("="*70)
