#!/usr/bin/env python3
"""
Generate Tutorial T11 Jupyter Notebook
Course: 21CSE558T - Deep Neural Network Architectures
Module 4, Week 11, DO4 Nov-3 Monday
"""

import json
import os

# Base notebook structure
def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def markdown_cell(content):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content.split('\n')
    }

def code_cell(content):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": content.split('\n')
    }

# Tutorial T11 Notebook Cells
tutorial_t11_cells = [
    markdown_cell("""# Tutorial T11: CIFAR-10 with Modern CNN

**Course:** 21CSE558T - Deep Neural Network Architectures
**Module 4:** CNNs (Week 2 of 3)
**Date:** November 3, 2025
**Duration:** 1 hour (50 minutes)

---

## Learning Objectives

By the end of this tutorial, you will:

1. ✅ Build baseline vs modern CNN architectures
2. ✅ Implement Batch Normalization correctly (Conv → BN → Activation)
3. ✅ Apply Dropout strategically (0.2 → 0.3 → 0.5)
4. ✅ Use Global Average Pooling (vs Flatten+Dense)
5. ✅ Implement data augmentation with ImageDataGenerator
6. ✅ Compare and analyze performance improvements
7. ✅ Debug overfitting systematically

---

## Tutorial Structure

**Part 1: Problem Setup (10 min)**
- Load CIFAR-10 dataset
- Explore and visualize samples
- Preprocess data (normalize, one-hot encode, validation split)

**Part 2: Building Architectures (15 min)**
- Baseline model (Week 10 style - no regularization)
- Modern model (Week 11 style - full regularization)

**Part 3: Data Augmentation (15 min)**
- Create augmentation pipeline
- Visualize augmented images
- Train with augmentation

**Part 4: Comparison & Analysis (10 min)**
- Plot training curves
- Compare test accuracies
- Analyze overfitting reduction
- Create confusion matrix

---

## Dataset: CIFAR-10

**60,000 RGB images (32×32)**
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- More challenging than Fashion-MNIST
- Real-world color images
- 50,000 training + 10,000 test

**Expected Results:**
- Baseline (no regularization): ~60-65% test accuracy (severe overfitting)
- Modern (full regularization): ~75-82% test accuracy (good generalization)

---"""),

    code_cell("""# Import libraries
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
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")"""),

    markdown_cell("""---

## PART 1: Load and Explore CIFAR-10 Dataset

### Task 1.1: Load CIFAR-10

Complete the TODO below to load the dataset."""),

    code_cell("""# TODO 1.1: Load CIFAR-10 dataset
# Hint: Use cifar10.load_data()
(x_train, y_train), (x_test, y_test) = None  # TODO: Replace None

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Training samples: {x_train.shape[0] if x_train is not None else 'TODO'}")
print(f"Test samples: {x_test.shape[0] if x_test is not None else 'TODO'}")
print(f"Image shape: {x_train.shape[1:] if x_train is not None else 'TODO'}")
print(f"Number of classes: {len(class_names)}")"""),

    markdown_cell("""### Task 1.2: Visualize Sample Images

Create a 3×3 grid showing 9 sample images with their labels."""),

    code_cell("""# TODO 1.2: Visualize sample images
def visualize_samples(images, labels, class_names, n=9):
    \"\"\"Visualize n sample images with labels\"\"\"
    # TODO: Complete this function
    # Hint: Use plt.subplots(3, 3) and plt.imshow()
    pass

# Uncomment to test:
# visualize_samples(x_train, y_train, class_names)"""),

    markdown_cell("""**Expected Output:** 3×3 grid of CIFAR-10 images with class labels

---

## PART 2: Data Preprocessing

### Task 2.1: Normalize Images

CIFAR-10 images have pixel values in range [0, 255]. We need to normalize to [0, 1]."""),

    code_cell("""# TODO 2.1: Normalize images to [0, 1]
# Hint: Divide by 255.0
x_train_norm = None  # TODO
x_test_norm = None   # TODO

print(f"Training data range: [{x_train_norm.min() if x_train_norm is not None else 'TODO'}, {x_train_norm.max() if x_train_norm is not None else 'TODO'}]")
print("Expected: [0.0, 1.0]")"""),

    markdown_cell("""### Task 2.2: One-Hot Encode Labels

Convert class indices (0-9) to one-hot vectors."""),

    code_cell("""# TODO 2.2: Convert labels to one-hot encoding
# Hint: Use to_categorical(labels, num_classes)
y_train_cat = None  # TODO
y_test_cat = None   # TODO

print(f"Original label shape: {y_train.shape if y_train is not None else 'TODO'}")
print(f"One-hot label shape: {y_train_cat.shape if y_train_cat is not None else 'TODO'}")
print("Expected: (50000, 10)")"""),

    markdown_cell("""### Task 2.3: Create Validation Split

Split 10% of training data for validation."""),

    code_cell("""# TODO 2.3: Create validation split (10% of training data)
# Hint: Use slicing to separate last 10% as validation
val_split = 0.1
val_samples = int(len(x_train_norm) * val_split) if x_train_norm is not None else 0

x_val = None    # TODO: Last val_samples of x_train_norm
y_val = None    # TODO: Last val_samples of y_train_cat
x_train_final = None  # TODO: Remaining training samples
y_train_final = None  # TODO: Remaining training labels

print(f"Final training samples: {x_train_final.shape[0] if x_train_final is not None else 'TODO'}")
print(f"Validation samples: {x_val.shape[0] if x_val is not None else 'TODO'}")
print(f"Test samples: {x_test_norm.shape[0] if x_test_norm is not None else 'TODO'}")
print("Expected: 45000, 5000, 10000")"""),

    markdown_cell("""---

## PART 3: Build Baseline Model (Week 10 Style)

### Architecture: Simple CNN (No Regularization)

**Structure:**
```
Conv2D(32) → MaxPool
Conv2D(64) → MaxPool
Flatten
Dense(128)
Dense(10)
```

**Problems:**
- ❌ No BatchNorm (slow training)
- ❌ No Dropout (overfitting)
- ❌ Flatten+Dense (parameter explosion)

### Task 3: Build Baseline Model

Complete the TODOs to build the baseline architecture."""),

    code_cell("""# TODO 3: Build baseline model
def build_baseline_model():
    \"\"\"
    Baseline CNN (Week 10 style):
    - Simple Conv → Pool → Conv → Pool → Flatten → Dense
    - NO BatchNorm, NO Dropout, NO modern techniques
    \"\"\"
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
# model_baseline.summary()"""),

    markdown_cell("""**Expected Output:**
- Total parameters: ~500,000
- Trainable parameters: ~500,000
- Most parameters in Flatten → Dense(128) layer

---

## PART 4: Build Modern Model (Week 11 Style)

### Architecture: Modern CNN (Full Regularization)

**Structure:**
```
Block 1:
Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.2)

Block 2:
Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.3)

Block 3:
Conv2D(128) → BN → ReLU → GlobalAvgPool

Output:
Dropout(0.5) → Dense(10)
```

**Improvements:**
- ✅ BatchNorm (faster training, regularization)
- ✅ Dropout (prevents overfitting)
- ✅ GlobalAvgPool (parameter reduction)
- ✅ Double conv per block (deeper features)

### Task 4: Build Modern Model

**IMPORTANT PATTERN:**
```python
Conv2D(filters, kernel_size, padding='same')  # NO activation!
BatchNormalization()
Activation('relu')
```

Complete the TODOs below."""),

    code_cell("""# TODO 4: Build modern model
def build_modern_model():
    \"\"\"
    Modern CNN (Week 11 style):
    - BatchNormalization after Conv, before activation
    - Dropout after pooling
    - GlobalAveragePooling instead of Flatten
    - Progressive filter growth: 32 → 64 → 128
    \"\"\"
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
# model_modern.summary()"""),

    markdown_cell("""**Expected Output:**
- Total parameters: ~220,000 (2.3× fewer than baseline!)
- GlobalAvgPool dramatically reduces parameters
- BatchNorm adds small parameter overhead

**Question:** Why fewer parameters despite more layers?

**Answer:** GlobalAveragePooling2D reduces 8×8×128 = 8,192 values to just 128 values (one per channel). This makes the final Dense layer much smaller:
- Baseline: Flatten (8,192) → Dense(128) = 1,048,576 parameters
- Modern: GlobalAvgPool (128) → Dense(10) = 1,280 parameters

---

## PART 5: Data Augmentation

### Why Augmentation?

CIFAR-10 has only 50,000 training images. Data augmentation creates variations:
- Rotation (±15°)
- Horizontal shift (±10%)
- Vertical shift (±10%)
- Horizontal flip (mirror)
- Zoom (90-110%)

**Result:** Effectively increases dataset size, improves generalization.

### Task 5: Create Augmentation Pipeline

**CRITICAL:** Only augment TRAINING data, NOT validation/test!"""),

    code_cell("""# TODO 5.1: Create ImageDataGenerator for training with augmentation
# Hint: rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
#       horizontal_flip=True, zoom_range=0.1
train_datagen = None  # TODO: Create ImageDataGenerator

# TODO 5.2: Create ImageDataGenerator for validation (NO augmentation!)
# Validation should use original images only
val_datagen = None  # TODO: Create ImageDataGenerator (no augmentation)

print("Augmentation setup complete!")
print("Training: WITH augmentation")
print("Validation/Test: NO augmentation (original images only)")"""),

    markdown_cell("""### Task 5.3: Visualize Augmentations

See what augmentation does to images."""),

    code_cell("""# TODO 5.3: Visualize augmented images
def visualize_augmentation(datagen, image, n=9):
    \"\"\"Show n augmented versions of the same image\"\"\"
    # TODO: Complete this function
    # Hint: Use datagen.flow() and next() to generate augmented images
    # Display in 3×3 grid
    pass

# Uncomment to test:
# sample_image = x_train_norm[0:1]  # Need batch dimension
# visualize_augmentation(train_datagen, sample_image)"""),

    markdown_cell("""**Expected Output:** 9 different augmented versions of the same image

**Verify:**
- ✅ Rotations look natural (±15° max)
- ✅ Shifts don't cut off important parts
- ✅ Flips make sense (airplane left/right OK)
- ❌ NO vertical flips (planes shouldn't be upside down!)

---

## PART 6: Training

### Training Configuration

- **Epochs:** 30 (with early stopping)
- **Batch size:** 64
- **Optimizer:** Adam
- **Loss:** Categorical crossentropy
- **Early stopping:** Patience=5 (stop if no improvement for 5 epochs)

### Task 6: Train Both Models"""),

    code_cell("""# TODO 6.1: Create Early Stopping callback
# Hint: monitor='val_loss', patience=5, restore_best_weights=True
early_stop = None  # TODO

# Training configuration
EPOCHS = 30
BATCH_SIZE = 64

print("Training configuration:")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Early stopping: Patience=5")"""),

    markdown_cell("""### Task 6.2: Train Baseline Model (No Augmentation)

This will take ~5-10 minutes on CPU."""),

    code_cell("""# TODO 6.2: Train baseline model (without augmentation)
print("=" * 70)
print("Training Baseline Model (No Regularization)")
print("=" * 70)

# history_baseline = model_baseline.fit(
#     x_train_final, y_train_final,
#     validation_data=(x_val, y_val),
#     epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     callbacks=[early_stop],
#     verbose=1
# )

print("Baseline training complete!")"""),

    markdown_cell("""### Task 6.3: Train Modern Model (With Augmentation)

This will take ~10-15 minutes on CPU (augmentation adds overhead)."""),

    code_cell("""# TODO 6.3: Train modern model (with augmentation)
print("\\n" + "=" * 70)
print("Training Modern Model (With Regularization + Augmentation)")
print("=" * 70)

# Hint: Use train_datagen.flow(x_train_final, y_train_final, batch_size=BATCH_SIZE)
# history_modern = model_modern.fit(
#     train_datagen.flow(x_train_final, y_train_final, batch_size=BATCH_SIZE),
#     steps_per_epoch=len(x_train_final) // BATCH_SIZE,
#     validation_data=(x_val, y_val),
#     epochs=EPOCHS,
#     callbacks=[early_stop],
#     verbose=1
# )

print("Modern training complete!")"""),

    markdown_cell("""**Expected Training Time:**
- Baseline: ~5-10 min on CPU
- Modern: ~10-15 min on CPU (augmentation overhead)
- With GPU: 2-3× faster

**Watch for:**
- Baseline: Train accuracy >> Val accuracy (overfitting!)
- Modern: Train ≈ Val accuracy (good generalization)

---

## PART 7: Evaluation and Comparison

### Task 7.1: Evaluate on Test Set"""),

    code_cell("""# TODO 7.1: Evaluate both models on test set
# test_loss_baseline, test_acc_baseline = model_baseline.evaluate(x_test_norm, y_test_cat)
# test_loss_modern, test_acc_modern = model_modern.evaluate(x_test_norm, y_test_cat)

# print("=" * 70)
# print("FINAL TEST RESULTS")
# print("=" * 70)
# print(f"Baseline Model (No Regularization): {test_acc_baseline:.2%}")
# print(f"Modern Model (Full Regularization):  {test_acc_modern:.2%}")
# print(f"Improvement:                         +{(test_acc_modern - test_acc_baseline):.2%}")
# print("=" * 70)"""),

    markdown_cell("""**Expected Results:**
- Baseline: ~60-65% test accuracy
- Modern: ~75-82% test accuracy
- Improvement: +15-20%

### Task 7.2: Plot Training Curves"""),

    code_cell("""# TODO 7.2: Plot training curves comparison
def plot_comparison(history1, history2, metric='accuracy'):
    \"\"\"Plot training curves for two models\"\"\"
    # TODO: Complete this function
    # Plot both training and validation curves
    # Use different colors for each model
    # Add legend, labels, title
    pass

# Uncomment to test:
# plot_comparison(history_baseline, history_modern, 'accuracy')
# plot_comparison(history_baseline, history_modern, 'loss')"""),

    markdown_cell("""**Expected Observations:**

**Baseline Model:**
- Train accuracy: 90%+
- Val accuracy: 60-65%
- Large gap (30%) = severe overfitting!

**Modern Model:**
- Train accuracy: 80-85%
- Val accuracy: 75-80%
- Small gap (3-5%) = good generalization!

**Key Insight:** Lower training accuracy but HIGHER test accuracy = regularization working!

### Task 7.3: Confusion Matrix

Identify which classes are most confused."""),

    code_cell("""# TODO 7.3: Create confusion matrix for modern model
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(model, x_test, y_test, class_names):
    \"\"\"Plot confusion matrix\"\"\"
    # TODO: Complete this function
    # 1. Get predictions: y_pred = model.predict(x_test).argmax(axis=1)
    # 2. Get true labels: y_true = y_test.argmax(axis=1)
    # 3. Compute confusion matrix: cm = confusion_matrix(y_true, y_pred)
    # 4. Plot with seaborn heatmap
    pass

# Uncomment to test:
# plot_confusion_matrix(model_modern, x_test_norm, y_test_cat, class_names)"""),

    markdown_cell("""**Common Confusions:**
- Cat ↔ Dog (similar appearance)
- Automobile ↔ Truck (both vehicles)
- Bird ↔ Airplane (both fly)

---

## PART 8: Analysis Questions

Answer these questions based on your results:

### Question 1: Parameter Counts
- Baseline parameters: _______
- Modern parameters: _______
- Why the difference? _______________________

### Question 2: Training Speed
- Baseline: _____ seconds/epoch
- Modern: _____ seconds/epoch
- Why is modern slower? _____________________

### Question 3: Overfitting Analysis
- Baseline train-val gap: ______%
- Modern train-val gap: ______%
- Which technique reduced overfitting most? __________

### Question 4: BatchNorm Placement
What happens if you put activation BEFORE BatchNorm?
Try it: Conv2D(32, activation='relu') → BatchNorm()
Result: _______________________

### Question 5: Dropout After Output?
What happens if you add Dropout(0.5) AFTER the final Dense(10)?
Result: _______________________

### Question 6: Which Classes Are Hardest?
From confusion matrix: _______________
Why? _______________

---

## BONUS: Experiment with Hyperparameters

### Experiment 1: Different Dropout Rates
Try: 0.1, 0.3, 0.5, 0.7
Which works best? _______

### Experiment 2: Augmentation Ablation
- No augmentation: _____% accuracy
- Only horizontal flip: _____% accuracy
- Full augmentation: _____% accuracy

### Experiment 3: GlobalAvgPool vs Flatten
Replace GlobalAveragePooling2D() with Flatten()
- Parameters increase to: _______
- Accuracy changes to: _______%

### Experiment 4: L2 Regularization
Add `kernel_regularizer=l2(0.001)` to Conv layers
Does it help? _______

---

## Summary: Key Takeaways

### Technical Lessons
1. **BatchNorm is essential** - Faster training, acts as regularization
2. **Dropout prevents overfitting** - Especially in FC layers
3. **Augmentation improves generalization** - Worth the training time cost
4. **GlobalAvgPool > Flatten** - Fewer parameters, less overfitting
5. **Regularization trades training accuracy for test accuracy** - This is good!

### Design Principles
1. **Start with BatchNorm** - Almost no downside, huge upside
2. **Add augmentation if dataset < 50K** - CIFAR-10 (50K) is borderline
3. **Add dropout if still overfitting** - Progressive rates (0.2 → 0.3 → 0.5)
4. **Use GlobalAvgPool** - Modern standard, parameter efficient
5. **Monitor train-val gap** - Primary overfitting indicator

### Debugging Checklist
- **Model not learning?** → Check BatchNorm placement, learning rate
- **Overfitting?** → Add/increase dropout, add augmentation
- **Underfitting?** → Reduce dropout, increase model capacity
- **Slow training?** → Reduce batch size, use GPU, cache augmentation
- **Poor test accuracy?** → Check augmentation appropriateness, add more data

---

## Homework Assignment (Due: Before Week 12)

### Task 1: Hyperparameter Tuning (40 points)
Experiment with different configurations:
- Dropout rates: 0.1, 0.3, 0.5, 0.7
- With/without BatchNorm
- Different pooling: Max, Average, Global
- Document results in table

### Task 2: Augmentation Analysis (30 points)
Train 3 models:
- No augmentation
- Only horizontal flip
- Full augmentation (rotation+flip+shift+zoom)
Compare and analyze impact

### Task 3: Transfer to Fashion-MNIST (30 points)
Apply your best architecture to Fashion-MNIST
- Adjust hyperparameters if needed
- Report accuracy and observations
- Compare: Does same strategy work?

---

## Troubleshooting

**If you encounter errors, check:**
1. Data normalized to [0, 1]?
2. Labels one-hot encoded?
3. BatchNorm BEFORE activation?
4. NO dropout after output layer?
5. Using train_datagen.flow() for augmentation?
6. NOT augmenting validation/test data?

**See `troubleshooting_guide.md` for detailed solutions!**

---

**Tutorial T11 Complete! 🚀**

**Next Week:** Transfer Learning with Pre-trained Models (VGG, ResNet)""")
]

# Create notebook
notebook = create_notebook(tutorial_t11_cells)

# Save notebook
output_file = "/Users/rameshbabu/data/projects/srm/lectures/Deep_Neural_Network_Architectures/course_planning/weekly_plans/week11-12-module4-cnn-layers/Do4-Nov-3-Monday/wip/tutorial_t11_cifar10.ipynb"

with open(output_file, 'w') as f:
    json.dump(notebook, f, indent=2)

print("✅ Created: tutorial_t11_cifar10.ipynb")
print(f"   Cells: {len(tutorial_t11_cells)}")
print(f"   Size: ~{len(json.dumps(notebook)) // 1024}KB")
print(f"   Location: {output_file}")
