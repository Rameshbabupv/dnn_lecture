# Week 10 Homework Assignment

**Deep Neural Network Architectures (21CSE558T)**
**Module 4: CNN Basics - Week 10**
**Due Date: Before Week 11 Lecture (November 3, 2025, Monday)**

---

## Assignment Overview

This homework reinforces concepts from Week 10 (CNN fundamentals) through:
1. Manual calculation practice
2. Architecture design with justification
3. Hands-on code experimentation

**Total Points: 100**

---

## Submission Guidelines

### What to Submit:
Create a ZIP file named: `Week10_YourName_RollNumber.zip` containing:
- `task1_calculations.pdf` (or `.docx`) - Handwritten/typed calculations
- `task2_architecture_design.pdf` (or `.docx`) - Architecture design document
- `task3_code.py` - Modified code from Tutorial T10
- `task3_observations.pdf` (or `.docx`) - Experimental observations

### How to Submit:
- Upload ZIP file to course management system
- Email to: [Instructor email]
- Subject line: "Week 10 HW - [Your Name] - [Roll Number]"

### Late Submission Policy:
- On time (by Nov 3, 11:59 PM): Full credit
- 1 day late: -10%
- 2 days late: -20%
- >2 days late: -50%

---

## Task 1: Manual Convolution Calculation (30 points)

### Objective
Practice calculating 2D convolution operations by hand to deeply understand the mathematical mechanics of CNNs.

### Problem Statement

Given the following **6×6 grayscale image** (pixel values):

```
Image I:
┌─────────────────┐
│  3  1  2  0  1  4 │
│  1  5  3  2  1  0 │
│  2  1  4  3  2  1 │
│  0  3  1  5  4  2 │
│  1  2  3  1  2  3 │
│  4  0  1  2  1  0 │
└─────────────────┘
```

And this **3×3 edge detection kernel**:

```
Kernel K:
┌──────────┐
│ -1  -1  -1 │
│  0   0   0 │
│  1   1   1 │
└──────────┘
```

**Parameters:**
- Stride: 1
- Padding: 0 (valid convolution)

### Requirements

**Part A: Calculate Output Dimensions (5 points)**
1. Use the formula: `output_size = (input_size - kernel_size + 2×padding) / stride + 1`
2. Calculate both height and width
3. Show your work

**Part B: Perform Convolution (20 points)**
1. Calculate **ALL** output values in the resulting feature map
2. For **at least 3 positions**, show:
   - The 3×3 region being convolved
   - Element-wise multiplication
   - Summation to get final value
3. Present final output as a matrix

**Example calculation for position (0,0):**
```
Region:        Kernel:        Multiplication:
┌───────┐     ┌──────────┐   ┌──────────────┐
│ 3 1 2 │  *  │ -1 -1 -1 │ = │ -3  -1  -2   │
│ 1 5 3 │     │  0  0  0 │   │  0   0   0   │
│ 2 1 4 │     │  1  1  1 │   │  2   1   4   │
└───────┘     └──────────┘   └──────────────┘

Sum = -3 + (-1) + (-2) + 0 + 0 + 0 + 2 + 1 + 4 = 1
Output[0,0] = 1
```

**Part C: Interpretation (5 points)**
1. What does this kernel detect? (horizontal edges, vertical edges, etc.)
2. Why? Explain based on kernel values
3. Which output values are largest? What does this tell us about the image?

### Grading Rubric

| Criteria | Points | Description |
|----------|--------|-------------|
| Output dimensions correct | 5 | Formula used, calculation shown |
| All output values calculated | 12 | Complete output matrix |
| Detailed steps for 3+ positions | 8 | Element-wise mult, summation shown |
| Interpretation | 5 | Kernel purpose, edge explanation |

---

## Task 2: CNN Architecture Design (40 points)

### Objective
Design a complete CNN architecture for MNIST digit classification with proper justification for all design choices.

### Problem Statement

Design a CNN to classify **MNIST handwritten digits** with the following specifications:

**Dataset:**
- Input: 28×28 grayscale images
- Classes: 10 (digits 0-9)
- Training: 60,000 images
- Test: 10,000 images

**Requirements:**
- Use at least 2 convolutional layers
- Include at least 1 pooling layer
- Target accuracy: ≥ 98% on test set
- Training time: < 5 minutes (CPU)

### Deliverables

**Part A: Architecture Specification (15 points)**

Fill in this table completely:

| Layer # | Layer Type | Parameters | Output Shape | # Parameters | Calculation |
|---------|------------|------------|--------------|--------------|-------------|
| Input   | Input      | -          | 28×28×1      | 0            | -           |
| 1       | Conv2D     | filters=?, kernel=?, activation=? | ?×?×? | ? | Show calculation |
| 2       | ?          | ?          | ?×?×?        | ?            | Show calculation |
| ...     | ...        | ...        | ...          | ...          | ...         |
| Output  | Dense      | units=10, activation=softmax | 10 | ? | Show calculation |

**Example row:**
| 1 | Conv2D | filters=32, kernel=(3,3), activation='relu' | 26×26×32 | 320 | (3×3×1+1)×32=320 |

**Part B: Output Dimension Calculations (10 points)**

For **each convolutional and pooling layer**, show the calculation:

**Example:**
```
Layer 1 - Conv2D (32 filters, 3×3 kernel, stride=1, padding='valid'):
  Input: 28×28×1
  Formula: (W - F + 2P) / S + 1
  Calculation: (28 - 3 + 0) / 1 + 1 = 26
  Output: 26×26×32
```

**Part C: Design Justification (10 points)**

Answer these questions (1-2 paragraphs each):

1. **Filter Counts:** Why did you choose these specific numbers of filters (e.g., 32, 64, 128)?

2. **Kernel Sizes:** Why did you choose these kernel sizes (e.g., 3×3, 5×5)?

3. **Pooling Strategy:** Why did you place pooling layers where you did? Why that pool size?

4. **Activation Functions:** Why ReLU for hidden layers? Why softmax for output?

5. **Number of Layers:** Why this depth? Why not deeper or shallower?

6. **Parameter Count:** What is your total parameter count? Compare to an equivalent MLP (Flatten → Dense → Output). Which is more efficient?

**Part D: Training Configuration (5 points)**

Specify:
- Optimizer: (e.g., Adam, SGD, RMSprop)
- Loss function: (e.g., categorical_crossentropy)
- Batch size: (e.g., 32, 64, 128)
- Number of epochs: (e.g., 10, 20)
- Learning rate: (if not default)

Briefly justify each choice.

### Grading Rubric

| Criteria | Points | Description |
|----------|--------|-------------|
| Complete architecture table | 15 | All layers specified with parameters |
| Dimension calculations | 10 | All calculations shown and correct |
| Design justifications | 10 | Thoughtful, technically sound reasoning |
| Training configuration | 5 | Appropriate choices with justification |

---

## Task 3: Code Experimentation (30 points)

### Objective
Gain hands-on experience modifying CNN architectures and analyzing the effects on performance.

### Base Code
Start with `tutorial_t10_solution.py` from Tutorial T10 (Fashion-MNIST CNN).

### Experiments

Perform **all three experiments** and document results:

#### **Experiment 1: Add Third Convolutional Block (10 points)**

**Modification:**
Add a third convolutional block after the second pooling layer:
```python
# Original architecture (2 conv blocks)
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(64, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),
layers.Flatten(),
...

# Modified architecture (3 conv blocks)
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(64, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(128, (3, 3), activation='relu'),  # NEW BLOCK
layers.MaxPooling2D((2, 2)),                      # NEW BLOCK
layers.Flatten(),
...
```

**Record:**
- New model parameter count: ?
- Training time: ? seconds/epoch
- Final training accuracy: ?
- Final validation accuracy: ?
- Test accuracy: ?

**Analysis:**
- Did accuracy improve? By how much?
- Did training time increase? By how much?
- Is the tradeoff worth it?

---

#### **Experiment 2: Compare Kernel Sizes (10 points)**

**Modification:**
Train three separate models with different kernel sizes:
- Model A: All 3×3 kernels (baseline)
- Model B: All 5×5 kernels
- Model C: All 7×7 kernels

Keep everything else constant (same number of filters, same architecture).

**Record (fill table):**

| Model | Kernel Size | Parameters | Train Time | Train Acc | Val Acc | Test Acc |
|-------|-------------|------------|------------|-----------|---------|----------|
| A     | 3×3         | ?          | ? sec/epoch | ?        | ?       | ?        |
| B     | 5×5         | ?          | ? sec/epoch | ?        | ?       | ?        |
| C     | 7×7         | ?          | ? sec/epoch | ?        | ?       | ?        |

**Analysis:**
- Which kernel size performs best? Why?
- How does kernel size affect parameter count?
- What is the tradeoff between kernel size and performance?

---

#### **Experiment 3: Filter Count Variation (10 points)**

**Modification:**
Train three models with different filter progressions:
- Model X: 16 → 32 filters (fewer)
- Model Y: 32 → 64 filters (baseline)
- Model Z: 64 → 128 filters (more)

**Record (fill table):**

| Model | Filters | Parameters | Train Time | Train Acc | Val Acc | Test Acc | Overfitting? |
|-------|---------|------------|------------|-----------|---------|----------|--------------|
| X     | 16→32   | ?          | ? sec/epoch | ?        | ?       | ?        | ?            |
| Y     | 32→64   | ?          | ? sec/epoch | ?        | ?       | ?        | ?            |
| Z     | 64→128  | ?          | ? sec/epoch | ?        | ?       | ?        | ?            |

**Analysis:**
- How does filter count affect accuracy?
- Is there a point of diminishing returns?
- Which model shows signs of overfitting (train_acc >> val_acc)?

---

### Overall Observations (Bonus +5 points)

Write 2-3 paragraphs addressing:

1. **Key Insights:** What did you learn about CNN architecture choices?

2. **Best Configuration:** Based on your experiments, what would be your recommended architecture for Fashion-MNIST? Why?

3. **Tradeoffs:** Discuss the tradeoff between:
   - Model complexity vs accuracy
   - Training time vs performance
   - Number of parameters vs overfitting risk

4. **Future Exploration:** What other experiments would you like to try?

### Code Submission Requirements

Your `task3_code.py` should:
- Include **all three experimental models**
- Have clear comments indicating each experiment
- Print results in a organized format
- Run without errors
- Be well-structured and readable

**Code Template:**
```python
"""
Week 10 Homework - Task 3: Code Experimentation
Student: [Your Name]
Roll Number: [Your Roll Number]
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

# Load and preprocess data
# ... (your code)

# ==============================================================================
# EXPERIMENT 1: Add Third Convolutional Block
# ==============================================================================
print("="*60)
print("EXPERIMENT 1: Third Convolutional Block")
print("="*60)

# Define model
model_exp1 = keras.Sequential([
    # ... your architecture ...
])

# Train and evaluate
# ... (your code)

print(f"Results:")
print(f"  Parameters: {model_exp1.count_params()}")
print(f"  Test Accuracy: {test_acc:.4f}")
# ... etc.

# ==============================================================================
# EXPERIMENT 2: Kernel Size Comparison
# ==============================================================================
# ... (similar structure)

# ==============================================================================
# EXPERIMENT 3: Filter Count Variation
# ==============================================================================
# ... (similar structure)
```

### Grading Rubric

| Criteria | Points | Description |
|----------|--------|-------------|
| Experiment 1 complete | 10 | Code runs, results documented |
| Experiment 2 complete | 10 | All 3 models trained, table filled |
| Experiment 3 complete | 10 | All 3 models trained, analysis included |
| Bonus observations | +5 | Insightful analysis, well-written |

---

## Summary Checklist

Before submitting, verify you have:

**Task 1:**
- [ ] Output dimensions calculated
- [ ] All convolution values computed
- [ ] Detailed steps shown for 3+ positions
- [ ] Kernel interpretation provided

**Task 2:**
- [ ] Complete architecture table
- [ ] Output dimension calculations for all layers
- [ ] Justifications for all design choices
- [ ] Training configuration specified

**Task 3:**
- [ ] Code runs without errors
- [ ] Experiment 1: Third conv block results
- [ ] Experiment 2: Three kernel size models
- [ ] Experiment 3: Three filter count models
- [ ] Observations document (2-3 paragraphs)

**Submission:**
- [ ] All files in one ZIP file
- [ ] ZIP file named correctly: `Week10_YourName_RollNumber.zip`
- [ ] Submitted before deadline (Nov 3, 11:59 PM)

---

## Grading Summary

| Task | Points | Description |
|------|--------|-------------|
| Task 1: Manual Calculation | 30 | Convolution calculations and interpretation |
| Task 2: Architecture Design | 40 | MNIST CNN design with justifications |
| Task 3: Code Experimentation | 30 | Three experiments with analysis |
| **Total** | **100** | |
| Bonus: Insightful observations | +5 | Exceptional analysis in Task 3 |

---

## Learning Objectives

Upon completing this homework, you will be able to:
1. ✅ Calculate convolution operations manually
2. ✅ Design CNN architectures with proper justification
3. ✅ Analyze the effect of architectural choices on performance
4. ✅ Make informed decisions about CNN hyperparameters
5. ✅ Understand tradeoffs in deep learning model design

---

## Additional Resources

- Week 10 Lecture Notes (DO3 Jupyter notebooks)
- Tutorial T10 materials (starter and solution code)
- Quick Reference Cheat Sheet
- Troubleshooting Guide
- Chollet Chapter 5: "Deep learning for computer vision"

---

## Academic Integrity

- You may discuss concepts with classmates
- All calculations, code, and writing must be your own
- Cite any external resources used
- Do not copy code from the internet without understanding and modification
- Plagiarism will result in zero points and academic consequences

---

## Questions?

If you have questions:
- Review the lecture materials and tutorial first
- Check the Troubleshooting Guide
- Post on course forum (no solution sharing!)
- Attend office hours: [Insert times]
- Email instructor: [Insert email]

---

**Good luck! Remember, the goal is deep understanding, not just completing tasks. Take your time, experiment, and learn! 🚀**

---

**Assignment Released:** October 29, 2025
**Due Date:** November 3, 2025, 11:59 PM
**Late Submissions:** Accepted with penalty (see policy above)
