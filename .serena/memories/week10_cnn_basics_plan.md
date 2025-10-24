# Week 10 - Module 4: CNN Basics Plan

## Overview
- **Week Number:** 10/15
- **Module:** 4 (Convolutional Neural Networks) - Week 1 of 3
- **Dates:** October 20-24, 2025
- **Critical Context:** First week of Module 4, bridging from Week 9 CNN introduction

## Session Schedule

### DO3 (Oct 22) - 2-Hour Lecture
- **Theme:** "HOW do CNNs Work? - Technical Deep Dive"
- **Hour 1:** Convolution mathematics (1D/2D, parameters, calculations)
- **Hour 2:** CNN architecture components (pooling, complete pipeline, 3D preview)

### DO4 (Oct 23) - 1-Hour Tutorial
- **Tutorial T10:** Building Programs to Perform Classification Using CNN In Keras
- **Focus:** First hands-on CNN implementation with Fashion-MNIST/CIFAR-10

## Learning Objectives
1. Calculate convolution operations manually (1D and 2D)
2. Understand convolution parameters (stride, padding, kernel size)
3. Design basic CNN architectures
4. Implement CNN classification in Keras
5. Visualize learned filters and feature maps
6. Compare CNN vs MLP performance

## Key Teaching Strategy
- **Bridge from Week 9:** Conceptual (WHY CNNs) → Technical (HOW CNNs work)
- **Calculation First:** Manual convolution → Code implementation
- **Visual-Heavy:** Diagrams, animations, feature maps
- **70-20-10 Rule:** 70% explanations, 20% visuals, 10% code

## Critical Content
- Convolution output dimension formula: `(W - F + 2P) / S + 1`
- LeNet-5 as classic example
- Parameter counting (conv vs FC layers)
- Complete CNN pipeline: Input → [Conv → ReLU → Pool] × N → Flatten → FC → Softmax

## Assessment Connection
- **Unit Test 2 (Oct 31):** 9 days after Week 10
- Expected questions: Convolution calculations, architecture design, parameter counting
- Tutorial T10 contributes to FT-IV (continuous assessment)

## Homework Assignment
1. Manual 2D convolution calculation
2. Design CNN for MNIST
3. Modify Tutorial T10 code with experiments

## Directory Structure
```
week10-module4-cnn-basics/
├── do3-oct-22/wip/  (lecture materials)
├── do4-oct-23/wip/  (tutorial materials)
└── week10-plan.md   (this plan document)
```

## Status
- ✅ Directory structure created
- ✅ Week 10 comprehensive plan created
- ⏳ Lecture notes (pending)
- ⏳ Tutorial materials (pending)
- ⏳ Assessment materials (pending)

## Related Memories
- `week9_do4_oct17_cnn_introduction_lecture` - Previous bridge lecture
- `project_overview` - Overall course structure
- `class_schedule_day_order_durations` - Session timing rules
