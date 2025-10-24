# 5-Mark Short Answer Questions (SAQs)
**Course Code**: 21CSE558T | **Course Title**: Deep Neural Network Architectures
**Test**: Formative Assessment I | **Coverage**: Module 1 & Module 2
**Total Questions**: 20 (5-mark explanatory questions)

---

## 📚 Question Bank Overview

This document contains all 20 two-mark short answer questions rewritten in the proper format. All questions are **explanatory/reasoning type** with **paragraph-style answers** based exclusively on content covered in your Week 1-6 lectures.

**⚠️ FORMAT**: Questions ask "Why/What happens/Give reasons" and expect **flowing paragraph answers** with cause-effect reasoning.

---

# 2-MARK SAQ QUESTIONS (20 Questions)

## Module 1: Introduction to Deep Learning (8 questions)

### Q1. You try to train a single perceptron to solve the XOR problem but it fails to converge even after many epochs. Explain why this happens and what fundamental limitation causes this failure.
**Module Coverage**: Module 1 | **Week Coverage**: Week 1-2 | **Lecture**: Perceptron and Boolean Logic
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI1 | **Marks**: 5 | **Difficulty**: Easy

**Expected Answer**:
The XOR problem requires separating points (0,1) and (1,0) from (0,0) and (1,1), which cannot be achieved with a single straight line. A single perceptron can only create linear decision boundaries, but XOR is not linearly separable and requires a non-linear boundary that a single perceptron cannot produce.

---

### Q2. During neural network training, you compute error for individual samples and then average them before updating weights. Explain why this averaging process is necessary and how it relates to the optimization objective.
**Module Coverage**: Module 1 | **Week Coverage**: Week 3-4 | **Lecture**: Forward Pass and Cost Functions
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI4 | **Marks**: 5 | **Difficulty**: Easy

**Expected Answer**:
The loss function measures error for individual training examples, but optimizing each sample separately would cause erratic weight updates. The cost function averages losses across the entire dataset to provide a stable optimization objective, allowing gradient descent to find consistent directions for weight updates that improve overall model performance.

---

### Q3. You build a deep neural network but use only linear activation functions. Despite having multiple layers, the network performs like a simple linear model. Explain why this happens.
**Module Coverage**: Module 1 | **Week Coverage**: Week 3, Day 3 | **Lecture**: Activation Functions Deep Dive
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI3 | **Marks**: 5 | **Difficulty**: Easy

**Expected Answer**:
Linear activation functions cause each layer to perform only linear transformations, and the composition of linear functions is still linear. Without non-linear activation functions, no matter how many layers you stack, the entire network behaves as a single linear transformation, making it impossible to learn complex patterns like XOR.

---

### Q4. You train a neural network without bias terms and notice it struggles to learn patterns where the decision boundary doesn't pass through the origin. Explain why bias is essential for this type of learning.
**Module Coverage**: Module 1 | **Week Coverage**: Week 3 | **Lecture**: Neural Network Layer Mathematics
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL2 | **PI**: CO1-PI1 | **Marks**: 5 | **Difficulty**: Easy

**Expected Answer**:
Without bias, the neuron can only produce outputs based on weighted inputs, forcing the decision boundary to pass through the origin. Bias allows the activation function to shift horizontally, enabling non-zero outputs when all inputs are zero and providing the flexibility needed to learn patterns whose decision boundaries don't pass through the origin.

---

### Q5. You replace sigmoid activations with ReLU in your deep network and observe faster training and better performance. Explain what causes this improvement.
**Module Coverage**: Module 1 | **Week Coverage**: Week 3, Day 3 | **Lecture**: Classical vs Modern Activation Functions
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL3 | **PI**: CO1-PI3 | **Marks**: 5 | **Difficulty**: Moderate

**Expected Answer**:
Sigmoid activations have a maximum derivative of 0.25, causing gradients to vanish exponentially in deep networks, which slows learning in early layers. ReLU has a derivative of 1 for positive inputs, allowing gradients to flow unchanged through many layers, enabling faster training and better performance in deep networks.

---

### Q6. Despite having only one hidden layer, your neural network successfully learns complex non-linear patterns. Explain the theoretical principle that guarantees this capability.
**Module Coverage**: Module 1 | **Week Coverage**: Week 3 | **Lecture**: Mathematical Foundations
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL3 | **PI**: CO1-PI1 | **Marks**: 5 | **Difficulty**: Moderate

**Expected Answer**:
The universal approximation theorem proves that a single hidden layer neural network with sufficient neurons can approximate any continuous function to arbitrary accuracy. This theoretical foundation justifies why neural networks can learn complex patterns and provides the mathematical basis for their problem-solving capabilities, even with relatively simple architectures.

---

### Q7. You observe that in your 10-layer deep network with sigmoid activations, the first few layers learn very slowly while the last layers converge quickly. Explain what causes this learning imbalance.
**Module Coverage**: Module 1 (linking to Module 2) | **Week Coverage**: Week 3 (preview), Week 5 (detailed) | **Lecture**: Gradient Flow and Sigmoid Problems
**PLO**: PLO-1 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI2 | **Marks**: 5 | **Difficulty**: Moderate

**Expected Answer**:
During backpropagation, gradients are computed using the chain rule, which multiplies derivatives from each layer. Since sigmoid's maximum derivative is 0.25, multiplying many small values causes gradients to become exponentially smaller as they propagate backward, resulting in negligible weight updates in early layers while later layers receive strong gradient signals.

---

### Q8. You add batch normalization layers to your deep network and observe that training becomes more stable and you can use higher learning rates. Explain why this technique enables these improvements.
**Module Coverage**: Module 1 (introduced), Module 2 (detailed) | **Week Coverage**: Week 5-6 | **Lecture**: Gradient Solutions and Normalization
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI5 | **Marks**: 5 | **Difficulty**: Moderate

**Expected Answer**:
Batch normalization normalizes layer inputs to maintain stable distributions during training, solving the internal covariate shift problem where input distributions change as previous layers update. This stabilization allows the network to use higher learning rates without diverging and reduces sensitivity to weight initialization, leading to faster and more stable training.

---

## Module 2: Optimization & Regularization (12 questions)

### Q9. You notice that your model performs differently when you change the learning rate versus when you modify the network architecture. Explain why these are fundamentally different types of configuration choices.
**Module Coverage**: Module 2 | **Week Coverage**: Week 4 | **Lecture**: Optimization Problem in Neural Networks
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI1 | **Marks**: 5 | **Difficulty**: Easy

**Expected Answer**:
Learning rate is a hyperparameter that you set before training and remains fixed throughout the optimization process, controlling how the model learns. Network architecture defines parameters (weights and biases) that the model learns during training by adjusting their values based on the data, making them fundamentally different types of configuration choices.

---

### Q10. Your model achieves 99% accuracy on training data but only 65% on test data. Explain what problem this indicates and why it occurs.
**Module Coverage**: Module 2 | **Week Coverage**: Week 6 | **Lecture**: Overfitting vs Underfitting Mastery
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI3 | **Marks**: 5 | **Difficulty**: Easy

**Expected Answer**:
This indicates overfitting, where the model has memorized the training data rather than learning generalizable patterns. The large gap between training and test accuracy occurs because the model has learned to fit noise and specific details in the training set that don't represent the underlying data distribution, causing poor performance on unseen data.

---

### Q11. During training, you randomly set 30% of neurons to zero in each forward pass but use all neurons during testing. Explain the reasoning behind this approach.
**Module Coverage**: Module 2 | **Week Coverage**: Week 6 | **Lecture**: Advanced Regularization Techniques
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI4 | **Marks**: 5 | **Difficulty**: Easy

**Expected Answer**:
This describes dropout regularization, which prevents overfitting by forcing the network to not rely on specific neurons during training. By randomly deactivating neurons, the network learns redundant representations and becomes more robust, while using all neurons during testing provides the full model capacity for optimal performance on new data.

---

### Q12. You implement early stopping and notice training stops at epoch 80 even though you set it to run for 200 epochs. Explain what triggered this behavior and its purpose.
**Module Coverage**: Module 2 | **Week Coverage**: Week 6 | **Lecture**: Overfitting Detection and Analysis
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL2 | **PI**: CO2-PI3 | **Marks**: 5 | **Difficulty**: Easy

**Expected Answer**:
Early stopping monitors validation loss during training and stops when it begins to increase, indicating the onset of overfitting. This automatic termination at epoch 80 occurred because the validation loss stopped decreasing and started rising, suggesting the model was beginning to memorize training data rather than learning generalizable patterns.

---

### Q13. You switch from batch gradient descent to stochastic gradient descent and observe that training becomes noisier but sometimes achieves better final results. Explain this trade-off.
**Module Coverage**: Module 2 | **Week Coverage**: Week 4 | **Lecture**: Gradient Descent Variants Implementation
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Marks**: 5 | **Difficulty**: Moderate

**Expected Answer**:
Stochastic gradient descent updates weights after each sample, creating noisy gradients that cause oscillations during training. However, this noise can be beneficial because it helps the optimizer escape local minima and explore the loss landscape more thoroughly, potentially finding better global solutions that batch gradient descent might miss due to its smoother but more constrained path.

---

### Q14. You add momentum to your gradient descent optimizer and observe faster convergence with fewer oscillations. Explain how momentum achieves these improvements.
**Module Coverage**: Module 2 | **Week Coverage**: Week 4 | **Lecture**: Advanced Optimization Algorithms
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Marks**: 5 | **Difficulty**: Moderate

**Expected Answer**:
Momentum accumulates past gradient directions in an exponentially decaying average, creating velocity that builds up in consistent directions and dampens in oscillating directions. This allows the optimizer to accelerate through flat regions and maintain direction in relevant dimensions while reducing oscillations in irrelevant dimensions, leading to faster and more stable convergence.

---

### Q15. You apply L1 regularization to your network and find that many weights become exactly zero, while L2 regularization only makes weights smaller. Explain what causes this fundamental difference.
**Module Coverage**: Module 2 | **Week Coverage**: Week 6, Day 3 | **Lecture**: L1 vs L2 Regularization Comparison
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI4 | **Marks**: 5 | **Difficulty**: Moderate

**Expected Answer**:
L1 regularization adds the absolute value of weights to the loss (λ∑|w_i|), creating a diamond-shaped constraint that has sharp corners at the axes where weights are exactly zero. L2 regularization uses squared weights (λ∑w_i²), creating a circular constraint that smoothly shrinks weights toward zero but rarely reaches exactly zero, explaining why L1 produces sparsity while L2 produces weight decay.

---

### Q16. You implement the Adam optimizer and notice it performs well across different learning rates compared to standard SGD. Explain what adaptive mechanisms enable this robustness.
**Module Coverage**: Module 2 | **Week Coverage**: Week 4 | **Lecture**: Advanced Optimizers
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Marks**: 5 | **Difficulty**: Moderate

**Expected Answer**:
Adam combines momentum's velocity tracking with RMSprop's adaptive learning rates, maintaining both first-moment (gradient) and second-moment (squared gradient) estimates for each parameter. This allows Adam to automatically adjust effective learning rates for each parameter based on historical gradients, making it less sensitive to the initial learning rate choice and more robust across different optimization landscapes.

---

### Q17. Your deep network suffers from exploding gradients where loss becomes NaN during training. Explain what causes this problem and how gradient clipping addresses it.
**Module Coverage**: Module 2 | **Week Coverage**: Week 5 | **Lecture**: Exploding Gradients Solutions
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI2 | **Marks**: 5 | **Difficulty**: Moderate

**Expected Answer**:
Exploding gradients occur when gradient magnitudes become extremely large through backpropagation, often due to poor weight initialization or high learning rates, causing weights to update by enormous amounts and destabilize training. Gradient clipping addresses this by setting a maximum threshold for gradient norms, scaling down gradients that exceed this limit while preserving their direction, preventing the catastrophic weight updates that cause training instability.

---

### Q18. You observe that a simple model underfits while a complex model overfits the same dataset. Explain this bias-variance trade-off and how it relates to model complexity.
**Module Coverage**: Module 2 | **Week Coverage**: Week 6 | **Lecture**: Bias-Variance Tradeoff Deep Dive
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI3 | **Marks**: 5 | **Difficulty**: Difficult

**Expected Answer**:
Simple models have high bias (underfitting) because they lack the capacity to capture complex patterns, while complex models have high variance (overfitting) because they can memorize training data noise. The bias-variance trade-off shows that as model complexity increases, bias decreases but variance increases, requiring careful balance to achieve optimal generalization performance on unseen data.

---

### Q19. You compare batch, stochastic, and mini-batch gradient descent on the same problem and get different convergence behaviors. Explain how data usage affects each method's characteristics.
**Module Coverage**: Module 2 | **Week Coverage**: Week 4, Day 4 | **Lecture**: Gradient Descent Variants Complete Implementation
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI1 | **Marks**: 5 | **Difficulty**: Difficult

**Expected Answer**:
Batch gradient descent uses the entire dataset for each update, providing stable but slow convergence and requiring high memory. Stochastic gradient descent uses one sample per update, creating noisy but fast updates with low memory requirements. Mini-batch gradient descent balances these trade-offs by using small batches, providing moderate stability and computational efficiency while maintaining reasonable memory usage.

---

### Q20. You initialize weights poorly and observe that gradients either vanish or explode during the first few epochs. Explain why proper weight initialization is crucial for stable training.
**Module Coverage**: Module 2 | **Week Coverage**: Week 5 | **Lecture**: Weight Initialization Strategies
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL4 | **PI**: CO2-PI2 | **Marks**: 5 | **Difficulty**: Difficult

**Expected Answer**:
Poor weight initialization can cause activation and gradient magnitudes to grow or shrink exponentially through the layers during forward and backward passes. Proper initialization methods like Xavier or He ensure that variances of activations and gradients remain stable across layers, preventing the signal from becoming too large (exploding) or too small (vanishing), which is essential for effective gradient flow and stable training dynamics.

---

## Summary Statistics

### Question Distribution:
- **Total Questions**: 20 (All 2-mark explanatory questions)
- **Module 1**: 8 questions (40%)
- **Module 2**: 12 questions (60%)

### Difficulty Distribution:
- **Easy**: 8 questions (40%)
- **Moderate**: 10 questions (50%)
- **Difficult**: 2 questions (10%)

### Question Format Features:
✅ **Scenario-based**: Each question presents a practical situation
✅ **Explanatory**: Ask "Why/What happens/Explain" requiring reasoning
✅ **Paragraph answers**: Flowing sentences with cause-effect relationships
✅ **Lecture-aligned**: Based exclusively on Week 1-6 content
✅ **Technical depth**: Include mathematical reasoning where appropriate

### Course Outcome Distribution:
- **CO-1**: 8 questions (40%)
- **CO-2**: 12 questions (60%)

### Week-wise Coverage Verification:
✅ **Week 1-2**: XOR problem, perceptron limitations
✅ **Week 3**: Activation functions, bias, mathematical foundations
✅ **Week 4**: Gradient descent variants, optimization algorithms
✅ **Week 5**: Gradient problems, initialization, batch normalization
✅ **Week 6**: Regularization, overfitting, bias-variance tradeoff

---

**📧 Note**: All questions rewritten to match proper format with explanatory scenarios and paragraph-style expected answers based exclusively on your Week 1-6 lecture content.