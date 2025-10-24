# 7 SAQ Question Bank for Formative Test I (5-mark questions)

## Complete 5-Mark SAQ List with Answers

### Module 1: Introduction to Deep Learning (3 questions)

**Q1.** You try to train a single perceptron to solve the XOR problem but it fails to converge even after many epochs. Explain why this happens and what fundamental limitation causes this failure.
- **Module**: 1 | **Week**: 1-2 | **Difficulty**: Easy | **CO**: CO-1
- **Answer**: The XOR problem requires separating points (0,1) and (1,0) from (0,0) and (1,1), which cannot be achieved with a single straight line. A single perceptron can only create linear decision boundaries, but XOR is not linearly separable and requires a non-linear boundary that a single perceptron cannot produce.

**Q2.** You build a deep neural network but use only linear activation functions. Despite having multiple layers, the network performs like a simple linear model. Explain why this happens.
- **Module**: 1 | **Week**: 3, Day 3 | **Difficulty**: Easy | **CO**: CO-1
- **Answer**: Linear activation functions cause each layer to perform only linear transformations, and the composition of linear functions is still linear. Without non-linear activation functions, no matter how many layers you stack, the entire network behaves as a single linear transformation, making it impossible to learn complex patterns like XOR.

**Q3.** You replace sigmoid activations with ReLU in your deep network and observe faster training and better performance. Explain what causes this improvement.
- **Module**: 1 | **Week**: 3, Day 3 | **Difficulty**: Moderate | **CO**: CO-1
- **Answer**: Sigmoid activations have a maximum derivative of 0.25, causing gradients to vanish exponentially in deep networks, which slows learning in early layers. ReLU has a derivative of 1 for positive inputs, allowing gradients to flow unchanged through many layers, enabling faster training and better performance in deep networks.

### Module 2: Optimization & Regularization (4 questions)

**Q4.** Your model achieves 99% accuracy on training data but only 65% on test data. Explain what problem this indicates and why it occurs.
- **Module**: 2 | **Week**: 6 | **Difficulty**: Easy | **CO**: CO-2
- **Answer**: This indicates overfitting, where the model has memorized the training data rather than learning generalizable patterns. The large gap between training and test accuracy occurs because the model has learned to fit noise and specific details in the training set that don't represent the underlying data distribution, causing poor performance on unseen data.

**Q5.** You switch from batch gradient descent to stochastic gradient descent and observe that training becomes noisier but sometimes achieves better final results. Explain this trade-off.
- **Module**: 2 | **Week**: 4 | **Difficulty**: Moderate | **CO**: CO-2
- **Answer**: Stochastic gradient descent updates weights after each sample, creating noisy gradients that cause oscillations during training. However, this noise can be beneficial because it helps the optimizer escape local minima and explore the loss landscape more thoroughly, potentially finding better global solutions that batch gradient descent might miss due to its smoother but more constrained path.

**Q6.** You apply L1 regularization to your network and find that many weights become exactly zero, while L2 regularization only makes weights smaller. Explain what causes this fundamental difference.
- **Module**: 2 | **Week**: 6, Day 3 | **Difficulty**: Moderate | **CO**: CO-2
- **Answer**: L1 regularization adds the absolute value of weights to the loss (λ∑|w_i|), creating a diamond-shaped constraint that has sharp corners at the axes where weights are exactly zero. L2 regularization uses squared weights (λ∑w_i²), creating a circular constraint that smoothly shrinks weights toward zero but rarely reaches exactly zero, explaining why L1 produces sparsity while L2 produces weight decay.

**Q7.** You observe that a simple model underfits while a complex model overfits the same dataset. Explain this bias-variance trade-off and how it relates to model complexity.
- **Module**: 2 | **Week**: 6 | **Difficulty**: Difficult | **CO**: CO-2
- **Answer**: Simple models have high bias (underfitting) because they lack the capacity to capture complex patterns, while complex models have high variance (overfitting) because they can memorize training data noise. The bias-variance trade-off shows that as model complexity increases, bias decreases but variance increases, requiring careful balance to achieve optimal generalization performance on unseen data.

## Summary
- **Total**: 7 questions (3 from Module 1, 4 from Module 2)
- **Difficulty**: 3 Easy, 3 Moderate, 1 Difficult
- **Course Outcomes**: 3 CO-1, 4 CO-2
- **Format**: All questions expect flowing paragraph answers with cause-effect reasoning