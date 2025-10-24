# Answer Key - Question Paper 2 (Formative Test I)
**Course Code**: 21CSE558T  
**Course Title**: Deep Neural Network Architectures  
**Test Duration**: 50 minutes  
**Total Marks**: 25  
**Coverage**: Modules 1-2 (Weeks 1-6)

---

# PART A: Multiple Choice Questions (10 × 1 = 10 marks)

### Q1. In TensorFlow, which data structure is the fundamental building block?
**Module**: 1 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: MCQ Q2
a) Array  
b) Matrix  
<span style="color: #008000;">c) Tensor</span>  
d) Vector  
**Answer**: c) Tensor

Justification: Tensors generalize scalars, vectors, and matrices and are the native objects TensorFlow manipulates, unlike the narrower structures in the other options.

---

### Q2. Which activation function can output negative values?
**Module**: 1 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: MCQ Q5
a) Sigmoid  
b) ReLU  
<span style="color: #008000;">c) Tanh</span>  
d) Softmax  
**Answer**: c) Tanh

Justification: Tanh outputs values between -1 and 1, permitting negatives, while sigmoid, ReLU, and softmax all stay non-negative.

---

### Q3. The backpropagation algorithm uses which mathematical concept?
**Module**: 1 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: MCQ Q12
a) Integration  
<span style="color: #008000;">b) Chain rule of differentiation</span>  
c) Matrix multiplication only  
d) Linear algebra only  
**Answer**: b) Chain rule of differentiation

Justification: Backprop relies on the chain rule to propagate gradients layer by layer, beyond mere integration or standalone linear algebra operations.

---

### Q4. In a multilayer perceptron, the universal approximation theorem states:
**Module**: 1 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: MCQ Q15
a) Any function can be approximated with infinite layers  
<span style="color: #008000;">b) A single hidden layer can approximate any continuous function</span>  
c) Only linear functions can be approximated  
d) Approximation is impossible with finite neurons  
**Answer**: b) A single hidden layer can approximate any continuous function

Justification: The universal approximation theorem guarantees a one-hidden-layer network can model any continuous function with enough neurons, contrary to the other claims.

---

### Q5. Learning rate determines:
**Module**: 2 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: MCQ Q22
a) Number of epochs  
<span style="color: #008000;">b) Size of steps toward minimum</span>  
c) Number of hidden layers  
d) Batch size  
**Answer**: b) Size of steps toward minimum

Justification: Learning rate scales each gradient step’s magnitude, whereas epochs, layers, or batch size are independent hyperparameters.

---

### Q6. Which regularization technique randomly sets some neurons to zero during training?
**Module**: 2 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: MCQ Q24
a) L1 regularization  
b) L2 regularization  
<span style="color: #008000;">c) Dropout</span>  
d) Early stopping  
**Answer**: c) Dropout

Justification: Dropout randomly zeroes activations during training, unlike L1, L2, or early stopping, which influence weights in other ways.

---

### Q7. L2 regularization adds which penalty to the loss function?
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q33
a) Sum of absolute values of weights  
<span style="color: #008000;">b) Sum of squared weights</span>  
c) Product of weights  
d) Maximum weight value  
**Answer**: b) Sum of squared weights

Justification: L2 adds λ∑w_i² to the loss, penalizing large weights, while L1 uses absolute values and the other choices aren’t standard penalties.

---

### Q8. The momentum parameter in gradient descent:
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q36
a) Increases the learning rate  
<span style="color: #008000;">b) Accelerates convergence in relevant directions</span>  
c) Prevents overfitting  
d) Normalizes the inputs  
**Answer**: b) Accelerates convergence in relevant directions

Justification: Momentum accumulates past gradients to speed movement along consistent directions rather than altering learning rates, regularization, or normalization.

---

### Q9. The bias-variance tradeoff is related to:
**Module**: 2 | **Difficulty**: Difficult | **CO**: CO-2 | **Source**: MCQ Q43
a) Computational complexity  
<span style="color: #008000;">b) Overfitting and underfitting</span>  
c) Memory usage  
d) Training time  
**Answer**: b) Overfitting and underfitting

Justification: Bias-variance trade-off balances underfitting versus overfitting, not computational, memory, or timing concerns.

---

### Q10. Which technique can help with both vanishing and exploding gradients?
**Module**: 2 | **Difficulty**: Difficult | **CO**: CO-2 | **Source**: MCQ Q44
a) Dropout  
<span style="color: #008000;">b) Residual connections</span>  
c) L2 regularization  
d) Early stopping  
**Answer**: b) Residual connections

Justification: Residual skips stabilize gradient flow across deep stacks, whereas dropout, L2, or early stopping target different training issues.

---

# PART B: Short Answer Questions (Answer any 3 × 5 = 15 marks)

### Q11. You build a deep neural network but use only linear activation functions. Despite having multiple layers, the network performs like a simple linear model. Explain why this happens.
**Module**: 1 | **Week**: 3 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: 7-SAQ Q2
**Answer**: Linear activation functions cause each layer to perform only linear transformations, and the composition of linear functions is still linear. Without non-linear activation functions, no matter how many layers you stack, the entire network behaves as a single linear transformation, making it impossible to learn complex patterns like XOR.

---

### Q12. You replace sigmoid activations with ReLU in your deep network and observe faster training and better performance. Explain what causes this improvement.
**Module**: 1 | **Week**: 3 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: 7-SAQ Q3
**Answer**: Sigmoid activations have a maximum derivative of 0.25, causing gradients to vanish exponentially in deep networks, which slows learning in early layers. ReLU has a derivative of 1 for positive inputs, allowing gradients to flow unchanged through many layers, enabling faster training and better performance in deep networks.

---

### Q13. Your model achieves 99% accuracy on training data but only 65% on test data. Explain what problem this indicates and why it occurs.
**Module**: 2 | **Week**: 6 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: 7-SAQ Q4
**Answer**: This indicates overfitting, where the model has memorized the training data rather than learning generalizable patterns. The large gap between training and test accuracy occurs because the model has learned to fit noise and specific details in the training set that do not represent the underlying data distribution, causing poor performance on unseen data.

---

### Q14. You switch from batch gradient descent to stochastic gradient descent and observe that training becomes noisier but sometimes achieves better final results. Explain this trade-off.
**Module**: 2 | **Week**: 4 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: 7-SAQ Q5
**Answer**: Stochastic gradient descent updates weights after each sample, creating noisy gradients that cause oscillations during training. However, this noise can be beneficial because it helps the optimizer escape local minima and explore the loss landscape more thoroughly, potentially finding better global solutions that batch gradient descent might miss due to its smoother but more constrained path.

---

### Q15. You apply L1 regularization to your network and find that many weights become exactly zero, while L2 regularization only makes weights smaller. Explain what causes this fundamental difference.
**Module**: 2 | **Week**: 6 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: 7-SAQ Q6
**Answer**: L1 regularization adds the absolute value of weights to the loss (λ∑|w_i|), creating a diamond-shaped constraint that has sharp corners at the axes where weights are exactly zero. L2 regularization uses squared weights (λ∑w_i²), creating a circular constraint that smoothly shrinks weights toward zero but rarely reaches exactly zero, explaining why L1 produces sparsity while L2 produces weight decay.

---

## Distribution Summary:
- **MCQs**: Module 1: 4, Module 2: 6 | Easy: 4, Moderate: 4, Difficult: 2
- **SAQs**: Module 1: 2, Module 2: 3 | Easy: 2, Moderate: 3, Difficult: 0
- **Course Outcomes**: CO-1: 6, CO-2: 9
- **Total Marks**: 25 | **Time**: 50 minutes
