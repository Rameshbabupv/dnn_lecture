# Answer Key - Question Paper 1 (Formative Test I)
**Course Code**: 21CSE558T  
**Course Title**: Deep Neural Network Architectures  
**Test Duration**: 50 minutes  
**Total Marks**: 25  
**Coverage**: Modules 1-2 (Weeks 1-6)

---

# PART A: Multiple Choice Questions (10 × 1 = 10 marks)

### Q1. The XOR problem cannot be solved by a single perceptron because:
**Module**: 1 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: MCQ Q1
a) It requires too many inputs  
<span style="color: #008000;">b) It is not linearly separable</span>  
c) It needs multiple outputs  
d) It requires complex activation functions  
**Answer**: b) It is not linearly separable

Justification: XOR fails because a single perceptron can’t separate the classes with one line, whereas the other options describe issues unrelated to linear separability.

---

### Q2. Which activation function is most commonly used in hidden layers of modern deep networks?
**Module**: 1 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: MCQ Q11
a) Sigmoid  
b) Tanh  
<span style="color: #008000;">c) ReLU</span>  
d) Linear  
**Answer**: c) ReLU

Justification: ReLU stays active with a unit slope for positive inputs, avoiding the saturation or limited expressiveness that affects sigmoid, tanh, or linear units.

---

### Q3. The derivative of ReLU function for positive inputs is:
**Module**: 1 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: MCQ Q13
a) 0  
<span style="color: #008000;">b) 1</span>  
c) x  
d) e^x  
**Answer**: b) 1

Justification: The ReLU gradient equals 1 whenever the input is positive, unlike the zero, x-dependent, or exponential values proposed by the other options.

---

### Q4. The main advantage of using ReLU over sigmoid activation is:
**Module**: 1 | **Difficulty**: Difficult | **CO**: CO-1 | **Source**: MCQ Q18
a) Smoother gradients  
b) Bounded output range  
<span style="color: #008000;">c) Mitigation of vanishing gradient problem</span>  
d) Better for binary classification  
**Answer**: c) Mitigation of vanishing gradient problem

Justification: ReLU keeps gradients from collapsing by passing them unchanged for positive activations, unlike smoother or bounded activations that saturate.

---

### Q5. Which gradient descent variant uses the entire dataset for each update?
**Module**: 2 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: MCQ Q21
a) Stochastic Gradient Descent  
b) Mini-batch Gradient Descent  
<span style="color: #008000;">c) Batch Gradient Descent</span>  
d) Adam Optimizer  
**Answer**: c) Batch Gradient Descent

Justification: Batch gradient descent sums losses over the whole dataset each update, while SGD, mini-batch, and Adam use smaller subsets or adaptive schemes.

---

### Q6. Overfitting occurs when:
**Module**: 2 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: MCQ Q23
a) Model performs well on both training and test data  
b) Model performs poorly on training data  
<span style="color: #008000;">c) Model performs well on training but poorly on test data</span>  
d) Model cannot learn from data  
**Answer**: c) Model performs well on training but poorly on test data

Justification: Overfitting is exactly the train-high/test-low gap, whereas the other descriptions correspond to good fit, underfitting, or total failure.

---

### Q7. The purpose of validation set is to:
**Module**: 2 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: MCQ Q30
a) Train the model  
<span style="color: #008000;">b) Tune hyperparameters and monitor overfitting</span>  
c) Test final performance  
d) Increase training data  
**Answer**: b) Tune hyperparameters and monitor overfitting

Justification: Validation sets guide tuning and early stopping, not parameter learning, final evaluation, or enlarging the training pool.

---

### Q8. The Adam optimizer combines:
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q32
<span style="color: #008000;">a) Momentum and RMSprop</span>  
b) SGD and batch gradient descent  
c) L1 and L2 regularization  
d) Dropout and batch normalization  
**Answer**: a) Momentum and RMSprop

Justification: Adam fuses momentum’s velocity term with RMSprop’s adaptive scaling, unlike the mismatched combinations listed in the other choices.

---

### Q9. Batch normalization is applied:
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q37
a) Only at the input layer  
b) Only at the output layer  
<span style="color: #008000;">c) Between layers during training</span>  
d) Only during testing  
**Answer**: c) Between layers during training

Justification: Batch normalization normalizes intermediate activations each training step, rather than operating solely at inputs, outputs, or only during testing.

---

### Q10. Weight initialization using Xavier/Glorot method aims to:
**Module**: 2 | **Difficulty**: Difficult | **CO**: CO-2 | **Source**: MCQ Q41
a) Minimize the loss function  
<span style="color: #008000;">b) Maintain gradient flow through layers</span>  
c) Increase computational speed  
d) Reduce memory usage  
**Answer**: b) Maintain gradient flow through layers

Justification: Xavier keeps activation variance balanced to preserve gradients, which the other options misattribute to loss minimization, runtime, or memory effects.

---

# PART B: Short Answer Questions (Answer any 3 × 5 = 15 marks)

### Q11. You try to train a single perceptron to solve the XOR problem but it fails to converge even after many epochs. Explain why this happens and what fundamental limitation causes this failure.
**Module**: 1 | **Week**: 1-2 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: 7-SAQ Q1
**Answer**: The XOR problem requires separating points (0,1) and (1,0) from (0,0) and (1,1), which cannot be achieved with a single straight line. A single perceptron can only create linear decision boundaries, but XOR is not linearly separable and requires a non-linear boundary that a single perceptron cannot produce.

---

### Q12. You replace sigmoid activations with ReLU in your deep network and observe faster training and better performance. Explain what causes this improvement.
**Module**: 1 | **Week**: 3 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: 7-SAQ Q3
**Answer**: Sigmoid activations have a maximum derivative of 0.25, causing gradients to vanish exponentially in deep networks, which slows learning in early layers. ReLU has a derivative of 1 for positive inputs, allowing gradients to flow unchanged through many layers, enabling faster training and better performance in deep networks.

---

### Q13. Your model achieves 99% accuracy on training data but only 65% on test data. Explain what problem this indicates and why it occurs.
**Module**: 2 | **Week**: 6 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: 7-SAQ Q4
**Answer**: This indicates overfitting, where the model has memorized the training data rather than learning generalizable patterns. The large gap between training and test accuracy occurs because the model has learned to fit noise and specific details in the training set that do not represent the underlying data distribution, causing poor performance on unseen data.

---

### Q14. You observe that a simple model underfits while a complex model overfits the same dataset. Explain this bias-variance trade-off and how it relates to model complexity.
**Module**: 2 | **Week**: 6 | **Difficulty**: Difficult | **CO**: CO-2 | **Source**: 7-SAQ Q7
**Answer**: Simple models have high bias (underfitting) because they lack the capacity to capture complex patterns, while complex models have high variance (overfitting) because they can memorize training data noise. The bias-variance trade-off shows that as model complexity increases, bias decreases but variance increases, requiring careful balance to achieve optimal generalization performance on unseen data.

---

### Q15. You switch from batch gradient descent to stochastic gradient descent and observe that training becomes noisier but sometimes achieves better final results. Explain this trade-off.
**Module**: 2 | **Week**: 4 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: 7-SAQ Q5
**Answer**: Stochastic gradient descent updates weights after each sample, creating noisy gradients that cause oscillations during training. However, this noise can be beneficial because it helps the optimizer escape local minima and explore the loss landscape more thoroughly, potentially finding better global solutions that batch gradient descent might miss due to its smoother but more constrained path.

---

## Distribution Summary:
- **MCQs**: Module 1: 4, Module 2: 6 | Easy: 4, Moderate: 4, Difficult: 2
- **SAQs**: Module 1: 2, Module 2: 3 | Easy: 2, Moderate: 2, Difficult: 1
- **Course Outcomes**: CO-1: 6, CO-2: 9
- **Total Marks**: 25 | **Time**: 50 minutes
