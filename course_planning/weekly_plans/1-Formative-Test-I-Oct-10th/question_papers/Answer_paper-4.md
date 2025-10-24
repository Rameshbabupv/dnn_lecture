# Answer Key - Question Paper 4 - Formative Test I
**Course Code**: 21CSE558T
**Course Title**: Deep Neural Network Architectures
**Test Duration**: 50 minutes
**Total Marks**: 25
**Coverage**: Modules 1-2 (Weeks 1-6)

---

## Instructions:
1. Answer ALL questions in Part A (MCQs)
2. Answer ANY THREE questions from Part B (SAQs)
3. Each MCQ carries 1 mark, each SAQ carries 5 marks
4. Total marks: 25 (Part A: 10 marks + Part B: 15 marks)

---

# PART A: Multiple Choice Questions (10 × 1 = 10 marks)

### Q1. The ReLU activation function is defined as:
**Module**: 1 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: MCQ Q6

<span style="color: #008000;">a) max(0, x)</span>
b) 1/(1 + e^(-x))
c) tanh(x)
d) x/(1 + |x|)

**Answer**: a) max(0, x)

Justification: ReLU is defined as max(0,x), whereas the sigmoid, tanh, and softsign formulas transform inputs differently and do not match this definition.

---

### Q2. Which component is NOT part of the basic perceptron model?
**Module**: 1 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: MCQ Q15

a) Weights
b) Bias
c) Activation function
<span style="color: #008000;">d) Convolutional layer</span>

**Answer**: d) Convolutional layer

Justification: A perceptron uses weights, a bias, and an activation function, while convolutional layers belong to CNNs and are absent in the basic model.

---

### Q3. Mini-batch gradient descent is preferred over batch gradient descent because:
**Module**: 2 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: MCQ Q22

a) It always finds better solutions
<span style="color: #008000;">b) It balances computational efficiency and gradient accuracy</span>
c) It requires no hyperparameter tuning
d) It prevents overfitting

**Answer**: b) It balances computational efficiency and gradient accuracy

Justification: Mini-batches keep updates informative without the full-batch cost, unlike claims that it guarantees better solutions, removes tuning, or inherently stops overfitting.

---

### Q4. L2 regularization (weight decay) helps prevent overfitting by:
**Module**: 2 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: MCQ Q26

a) Increasing the learning rate
<span style="color: #008000;">b) Adding penalty for large weights</span>
c) Reducing the number of epochs
d) Using more training data

**Answer**: b) Adding penalty for large weights

Justification: L2 regularization adds a squared-weight penalty that shrinks large weights, unlike changes to learning rate, epochs, or dataset size.

---

### Q5. A cost function differs from a loss function in that:
**Module**: 1 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: MCQ Q12

a) Cost is for regression, loss is for classification
<span style="color: #008000;">b) Cost is average loss over entire dataset</span>
c) Cost is always squared, loss can be any function
d) There is no difference

**Answer**: b) Cost is average loss over entire dataset

Justification: Cost aggregates losses across all samples, whereas the other statements confuse task type, specific formulas, or claim no distinction.

---

### Q6. The vanishing gradient problem in deep networks is primarily caused by:
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q32

a) Too small learning rates
<span style="color: #008000;">b) Multiplication of small derivatives during backpropagation</span>
c) Large batch sizes
d) Poor data normalization

**Answer**: b) Multiplication of small derivatives during backpropagation

Justification: Repeatedly multiplying small activation derivatives causes gradients to vanish, unlike learning rate settings, batch size, or normalization issues.

---

### Q7. What is the main advantage of using TensorFlow over implementing neural networks from scratch?
**Module**: 1 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: MCQ Q17

a) Better accuracy
<span style="color: #008000;">b) Automatic differentiation and GPU support</span>
c) Smaller model size
d) Faster training guaranteed

**Answer**: b) Automatic differentiation and GPU support

Justification: TensorFlow’s core benefit is graph-based autodiff with accelerator support, not inherent accuracy gains, model compactness, or guaranteed speedups.

---

### Q8. Which initialization strategy is specifically designed for ReLU activations?
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q37

a) Zero initialization
b) Xavier/Glorot initialization
<span style="color: #008000;">c) He initialization</span>
d) Random uniform initialization

**Answer**: c) He initialization

Justification: He initialization maintains variance for ReLU layers, whereas zero, Xavier, or arbitrary uniform schemes mis-handle ReLU’s asymmetric activation.

---

### Q9. Batch normalization helps training by:
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q41

a) Reducing the number of parameters
<span style="color: #008000;">b) Normalizing layer inputs to reduce internal covariate shift</span>
c) Increasing the learning rate automatically
d) Preventing gradient descent

**Answer**: b) Normalizing layer inputs to reduce internal covariate shift

Justification: Batch normalization re-centers and scales activations, whereas the other options describe effects it does not directly provide.

---

### Q10. In the context of neural networks, what does "generalization" refer to?
**Module**: 2 | **Difficulty**: Difficult | **CO**: CO-2 | **Source**: MCQ Q42

a) Ability to work with different programming languages
<span style="color: #008000;">b) Performance on unseen test data</span>
c) Ability to handle different data types
d) Scalability to larger datasets

**Answer**: b) Performance on unseen test data

Justification: Generalization reflects how a trained model handles new data, not programming flexibility, datatype handling, or mere scalability.

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

**Answer**: This indicates overfitting, where the model has memorized the training data rather than learning generalizable patterns. The large gap between training and test accuracy occurs because the model has learned to fit noise and specific details in the training set that don't represent the underlying data distribution, causing poor performance on unseen data.

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
- **MCQs**: Module 1: 4, Module 2: 6 | Easy: 4, Moderate: 5, Difficult: 1
- **SAQs**: Module 1: 2, Module 2: 3 | Easy: 2, Moderate: 2, Difficult: 1
- **Course Outcomes**: CO-1: 6, CO-2: 9
- **Total Marks**: 25 | **Time**: 50 minutes

## Updated SAQ Coverage:
- **XOR Problem**: Perceptron linear separability limitations (Week 1-2)
- **ReLU Benefits**: Vanishing gradient solutions vs sigmoid (Week 3)
- **Overfitting Recognition**: Training-test performance gaps (Week 6)
- **Bias-Variance Trade-off**: Model complexity and generalization (Week 6)
- **SGD vs Batch GD**: Noise advantages in optimization (Week 4)
