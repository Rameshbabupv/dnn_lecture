# Answer Key - Question Paper 3 (Formative Test I)
**Course Code**: 21CSE558T  
**Course Title**: Deep Neural Network Architectures  
**Test Duration**: 50 minutes  
**Total Marks**: 25  
**Coverage**: Modules 1-2 (Weeks 1-6)

---

# PART A: Multiple Choice Questions (10 × 1 = 10 marks)

### Q1. The sigmoid activation function outputs values in the range:
**Module**: 1 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: MCQ Q3
a) (-1, 1)  
<span style="color: #008000;">b) (0, 1)</span>  
c) (-∞, ∞)  
d) (0, ∞)  
**Answer**: b) (0, 1)

Justification: Sigmoid outputs probabilities between 0 and 1, whereas tanh spans -1 to 1 and the others extend to infinity.

---

### Q2. Which TensorFlow function is used to create a tensor filled with zeros?
**Module**: 1 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: MCQ Q7
a) tf.ones()  
<span style="color: #008000;">b) tf.zeros()</span>  
c) tf.empty()  
d) tf.fill()  
**Answer**: b) tf.zeros()

Justification: tf.zeros() creates zero-filled tensors directly, unlike tf.ones, tf.empty, or tf.fill.

---

### Q3. The bias term in a neuron:
**Module**: 1 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: MCQ Q14
a) Prevents overfitting  
<span style="color: #008000;">b) Allows shifting of the activation function</span>  
c) Reduces computational complexity  
d) Eliminates the need for weights  
**Answer**: b) Allows shifting of the activation function

Justification: The bias shifts the activation threshold, while the other statements confuse its role with regularization or weight usage.

---

### Q4. TensorFlow operations are executed:
**Module**: 1 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: MCQ Q17
a) Immediately when defined  
<span style="color: #008000;">b) In a computational graph</span>  
c) Only during compilation  
d) Randomly during runtime  
**Answer**: b) In a computational graph

Justification: TensorFlow builds and then executes computation graphs, rather than running operations immediately or randomly.

---

### Q5. Early stopping prevents overfitting by:
**Module**: 2 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: MCQ Q25
<span style="color: #008000;">a) Stopping training when validation loss increases</span>  
b) Reducing the learning rate  
c) Adding regularization terms  
d) Increasing the batch size  
**Answer**: a) Stopping training when validation loss increases

Justification: Early stopping halts when validation performance worsens, unlike adjustments to learning rate, regularization terms, or batch size.

---

### Q6. The dropout rate typically ranges between:
**Module**: 2 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: MCQ Q28
a) 0 to 0.1  
<span style="color: #008000;">b) 0.2 to 0.5</span>  
c) 0.6 to 0.9  
d) 0.9 to 1.0  
**Answer**: b) 0.2 to 0.5

Justification: Practical dropout rates commonly lie between 20% and 50%, with the other ranges being atypically low or high.

---

### Q7. The vanishing gradient problem is most severe with which activation function?
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q31
a) ReLU  
<span style="color: #008000;">b) Sigmoid</span>  
c) Leaky ReLU  
d) Swish  
**Answer**: b) Sigmoid

Justification: Sigmoid saturates at the extremes, causing tiny derivatives and severe vanishing gradients compared to ReLU variants.

---

### Q8. Which optimizer adapts the learning rate for each parameter individually?
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q35
a) SGD  
b) Momentum  
<span style="color: #008000;">c) AdaGrad</span>  
d) Standard gradient descent  
**Answer**: c) AdaGrad

Justification: AdaGrad scales learning rates per parameter based on past gradients, unlike the fixed-rate methods listed.

---

### Q9. The RMSprop optimizer addresses which problem of AdaGrad?
**Module**: 2 | **Difficulty**: Difficult | **CO**: CO-2 | **Source**: MCQ Q42
a) Slow convergence  
b) High memory usage  
<span style="color: #008000;">c) Aggressive learning rate decay</span>  
d) Poor generalization  
**Answer**: c) Aggressive learning rate decay

Justification: RMSprop counteracts AdaGrad’s shrinking learning rates by using an exponential moving average instead of cumulative sums.

---

### Q10. The learning rate finder technique helps to:
**Module**: 2 | **Difficulty**: Difficult | **CO**: CO-2 | **Source**: MCQ Q45
a) Find optimal architecture  
<span style="color: #008000;">b) Determine optimal learning rate range</span>  
c) Prevent overfitting  
d) Reduce training time  
**Answer**: b) Determine optimal learning rate range

Justification: The learning rate finder probes rates to locate a good range, rather than choosing architectures or directly preventing overfitting.

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

### Q14. You switch from batch gradient descent to stochastic gradient descent and observe that training becomes noisier but sometimes achieves better final results. Explain this trade-off.
**Module**: 2 | **Week**: 4 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: 7-SAQ Q5
**Answer**: Stochastic gradient descent updates weights after each sample, creating noisy gradients that cause oscillations during training. However, this noise can be beneficial because it helps the optimizer escape local minima and explore the loss landscape more thoroughly, potentially finding better global solutions that batch gradient descent might miss due to its smoother but more constrained path.

---

### Q15. You observe that a simple model underfits while a complex model overfits the same dataset. Explain this bias-variance trade-off and how it relates to model complexity.
**Module**: 2 | **Week**: 6 | **Difficulty**: Difficult | **CO**: CO-2 | **Source**: 7-SAQ Q7
**Answer**: Simple models have high bias (underfitting) because they lack the capacity to capture complex patterns, while complex models have high variance (overfitting) because they can memorize training data noise. The bias-variance trade-off shows that as model complexity increases, bias decreases but variance increases, requiring careful balance to achieve optimal generalization performance on unseen data.

---

## Distribution Summary:
- **MCQs**: Module 1: 4, Module 2: 6 | Easy: 4, Moderate: 4, Difficult: 2
- **SAQs**: Module 1: 2, Module 2: 3 | Easy: 2, Moderate: 2, Difficult: 1
- **Course Outcomes**: CO-1: 6, CO-2: 9
- **Total Marks**: 25 | **Time**: 50 minutes
