# Question Paper 3 - Formative Test I
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

### Q1. The sigmoid activation function outputs values in the range:
**Module**: 1 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: MCQ Q3

a) (-1, 1)  
b) (0, 1)  
c) (-∞, ∞)  
d) (0, ∞)

---

### Q2. Which TensorFlow function is used to create a tensor filled with zeros?
**Module**: 1 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: MCQ Q7

a) tf.ones()  
b) tf.zeros()  
c) tf.empty()  
d) tf.fill()

---

### Q3. The bias term in a neuron:
**Module**: 1 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: MCQ Q14

a) Prevents overfitting  
b) Allows shifting of the activation function  
c) Reduces computational complexity  
d) Eliminates the need for weights

---

### Q4. TensorFlow operations are executed:
**Module**: 1 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: MCQ Q17

a) Immediately when defined  
b) In a computational graph  
c) Only during compilation  
d) Randomly during runtime

---

### Q5. Early stopping prevents overfitting by:
**Module**: 2 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: MCQ Q25

a) Stopping training when validation loss increases  
b) Reducing the learning rate  
c) Adding regularization terms  
d) Increasing the batch size

---

### Q6. The dropout rate typically ranges between:
**Module**: 2 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: MCQ Q28

a) 0 to 0.1  
b) 0.2 to 0.5  
c) 0.6 to 0.9  
d) 0.9 to 1.0

---

### Q7. The vanishing gradient problem is most severe with which activation function?
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q31

a) ReLU  
b) Sigmoid  
c) Leaky ReLU  
d) Swish

---

### Q8. Which optimizer adapts the learning rate for each parameter individually?
**Module**: 2 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: MCQ Q35

a) SGD  
b) Momentum  
c) AdaGrad  
d) Standard gradient descent

---

### Q9. The RMSprop optimizer addresses which problem of AdaGrad?
**Module**: 2 | **Difficulty**: Difficult | **CO**: CO-2 | **Source**: MCQ Q42

a) Slow convergence  
b) High memory usage  
c) Aggressive learning rate decay  
d) Poor generalization

---

### Q10. The learning rate finder technique helps to:
**Module**: 2 | **Difficulty**: Difficult | **CO**: CO-2 | **Source**: MCQ Q45

a) Find optimal architecture  
b) Determine optimal learning rate range  
c) Prevent overfitting  
d) Reduce training time

---

# PART B: Short Answer Questions (Answer any 3 × 5 = 15 marks)

### Q11. You try to train a single perceptron to solve the XOR problem but it fails to converge even after many epochs. Explain why this happens and what fundamental limitation causes this failure.
**Module**: 1 | **Week**: 1-2 | **Difficulty**: Easy | **CO**: CO-1 | **Source**: 7-SAQ Q1

---

### Q12. You replace sigmoid activations with ReLU in your deep network and observe faster training and better performance. Explain what causes this improvement.
**Module**: 1 | **Week**: 3 | **Difficulty**: Moderate | **CO**: CO-1 | **Source**: 7-SAQ Q3

---

### Q13. Your model achieves 99% accuracy on training data but only 65% on test data. Explain what problem this indicates and why it occurs.
**Module**: 2 | **Week**: 6 | **Difficulty**: Easy | **CO**: CO-2 | **Source**: 7-SAQ Q4

---

### Q14. You switch from batch gradient descent to stochastic gradient descent and observe that training becomes noisier but sometimes achieves better final results. Explain this trade-off.
**Module**: 2 | **Week**: 4 | **Difficulty**: Moderate | **CO**: CO-2 | **Source**: 7-SAQ Q5

---

### Q15. You observe that a simple model underfits while a complex model overfits the same dataset. Explain this bias-variance trade-off and how it relates to model complexity.
**Module**: 2 | **Week**: 6 | **Difficulty**: Difficult | **CO**: CO-2 | **Source**: 7-SAQ Q7

---

## Distribution Summary:
- **MCQs**: Module 1: 4, Module 2: 6 | Easy: 4, Moderate: 4, Difficult: 2
- **SAQs**: Module 1: 2, Module 2: 3 | Easy: 2, Moderate: 2, Difficult: 1
- **Course Outcomes**: CO-1: 6, CO-2: 9
- **Total Marks**: 25 | **Time**: 50 minutes
