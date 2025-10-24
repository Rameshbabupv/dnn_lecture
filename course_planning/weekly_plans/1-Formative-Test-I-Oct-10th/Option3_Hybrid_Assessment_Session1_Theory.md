# Unit Test 1 - Option 3A: Hybrid Assessment - Session 1 (Theory)
**Course**: 21CSE558T - Deep Neural Network Architectures
**Duration**: 60 minutes
**Total Marks**: 50 (out of 100 total)
**Date**: September 19, 2025
**Session**: 1 of 2

---

## Instructions
- This is Session 1 of the hybrid assessment
- Session 2 (Practical) will follow immediately after
- Answer all questions in this session
- Use clear, concise language
- Calculators are not permitted

---

## PART A: MULTIPLE CHOICE QUESTIONS (20 marks)
*Choose the best answer. Each question carries 1 mark.*

**Q1.** The primary cause of vanishing gradients in deep networks is:
- a) Large learning rates
- b) Activation functions with small derivatives
- c) Too many parameters
- d) Insufficient training data

**Q2.** Which activation function has the maximum derivative value of 0.25?
- a) ReLU
- b) Tanh
- c) Sigmoid
- d) Swish

**Q3.** Adam optimizer combines:
- a) SGD and AdaGrad
- b) Momentum and RMSprop
- c) SGD and RMSprop
- d) Momentum and AdaGrad

**Q4.** Batch normalization is typically applied:
- a) After activation function
- b) Before activation function
- c) Only at input layer
- d) Only at output layer

**Q5.** He initialization is designed for:
- a) Sigmoid networks
- b) Tanh networks
- c) ReLU networks
- d) Linear networks

**Q6.** Gradient clipping helps prevent:
- a) Vanishing gradients
- b) Exploding gradients
- c) Overfitting
- d) Underfitting

**Q7.** Skip connections in ResNet address:
- a) Computational complexity
- b) Memory requirements
- c) Gradient flow problems
- d) Overfitting issues

**Q8.** Dropout regularization works by:
- a) Reducing learning rate
- b) Randomly setting neurons to zero
- c) Adding noise to weights
- d) Normalizing activations

**Q9.** The learning rate determines:
- a) Network depth
- b) Batch size
- c) Step size in weight updates
- d) Number of epochs

**Q10.** Layer normalization differs from batch normalization by normalizing:
- a) Across batch dimension
- b) Across feature dimensions
- c) Only input layer
- d) Only output layer

**Q11.** Which is NOT a sign of exploding gradients?
- a) Loss becomes NaN
- b) Weights become very large
- c) Training becomes unstable
- d) Gradients become very small

**Q12.** Xavier/Glorot initialization is optimal for:
- a) ReLU activation
- b) Sigmoid/Tanh activations
- c) Linear activation
- d) Softmax activation

**Q13.** The main benefit of ReLU over sigmoid is:
- a) Bounded output
- b) Smooth function
- c) No vanishing gradient for positive inputs
- d) Probabilistic interpretation

**Q14.** Weight decay is a form of:
- a) Learning rate scheduling
- b) Regularization
- c) Normalization
- d) Initialization

**Q15.** Which optimizer adapts learning rates per parameter?
- a) SGD
- b) Momentum
- c) Adam
- d) All of the above

**Q16.** The saturation region of sigmoid causes:
- a) Exploding gradients
- b) Vanishing gradients
- c) Faster convergence
- d) Better accuracy

**Q17.** Residual connections help by:
- a) Reducing parameters
- b) Providing gradient shortcuts
- c) Increasing depth
- d) Normalizing activations

**Q18.** Early stopping is used to prevent:
- a) Vanishing gradients
- b) Exploding gradients
- c) Overfitting
- d) Slow convergence

**Q19.** The derivative of ReLU for positive inputs is:
- a) 0
- b) 0.25
- c) 0.5
- d) 1

**Q20.** Gradient magnitude < 1e-6 typically indicates:
- a) Healthy gradients
- b) Vanished gradients
- c) Exploded gradients
- d) Optimal learning

---

## PART B: SHORT ANSWER QUESTIONS (30 marks)
*Answer all questions. Be concise but complete.*

**Q21.** Explain why deep sigmoid networks suffer from vanishing gradients. Include the mathematical reason. (6 marks)

**Q22.** Compare and contrast the following activation functions: (8 marks)
- a) Sigmoid vs ReLU (gradient properties)
- b) Tanh vs Sigmoid (output range and zero-centering)

**Q23.** Describe three practical solutions to the vanishing gradient problem. For each solution, briefly explain how it works. (9 marks)

**Q24.** What is the difference between SGD, SGD with momentum, and Adam optimizer? When would you use each? (7 marks)

---

## Answer Sheet Template

**Name**: _________________ **Roll Number**: _________________

### Part A Answers:
| Q | Answer | Q | Answer | Q | Answer | Q | Answer |
|---|--------|---|--------|---|--------|---|--------|
| 1 |        | 6 |        | 11|        | 16|        |
| 2 |        | 7 |        | 12|        | 17|        |
| 3 |        | 8 |        | 13|        | 18|        |
| 4 |        | 9 |        | 14|        | 19|        |
| 5 |        | 10|        | 15|        | 20|        |

### Part B Answers:
Use the space below for your short answer responses.

**Q21.** Vanishing gradients in deep sigmoid networks:

---

**Q22.** Activation function comparisons:
a) Sigmoid vs ReLU:

b) Tanh vs Sigmoid:

---

**Q23.** Three solutions to vanishing gradients:
1. Solution:
   How it works:

2. Solution:
   How it works:

3. Solution:
   How it works:

---

**Q24.** Optimizer comparison:
SGD:

SGD with momentum:

Adam:

When to use each:

---

## Answer Key (For Instructor Use)

### Part A Answers:
1-b, 2-c, 3-b, 4-b, 5-c, 6-b, 7-c, 8-b, 9-c, 10-b, 11-d, 12-b, 13-c, 14-b, 15-c, 16-b, 17-b, 18-c, 19-d, 20-b

### Marking Distribution:
- **Part A**: 1 mark × 20 questions = 20 marks
- **Part B**: Q21(6) + Q22(8) + Q23(9) + Q24(7) = 30 marks
- **Total Session 1**: 50 marks

### Time Allocation Guidance:
- **Part A (MCQ)**: 25 minutes
- **Part B (Short Answer)**: 30 minutes
- **Review**: 5 minutes