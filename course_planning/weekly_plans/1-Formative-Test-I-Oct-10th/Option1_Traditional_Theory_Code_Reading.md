# Unit Test 1 - Option 1: Traditional Theory + Code Reading
**Course**: 21CSE558T - Deep Neural Network Architectures
**Duration**: 90 minutes
**Total Marks**: 100
**Date**: September 19, 2025

---

## Instructions
- Answer all questions
- Part A: Multiple Choice Questions (20 marks)
- Part B: Short Answer Questions (40 marks)
- Part C: Code Analysis Questions (40 marks)
- Use of calculators is not permitted

---

## PART A: MULTIPLE CHOICE QUESTIONS (20 marks)
*Choose the best answer. Each question carries 2 marks.*

**Q1.** The maximum value of the derivative of the sigmoid function is:
- a) 1.0
- b) 0.5
- c) 0.25
- d) 0.1

**Q2.** Which activation function is most prone to the vanishing gradient problem?
- a) ReLU
- b) Sigmoid
- c) Tanh
- d) Leaky ReLU

**Q3.** In TensorFlow, which function is used to compute gradients?
- a) tf.gradient()
- b) tf.GradientTape()
- c) tf.compute_gradients()
- d) tf.auto_gradient()

**Q4.** The Adam optimizer combines which two optimization techniques?
- a) SGD and Momentum
- b) RMSprop and Momentum
- c) AdaGrad and RMSprop
- d) SGD and AdaGrad

**Q5.** Batch normalization is typically applied:
- a) Before activation function
- b) After activation function
- c) Only at input layer
- d) Only at output layer

**Q6.** Which weight initialization method is suitable for ReLU networks?
- a) Xavier/Glorot initialization
- b) He initialization
- c) Zero initialization
- d) Random normal initialization

**Q7.** Dropout regularization helps prevent:
- a) Vanishing gradients
- b) Exploding gradients
- c) Overfitting
- d) Underfitting

**Q8.** The learning rate in gradient descent determines:
- a) Number of epochs
- b) Batch size
- c) Step size in parameter updates
- d) Network architecture

**Q9.** Skip connections in ResNet help address:
- a) Overfitting problem
- b) Vanishing gradient problem
- c) Computational complexity
- d) Memory requirements

**Q10.** Which metric is NOT typically used for classification problems?
- a) Accuracy
- b) Precision
- c) Mean Squared Error
- d) F1-Score

---

## PART B: SHORT ANSWER QUESTIONS (40 marks)
*Answer all questions. Show your work where applicable.*

**Q11.** Explain the vanishing gradient problem in deep neural networks. Why does it occur with sigmoid activation functions? (8 marks)

**Q12.** Compare ReLU and Sigmoid activation functions in terms of:
- a) Gradient flow properties (3 marks)
- b) Computational efficiency (2 marks)
- c) Output range (3 marks)

**Q13.** Describe three different methods to address the exploding gradient problem. Explain how each method works. (8 marks)

**Q14.** What is the purpose of normalization techniques in deep learning? Briefly explain batch normalization and its benefits. (8 marks)

**Q15.** Explain the difference between SGD, SGD with momentum, and Adam optimizer. When would you choose each? (8 marks)

---

## PART C: CODE ANALYSIS QUESTIONS (40 marks)
*Analyze the given code snippets and answer the questions.*

**Q16.** Analyze this TensorFlow code snippet: (15 marks)

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='sigmoid', input_shape=(784,)),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dense(32, activation='sigmoid'),
    tf.keras.layers.Dense(16, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='sgd', loss='categorical_crossentropy')
```

a) Identify two potential problems with this network architecture. (6 marks)
b) Suggest specific improvements to address these problems. (6 marks)
c) Rewrite the model definition with your improvements. (3 marks)

**Q17.** Given this gradient computation code: (12 marks)

```python
with tf.GradientTape() as tape:
    predictions = model(X_batch)
    loss = tf.reduce_mean(tf.square(predictions - y_batch))

gradients = tape.gradient(loss, model.trainable_variables)
```

a) What type of loss function is being used? Is it appropriate for classification? (4 marks)
b) How would you modify this code to monitor gradient magnitudes? (4 marks)
c) Write code to implement gradient clipping with threshold 1.0. (4 marks)

**Q18.** Network Architecture Analysis: (13 marks)

```python
# Network A
model_a = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Network B
model_b = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dense(32, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

a) Which network is more likely to suffer from vanishing gradients? Explain why. (5 marks)
b) For a binary classification task, which network would you recommend? (4 marks)
c) Suggest one improvement for each network. (4 marks)

---

## ANSWER KEY (For Instructor Use)

### Part A Answers:
1. c) 0.25, 2. b) Sigmoid, 3. b) tf.GradientTape(), 4. b) RMSprop and Momentum, 5. a) Before activation function, 6. b) He initialization, 7. c) Overfitting, 8. c) Step size in parameter updates, 9. b) Vanishing gradient problem, 10. c) Mean Squared Error

### Marking Scheme:
- **Part A**: 2 marks per question × 10 = 20 marks
- **Part B**: Variable marks as indicated, total 40 marks
- **Part C**: Variable marks as indicated, total 40 marks
- **Total**: 100 marks

### Grade Distribution:
- **90-100**: Excellent understanding
- **80-89**: Good understanding
- **70-79**: Satisfactory understanding
- **60-69**: Basic understanding
- **Below 60**: Needs improvement