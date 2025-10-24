# 5-Mark Computational Questions
**Course Code**: 21CSE558T | **Course Title**: Deep Neural Network Architectures
**Test**: Formative Assessment I | **Coverage**: Module 1 & Module 2
**Total Questions**: 10 (5-mark computational questions)

---

## 📚 Question Bank Overview

This document contains all 10 five-mark computational questions for detailed calculations and analysis. All questions require **step-by-step mathematical solutions** and are based exclusively on content covered in your Week 1-6 lectures.

**⚠️ FORMAT**: Questions require mathematical calculations, implementations, and comprehensive analysis with complete working shown.

---

# 5-MARK COMPUTATIONAL QUESTIONS (10 Questions)

## Module 1: Introduction to Deep Learning (4 questions)

### Q1. Perceptron Forward Pass Calculation
**Module Coverage**: Module 1 | **Week Coverage**: Week 2 | **Lecture**: Perceptron Mathematics
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL3 | **PI**: CO1-PI1 | **Marks**: 5 | **Difficulty**: Easy

You have a single perceptron with the following parameters:
- Input vector: X = [0.8, 0.6]
- Weight vector: W = [0.4, -0.7]
- Bias: b = 0.3
- Activation function: Sigmoid

**Task:** Calculate the complete forward pass output of this perceptron. Show all intermediate steps including weighted sum calculation and activation function application.

---

### Q2. AND Gate Perceptron Implementation
**Module Coverage**: Module 1 | **Week Coverage**: Week 3 | **Lecture**: Boolean Logic with Perceptrons
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL3 | **PI**: CO1-PI1 | **Marks**: 5 | **Difficulty**: Easy

Consider a simple 2-input perceptron implementing the AND logic gate:
- Input combinations: [0,0], [0,1], [1,0], [1,1]
- Weights: W = [0.5, 0.5]
- Bias: b = -0.7
- Activation: Step function (output 1 if input ≥ 0, else 0)

**Task:** Compute the output for all four input combinations and verify that this perceptron correctly implements the AND gate. Show calculations for each input case.

---

### Q3. MLP Forward Propagation
**Module Coverage**: Module 1 | **Week Coverage**: Week 4 | **Lecture**: Multi-Layer Perceptron Implementation
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI2 | **Marks**: 5 | **Difficulty**: Moderate

Given a 2-layer MLP with the following TensorFlow/Keras structure:
- Input layer: 2 neurons
- Hidden layer: 3 neurons with ReLU activation
- Output layer: 1 neuron with sigmoid activation
- Input: X = [1.2, -0.8]
- Hidden layer weights: W1 = [[0.5, -0.3, 0.7], [0.2, 0.9, -0.4]]
- Hidden layer bias: b1 = [0.1, -0.2, 0.3]
- Output layer weights: W2 = [[0.6], [-0.5], [0.8]]
- Output layer bias: b2 = [0.2]

**Task:** Perform complete forward propagation through this network. Calculate hidden layer outputs after ReLU activation, then compute the final sigmoid output.

---

### Q4. Gradient Descent Weight Update
**Module Coverage**: Module 1 | **Week Coverage**: Week 4 | **Lecture**: Backpropagation and Weight Updates
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI2 | **Marks**: 5 | **Difficulty**: Moderate

A neural network is trained using TensorFlow with the following parameters:
- Learning rate: α = 0.01
- Current weight: w = 0.8
- Current bias: b = 0.3
- Training sample: input x = 2.0, target y = 1.0
- Predicted output: ŷ = 0.7
- Loss function: Mean Squared Error (MSE)

**Task:** Calculate one step of gradient descent update for both weight and bias. Show the gradient calculations ∂L/∂w and ∂L/∂b, then compute the updated weight and bias values.

---

## Module 2: Optimization & Regularization (6 questions)

### Q5. Activation Functions Comparison
**Module Coverage**: Module 2 | **Week Coverage**: Week 5 | **Lecture**: Activation Function Analysis
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Marks**: 5 | **Difficulty**: Easy

Compare the computational efficiency of three activation functions in a neural network:
- Input values: [-2.0, 0.0, 2.0]
- Functions to evaluate: Sigmoid, ReLU, and Tanh

**Task:** Calculate the output of each activation function for all three input values. Show the mathematical formulas used and compute the exact numerical results.

---

### Q6. Dropout Regularization Application
**Module Coverage**: Module 2 | **Week Coverage**: Week 5 | **Lecture**: Regularization Techniques
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI4 | **Marks**: 5 | **Difficulty**: Moderate

A neural network layer with dropout regularization has the following configuration:
- Input activations: [0.8, 0.6, 0.9, 0.7, 0.5]
- Dropout rate: 0.3 (30% neurons dropped)
- Dropout mask (randomly generated): [1, 0, 1, 1, 0]
- Scaling factor during training: 1/(1-dropout_rate)

**Task:** Apply dropout to the input activations during training phase. Calculate the final output after applying the dropout mask and scaling factor. Show how dropout affects each activation value.

---

### Q7. Batch Normalization Calculation
**Module Coverage**: Module 2 | **Week Coverage**: Week 6 | **Lecture**: Normalization Techniques
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI5 | **Marks**: 5 | **Difficulty**: Moderate

Given a batch normalization layer in TensorFlow with the following statistics:
- Input batch: [2.0, 4.0, 6.0, 8.0]
- Learned parameters: γ (scale) = 1.5, β (shift) = 0.5
- Small constant: ε = 1e-5

**Task:** Perform batch normalization on this input batch. Calculate the batch mean, batch variance, normalized values, and final output after scaling and shifting. Show all intermediate calculations.

---

### Q8. OR Gate Perceptron Design
**Module Coverage**: Module 1 | **Week Coverage**: Week 3 | **Lecture**: Logic Gate Implementation
**PLO**: PLO-1 | **CO**: CO-1 | **BL**: BL3 | **PI**: CO1-PI1 | **Marks**: 5 | **Difficulty**: Easy

Design a perceptron to solve the OR logic gate problem:
- Truth table: [0,0]→0, [0,1]→1, [1,0]→1, [1,1]→1
- Given weights: W = [0.6, 0.6]
- Bias: b = -0.2
- Activation: Step function

**Task:** Test this perceptron configuration on all four input combinations of the OR gate. Calculate the weighted sum and final output for each case, and verify the correctness of the implementation.

---

### Q9. Gradient Descent vs Momentum Comparison
**Module Coverage**: Module 2 | **Week Coverage**: Week 6 | **Lecture**: Advanced Optimization
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI1 | **Marks**: 5 | **Difficulty**: Moderate

A neural network uses different optimization algorithms. Compare one iteration of gradient descent vs. momentum-based gradient descent:
- Current gradient: ∇w = 0.4
- Learning rate: α = 0.1
- Current weight: w = 0.8
- For momentum: previous velocity v_prev = 0.2, momentum coefficient β = 0.9

**Task:** Calculate the weight update for both standard gradient descent and momentum-based gradient descent. Show the velocity calculation for momentum method and compare the final weight updates.

---

### Q10. 3-Layer Network Forward Pass and Loss
**Module Coverage**: Module 2 | **Week Coverage**: Week 6 | **Lecture**: Complete Network Training
**PLO**: PLO-2 | **CO**: CO-2 | **BL**: BL3 | **PI**: CO2-PI2 | **Marks**: 5 | **Difficulty**: Difficult

A 3-layer neural network has the following architecture and is being trained:
- Input layer: 2 neurons
- Hidden layer: 2 neurons with sigmoid activation
- Output layer: 1 neuron with sigmoid activation
- Input: X = [1.0, 0.5]
- Target: y = 0.8
- Current weights and biases:
  - W1 = [[0.3, 0.7], [0.2, 0.8]], b1 = [0.1, 0.2]
  - W2 = [[0.6], [0.4]], b2 = [0.3]

**Task:** Perform complete forward propagation to get the prediction, calculate the MSE loss, then compute the error gradient at the output layer (∂L/∂output). Show all calculations including intermediate activations and loss computation.

---

## Summary Statistics

### Question Distribution:
- **Total Questions**: 10 (All 5-mark computational questions)
- **Module 1**: 4 questions (40%)
- **Module 2**: 6 questions (60%)

### Difficulty Distribution:
- **Easy**: 4 questions (40%)
- **Moderate**: 5 questions (50%)
- **Difficult**: 1 question (10%)

### Course Outcome Distribution:
- **CO-1**: 4 questions (40%)
- **CO-2**: 6 questions (60%)

---

**📧 Note**: All questions require complete mathematical working and step-by-step solutions based exclusively on your Week 1-6 lecture content.