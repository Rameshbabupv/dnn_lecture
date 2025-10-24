# 5-Mark Questions - Deep Neural Network Architectures (21CSE558T)
**Formative Test I - October 10th, 2024**
**Modules 1-2 Coverage (Week 1-6)**

---

## Question 1 (Module 1 - Week 2) [Easy] [CO-1, PO-1, BL3, PI-1.6.1]

You have a single perceptron with the following parameters:
- Input vector: X = [0.8, 0.6]
- Weight vector: W = [0.4, -0.7]
- Bias: b = 0.3
- Activation function: Sigmoid

**Task:** Calculate the complete forward pass output of this perceptron. Show all intermediate steps including weighted sum calculation and activation function application.

---

## Question 2 (Module 1 - Week 3) [Easy] [CO-1, PO-1, BL3, PI-1.6.1]

Consider a simple 2-input perceptron implementing the AND logic gate:
- Input combinations: [0,0], [0,1], [1,0], [1,1]
- Weights: W = [0.5, 0.5]
- Bias: b = -0.7
- Activation: Step function (output 1 if input ≥ 0, else 0)

**Task:** Compute the output for all four input combinations and verify that this perceptron correctly implements the AND gate. Show calculations for each input case.

---

## Question 3 (Module 1 - Week 4) [Moderate] [CO-2, PO-2, BL3, PI-2.6.2]

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

## Question 4 (Module 1 - Week 4) [Moderate] [CO-2, PO-2, BL3, PI-2.6.2]

A neural network is trained using TensorFlow with the following parameters:
- Learning rate: α = 0.01
- Current weight: w = 0.8
- Current bias: b = 0.3
- Training sample: input x = 2.0, target y = 1.0
- Predicted output: ŷ = 0.7
- Loss function: Mean Squared Error (MSE)

**Task:** Calculate one step of gradient descent update for both weight and bias. Show the gradient calculations ∂L/∂w and ∂L/∂b, then compute the updated weight and bias values.

---

## Question 5 (Module 2 - Week 5) [Easy] [CO-2, PO-1, BL3, PI-1.6.1]

Compare the computational efficiency of three activation functions in a neural network:
- Input values: [-2.0, 0.0, 2.0]
- Functions to evaluate: Sigmoid, ReLU, and Tanh

**Task:** Calculate the output of each activation function for all three input values. Show the mathematical formulas used and compute the exact numerical results.

---

## Question 6 (Module 2 - Week 5) [Moderate] [CO-2, PO-2, BL3, PI-2.6.2]

A neural network layer with dropout regularization has the following configuration:
- Input activations: [0.8, 0.6, 0.9, 0.7, 0.5]
- Dropout rate: 0.3 (30% neurons dropped)
- Dropout mask (randomly generated): [1, 0, 1, 1, 0]
- Scaling factor during training: 1/(1-dropout_rate)

**Task:** Apply dropout to the input activations during training phase. Calculate the final output after applying the dropout mask and scaling factor. Show how dropout affects each activation value.

---

## Question 7 (Module 2 - Week 6) [Moderate] [CO-2, PO-2, BL3, PI-2.6.2]

Given a batch normalization layer in TensorFlow with the following statistics:
- Input batch: [2.0, 4.0, 6.0, 8.0]
- Learned parameters: γ (scale) = 1.5, β (shift) = 0.5
- Small constant: ε = 1e-5

**Task:** Perform batch normalization on this input batch. Calculate the batch mean, batch variance, normalized values, and final output after scaling and shifting. Show all intermediate calculations.

---

## Question 8 (Module 1 - Week 3) [Easy] [CO-1, PO-1, BL3, PI-1.6.1]

Design a perceptron to solve the OR logic gate problem:
- Truth table: [0,0]→0, [0,1]→1, [1,0]→1, [1,1]→1
- Given weights: W = [0.6, 0.6]
- Bias: b = -0.2
- Activation: Step function

**Task:** Test this perceptron configuration on all four input combinations of the OR gate. Calculate the weighted sum and final output for each case, and verify the correctness of the implementation.

---

## Question 9 (Module 2 - Week 6) [Moderate] [CO-2, PO-2, BL3, PI-2.6.2]

A neural network uses different optimization algorithms. Compare one iteration of gradient descent vs. momentum-based gradient descent:
- Current gradient: ∇w = 0.4
- Learning rate: α = 0.1
- Current weight: w = 0.8
- For momentum: previous velocity v_prev = 0.2, momentum coefficient β = 0.9

**Task:** Calculate the weight update for both standard gradient descent and momentum-based gradient descent. Show the velocity calculation for momentum method and compare the final weight updates.

---

## Question 10 (Module 2 - Week 6) [Difficult] [CO-2, PO-2, BL3, PI-2.6.2]

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