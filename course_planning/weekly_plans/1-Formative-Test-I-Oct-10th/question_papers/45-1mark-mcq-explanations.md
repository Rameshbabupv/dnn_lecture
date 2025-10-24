# 1-Mark MCQ Comprehensive Answer Key
**Course Code**: 21CSE558T | **Course Title**: Deep Neural Network Architectures
**Test**: Formative Assessment I | **Coverage**: Module 1 & Module 2
**Total Questions**: 45 (1-mark MCQ only)

---

## 📚 Answer Key with Detailed Explanations

*This answer key explains not only why the correct answer is right, but also why each wrong option is incorrect. Use this for deep understanding, not just memorization.*

---

# MODULE 1: INTRODUCTION TO DEEP LEARNING (Questions 1-20)

## Q1. The XOR problem cannot be solved by a single perceptron because:
**Module Coverage**: Module 1 | **Week Coverage**: Week 1-2 | **Lecture**: Perceptron and Boolean Logic

**Correct Answer**: b) It is not linearly separable

### Why b) is CORRECT:
The XOR problem requires separating points (0,1) and (1,0) from (0,0) and (1,1). A single perceptron can only create a linear decision boundary (straight line), but XOR requires a non-linear boundary. This was covered in Week 1-2 when we discussed the limitations of single perceptrons and why they cannot solve non-linearly separable problems.

### Why other options are WRONG:
- **a) It requires too many inputs**: WRONG - XOR uses only 2 inputs, which is perfectly manageable for a perceptron
- **c) It needs multiple outputs**: WRONG - XOR has only 1 output (0 or 1)
- **d) It requires complex activation functions**: WRONG - The issue isn't the activation function, but the linear separability constraint

---

## Q2. In TensorFlow, which data structure is the fundamental building block?
**Module Coverage**: Module 1 | **Week Coverage**: Week 3 | **Tutorial**: T3 - Tensor Operations

**Correct Answer**: c) Tensor

### Why c) is CORRECT:
Tensors are N-dimensional arrays that generalize scalars (0D), vectors (1D), and matrices (2D). All TensorFlow operations work on tensors, hence the name "TensorFlow." This was demonstrated in Tutorial T3 where we performed tensor operations.

### Why other options are WRONG:
- **a) Array**: WRONG - While tensors are array-like, "array" is too generic and not TensorFlow-specific
- **b) Matrix**: WRONG - Matrices are 2D, but tensors can be any dimension (0D, 1D, 2D, 3D, etc.)
- **d) Vector**: WRONG - Vectors are 1D, but tensors encompass all dimensions

---

## Q3. The sigmoid activation function outputs values in the range:
**Module Coverage**: Module 1 | **Week Coverage**: Week 3, Day 3 | **Lecture**: Activation Functions Deep Dive

**Correct Answer**: b) (0, 1)

### Why b) is CORRECT:
Sigmoid function σ(x) = 1/(1 + e^(-x)) asymptotically approaches 0 as x→-∞ and approaches 1 as x→+∞, but never actually reaches these values. This was mathematically proven in Week 3, Day 3 lecture on activation functions.

### Why other options are WRONG:
- **a) (-1, 1)**: WRONG - This is the range of tanh function, not sigmoid
- **c) (-∞, ∞)**: WRONG - This is the range of linear activation or ReLU for positive values
- **d) (0, ∞)**: WRONG - This is the range of ReLU function

---

## Q4. Which loss function is typically used for binary classification?
**Module Coverage**: Module 1 | **Week Coverage**: Week 3 | **Lecture**: Forward Pass and Loss Functions

**Correct Answer**: c) Binary Cross-entropy

### Why c) is CORRECT:
Binary cross-entropy L = -[y×log(ŷ) + (1-y)×log(1-ŷ)] is specifically designed for binary classification where the output is a probability between 0 and 1 (using sigmoid activation). This was covered in Week 3 when discussing loss functions for different problem types.

### Why other options are WRONG:
- **a) Mean Squared Error**: WRONG - Used for regression problems, not classification
- **b) Mean Absolute Error**: WRONG - Also for regression, measures absolute differences
- **d) Categorical Cross-entropy**: WRONG - Used for multi-class classification (>2 classes)

---

## Q5. Which activation function can output negative values?
**Module Coverage**: Module 1 | **Week Coverage**: Week 3, Day 3 | **Lecture**: Activation Functions Properties

**Correct Answer**: c) Tanh

### Why c) is CORRECT:
Tanh function outputs values in range [-1, 1], making it zero-centered unlike sigmoid. The mathematical formula tanh(x) = (e^x - e^(-x))/(e^x + e^(-x)) clearly shows it can produce negative values. This was demonstrated in Week 3 activation functions lecture.

### Why other options are WRONG:
- **a) Sigmoid**: WRONG - Range is (0, 1), always positive
- **b) ReLU**: WRONG - Range is [0, ∞), always non-negative (f(x) = max(0, x))
- **d) Softmax**: WRONG - Outputs probabilities that sum to 1, always positive

---

## Q6. The perceptron learning algorithm can solve:
**Module Coverage**: Module 1 | **Week Coverage**: Week 1-2 | **Lecture**: Perceptron Limitations

**Correct Answer**: a) Only linearly separable problems

### Why a) is CORRECT:
Single perceptron creates a linear decision boundary and can only solve problems where classes can be separated by a straight line (hyperplane in higher dimensions). This fundamental limitation was covered in Week 1-2 and is why we need multi-layer networks.

### Why other options are WRONG:
- **b) Any classification problem**: WRONG - Cannot solve XOR, which is non-linearly separable
- **c) Only regression problems**: WRONG - Perceptron is for classification, not regression
- **d) Non-linear problems directly**: WRONG - This is exactly what perceptron cannot do

---

## Q7. Which TensorFlow function is used to create a tensor filled with zeros?
**Module Coverage**: Module 1 | **Week Coverage**: Week 3 | **Tutorial**: T3 - Tensor Operations

**Correct Answer**: b) tf.zeros()

### Why b) is CORRECT:
tf.zeros(shape) creates a tensor of specified shape filled with zeros. Example: tf.zeros([3, 4]) creates a 3×4 matrix of zeros. This was practiced in Tutorial T3 tensor operations.

### Why other options are WRONG:
- **a) tf.ones()**: WRONG - Creates tensor filled with ones, not zeros
- **c) tf.empty()**: WRONG - Creates uninitialized tensor with random values
- **d) tf.fill()**: WRONG - Requires a value parameter, not specifically for zeros

---

## Q8. Which type of neural network layer performs weighted sum of inputs?
**Module Coverage**: Module 1 | **Week Coverage**: Week 2-3 | **Lecture**: Layer Architecture and Forward Pass

**Correct Answer**: b) Dense layer

### Why b) is CORRECT:
Dense (fully connected) layer computes output = activation(dot(input, weights) + bias), performing weighted sum of all inputs. This mathematical operation was covered in Week 2-3 layer architecture discussions.

### Why other options are WRONG:
- **a) Activation layer**: WRONG - Only applies activation function, doesn't compute weighted sum
- **c) Pooling layer**: WRONG - Reduces spatial dimensions, typically used in CNNs
- **d) Dropout layer**: WRONG - Randomly sets neurons to zero for regularization

---

## Q9. In forward propagation, the output of each layer is:
**Module Coverage**: Module 1 | **Week Coverage**: Week 2-3 | **Lecture**: Forward Pass Mathematics

**Correct Answer**: b) Input to the next layer

### Why b) is CORRECT:
Data flows forward through the network: output of layer i becomes input to layer i+1. This sequential flow was demonstrated mathematically in Week 2-3 forward propagation lectures.

### Why other options are WRONG:
- **a) Input to the previous layer**: WRONG - Data flows forward, not backward
- **c) Stored for backward pass only**: WRONG - While stored for backprop, it's also used for forward flow
- **d) Discarded immediately**: WRONG - Output is needed for next layer and backpropagation

---

## Q10. The softmax activation function is primarily used for:
**Module Coverage**: Module 1 | **Week Coverage**: Week 3 | **Lecture**: Activation Functions Applications

**Correct Answer**: b) Multi-class classification

### Why b) is CORRECT:
Softmax converts raw scores to probabilities that sum to 1, making it perfect for multi-class classification where we need to select one class from multiple options. This was covered in Week 3 activation functions lecture.

### Why other options are WRONG:
- **a) Binary classification**: WRONG - Use sigmoid for binary classification
- **c) Regression problems**: WRONG - Use linear activation for regression
- **d) Feature extraction**: WRONG - Softmax is for final classification, not feature extraction

---

## Q11. Which activation function is most commonly used in hidden layers of modern deep networks?
**Module Coverage**: Module 1 | **Week Coverage**: Week 3 | **Lecture**: Modern Activation Functions

**Correct Answer**: c) ReLU

### Why c) is CORRECT:
ReLU (Rectified Linear Unit) solves the vanishing gradient problem and is computationally efficient, making it the default choice for hidden layers in modern deep networks. This preference was explained in Week 3 activation functions lecture.

### Why other options are WRONG:
- **a) Sigmoid**: WRONG - Causes vanishing gradient problem in deep networks
- **b) Tanh**: WRONG - Better than sigmoid but still has vanishing gradient issues
- **d) Linear**: WRONG - Would make the entire network linear regardless of depth

---

## Q12. The backpropagation algorithm uses which mathematical concept?
**Module Coverage**: Module 1 | **Week Coverage**: Week 3 | **Lecture**: Backpropagation Mathematical Framework

**Correct Answer**: b) Chain rule of differentiation

### Why b) is CORRECT:
Backpropagation applies the chain rule to compute gradients: ∂L/∂w = ∂L/∂y × ∂y/∂z × ∂z/∂w. This mathematical foundation was covered in Week 3 backpropagation framework lecture.

### Why other options are WRONG:
- **a) Integration**: WRONG - Backpropagation uses differentiation, not integration
- **c) Matrix multiplication only**: WRONG - While matrix operations are used, chain rule is the core concept
- **d) Linear algebra only**: WRONG - Linear algebra is used, but chain rule is the key mathematical principle

---

## Q13. The derivative of ReLU function for positive inputs is:
**Module Coverage**: Module 1 | **Week Coverage**: Week 3 | **Lecture**: Activation Function Mathematics

**Correct Answer**: b) 1

### Why b) is CORRECT:
ReLU(x) = max(0, x), so its derivative is 1 for x > 0 and 0 for x < 0. This mathematical property was derived in Week 3 activation functions lecture.

### Why other options are WRONG:
- **a) 0**: WRONG - This is the derivative for negative inputs, not positive
- **c) x**: WRONG - This would be the derivative of x²/2
- **d) e^x**: WRONG - This is the derivative of exponential function

---

## Q14. The bias term in a neuron:
**Module Coverage**: Module 1 | **Week Coverage**: Week 3 | **Lecture**: Neural Network Layer Mathematics

**Correct Answer**: b) Allows shifting of the activation function

### Why b) is CORRECT:
Bias shifts the activation function horizontally, enabling non-zero output when all inputs are zero. This is essential for learning patterns not passing through the origin, as covered in Week 3 layer mathematics.

### Why other options are WRONG:
- **a) Prevents overfitting**: WRONG - Regularization techniques prevent overfitting, not bias
- **c) Reduces computational complexity**: WRONG - Bias adds computation, doesn't reduce it
- **d) Eliminates the need for weights**: WRONG - Both weights and bias are needed

---

## Q15. In a multilayer perceptron, the universal approximation theorem states:
**Module Coverage**: Module 1 | **Week Coverage**: Week 3 | **Lecture**: Mathematical Foundations

**Correct Answer**: b) A single hidden layer can approximate any continuous function

### Why b) is CORRECT:
Proven by Cybenko (1989): a single hidden layer with sufficient neurons can approximate any continuous function to arbitrary accuracy. This theoretical foundation was discussed in Week 3 mathematical foundations.

### Why other options are WRONG:
- **a) Any function can be approximated with infinite layers**: WRONG - The theorem specifically applies to single hidden layer
- **c) Only linear functions can be approximated**: WRONG - The power is in approximating non-linear functions
- **d) Approximation is impossible with finite neurons**: WRONG - Contradicts the theorem

---

## Q16. The mathematical representation of a perceptron output is:
**Module Coverage**: Module 1 | **Week Coverage**: Week 1-2 | **Lecture**: Perceptron Mathematics

**Correct Answer**: b) y = σ(wx + b)

### Why b) is CORRECT:
Perceptron applies activation function σ to the weighted sum of inputs plus bias. This fundamental equation was covered in Week 1-2 perceptron mathematics.

### Why other options are WRONG:
- **a) y = wx + b**: WRONG - Missing activation function (this is just linear combination)
- **c) y = w²x + b**: WRONG - Uses squared weights, not standard perceptron formula
- **d) y = log(wx + b)**: WRONG - Uses logarithm, not typical activation function

---

## Q17. TensorFlow operations are executed:
**Module Coverage**: Module 1 | **Week Coverage**: Week 3 | **Tutorial**: T3 - TensorFlow Execution

**Correct Answer**: b) In a computational graph

### Why b) is CORRECT:
TensorFlow builds a computational graph where operations are nodes and tensors are edges. This graph-based execution was explained in Week 3 TensorFlow basics and Tutorial T3.

### Why other options are WRONG:
- **a) Immediately when defined**: WRONG - This describes eager execution mode, not default behavior
- **c) Only during compilation**: WRONG - Execution happens at runtime, not compilation
- **d) Randomly during runtime**: WRONG - Execution follows the computational graph structure

---

## Q18. The main advantage of using ReLU over sigmoid activation is:
**Module Coverage**: Module 1 | **Week Coverage**: Week 3 | **Lecture**: Activation Function Comparison

**Correct Answer**: c) Mitigation of vanishing gradient problem

### Why c) is CORRECT:
ReLU has derivative = 1 for positive inputs, preventing gradient vanishing that occurs with sigmoid (max derivative = 0.25). This advantage was emphasized in Week 3 activation functions comparison.

### Why other options are WRONG:
- **a) Smoother gradients**: WRONG - ReLU has discontinuous derivative at x=0
- **b) Bounded output range**: WRONG - ReLU is unbounded [0, ∞), sigmoid is bounded (0, 1)
- **d) Better for binary classification**: WRONG - Sigmoid is still preferred for binary output layers

---

## Q19. In TensorFlow, eager execution means:
**Module Coverage**: Module 1 | **Week Coverage**: Week 3 | **Tutorial**: T3 - TensorFlow Execution Modes

**Correct Answer**: a) Operations are executed immediately

### Why a) is CORRECT:
Eager execution evaluates operations immediately without building a computational graph first. This was demonstrated in Week 3 TensorFlow basics and Tutorial T3.

### Why other options are WRONG:
- **b) Operations are cached for later**: WRONG - This describes graph mode, not eager execution
- **c) Operations run in parallel**: WRONG - Parallelization is a separate concept from execution mode
- **d) Operations are optimized automatically**: WRONG - Optimization happens in graph mode

---

## Q20. The vanishing gradient problem primarily affects:
**Module Coverage**: Module 1 (transitioning to Module 2) | **Week Coverage**: Week 3 (preview of Week 5) | **Lecture**: Gradient Flow Preview

**Correct Answer**: b) Deep networks with sigmoid activations

### Why b) is CORRECT:
Sigmoid's maximum derivative is 0.25, so multiplying many sigmoid derivatives in deep networks causes gradients to vanish exponentially. This problem was introduced in Week 3 and detailed in Week 5.

### Why other options are WRONG:
- **a) Shallow networks only**: WRONG - Shallow networks don't have enough layers for vanishing gradients
- **c) Networks with ReLU activations**: WRONG - ReLU (derivative = 1) actually helps prevent vanishing gradients
- **d) Output layer only**: WRONG - The problem affects early layers, not just the output layer

---

# MODULE 2: OPTIMIZATION & REGULARIZATION (Questions 21-45)

## Q21. Which gradient descent variant uses the entire dataset for each update?
**Module Coverage**: Module 2 | **Week Coverage**: Week 4 | **Lecture**: Gradient Descent Variants

**Correct Answer**: c) Batch Gradient Descent

### Why c) is CORRECT:
Batch gradient descent computes gradients using all training samples before making a weight update. This was covered in Week 4, Day 4 gradient descent variants implementation.

### Why other options are WRONG:
- **a) Stochastic Gradient Descent**: WRONG - Uses one sample per update
- **b) Mini-batch Gradient Descent**: WRONG - Uses a small subset of samples
- **d) Adam Optimizer**: WRONG - Adam is an adaptive optimizer, not a gradient descent variant based on batch size

---

## Q22. Learning rate determines:
**Module Coverage**: Module 2 | **Week Coverage**: Week 4 | **Lecture**: Learning Rate Analysis

**Correct Answer**: b) Size of steps toward minimum

### Why b) is CORRECT:
Learning rate α controls how large steps we take in the direction opposite to the gradient: θ := θ - α × ∇J(θ). This was covered in Week 4 learning rate analysis with the "Goldilocks Problem."

### Why other options are WRONG:
- **a) Number of epochs**: WRONG - Epochs are determined by training schedule, not learning rate
- **c) Number of hidden layers**: WRONG - Architecture choice, independent of learning rate
- **d) Batch size**: WRONG - Separate hyperparameter from learning rate

---

## Q23. Overfitting occurs when:
**Module Coverage**: Module 2 | **Week Coverage**: Week 6 | **Lecture**: Overfitting vs Underfitting

**Correct Answer**: c) Model performs well on training but poorly on test data

### Why c) is CORRECT:
Overfitting means the model memorizes training data but fails to generalize to new data, showing high training accuracy but low test accuracy. This was covered in Week 6 bias-variance tradeoff.

### Why other options are WRONG:
- **a) Model performs well on both training and test data**: WRONG - This indicates good generalization
- **b) Model performs poorly on training data**: WRONG - This indicates underfitting
- **d) Model cannot learn from data**: WRONG - This also indicates underfitting or model failure

---

## Q24. Which regularization technique randomly sets some neurons to zero during training?
**Module Coverage**: Module 2 | **Week Coverage**: Week 6 | **Lecture**: Advanced Regularization (mentioned in context)

**Correct Answer**: c) Dropout

### Why c) is CORRECT:
Dropout randomly deactivates neurons during training with probability p (typically 0.2-0.5), preventing co-adaptation and reducing overfitting. This technique was mentioned in Week 6 regularization context.

### Why other options are WRONG:
- **a) L1 regularization**: WRONG - Adds penalty based on absolute weight values
- **b) L2 regularization**: WRONG - Adds penalty based on squared weight values
- **d) Early stopping**: WRONG - Stops training when validation loss increases

---

## Q25. Early stopping prevents overfitting by:
**Module Coverage**: Module 2 | **Week Coverage**: Week 6 | **Lecture**: Overfitting Detection and Prevention

**Correct Answer**: a) Stopping training when validation loss increases

### Why a) is CORRECT:
Early stopping monitors validation loss and stops training when it starts increasing, preventing the model from memorizing training data. This was covered in Week 6 overfitting prevention strategies.

### Why other options are WRONG:
- **b) Reducing the learning rate**: WRONG - This is learning rate scheduling, not early stopping
- **c) Adding regularization terms**: WRONG - This is L1/L2 regularization
- **d) Increasing the batch size**: WRONG - Batch size affects training dynamics, not early stopping

---

## Q26. Underfitting can be reduced by:
**Module Coverage**: Module 2 | **Week Coverage**: Week 6 | **Lecture**: Bias-Variance Tradeoff

**Correct Answer**: c) Increasing model complexity

### Why c) is CORRECT:
Underfitting (high bias) occurs when the model is too simple to capture data patterns. Increasing complexity (more layers, neurons) helps the model learn better. This was covered in Week 6 bias-variance analysis.

### Why other options are WRONG:
- **a) Adding more regularization**: WRONG - This reduces complexity and worsens underfitting
- **b) Reducing model complexity**: WRONG - This would make underfitting worse
- **d) Using smaller datasets**: WRONG - More data typically helps with underfitting

---

## Q27. Which normalization technique normalizes across the batch dimension?
**Module Coverage**: Module 2 | **Week Coverage**: Week 5-6 | **Lecture**: Batch Normalization

**Correct Answer**: c) Batch normalization

### Why c) is CORRECT:
Batch normalization normalizes inputs across the batch dimension, computing mean and variance over the current mini-batch. This was covered in Week 5 gradient solutions and Week 6 regularization.

### Why other options are WRONG:
- **a) Layer normalization**: WRONG - Normalizes across the feature dimension
- **b) Instance normalization**: WRONG - Normalizes each instance independently
- **d) Group normalization**: WRONG - Normalizes within groups of channels

---

## Q28. The dropout rate typically ranges between:
**Module Coverage**: Module 2 | **Week Coverage**: Week 6 | **Lecture**: Practical Regularization Guidelines

**Correct Answer**: b) 0.2 to 0.5

### Why b) is CORRECT:
Typical dropout rates are 0.2-0.5, meaning 20-50% of neurons are randomly deactivated. Higher rates can prevent learning, lower rates provide insufficient regularization. This range was mentioned in Week 6 practical guidelines.

### Why other options are WRONG:
- **a) 0 to 0.1**: WRONG - Too low to provide effective regularization
- **c) 0.6 to 0.9**: WRONG - Too high, would severely impair learning
- **d) 0.9 to 1.0**: WRONG - Would disable most/all neurons, preventing learning

---

## Q29. Which statement about stochastic gradient descent is true?
**Module Coverage**: Module 2 | **Week Coverage**: Week 4 | **Lecture**: SGD Properties

**Correct Answer**: b) Uses one sample for each update

### Why b) is CORRECT:
SGD updates weights after processing each individual training sample, making it noisy but computationally efficient. This was covered in Week 4 gradient descent variants.

### Why other options are WRONG:
- **a) Uses entire dataset for each update**: WRONG - This describes batch gradient descent
- **c) Always converges to global minimum**: WRONG - SGD can get stuck in local minima due to noise
- **d) Requires large memory**: WRONG - SGD is memory efficient since it processes one sample at a time

---

## Q30. The purpose of validation set is to:
**Module Coverage**: Module 2 | **Week Coverage**: Week 6 | **Lecture**: Model Evaluation and Overfitting Detection

**Correct Answer**: b) Tune hyperparameters and monitor overfitting

### Why b) is CORRECT:
Validation set provides unbiased evaluation during training to tune hyperparameters and detect overfitting by monitoring validation loss. This was covered in Week 6 overfitting detection.

### Why other options are WRONG:
- **a) Train the model**: WRONG - Training set is used for training
- **c) Test final performance**: WRONG - Test set is used for final evaluation
- **d) Increase training data**: WRONG - Validation set is separate from training data

---

## Q31. The vanishing gradient problem is most severe with which activation function?
**Module Coverage**: Module 2 | **Week Coverage**: Week 5 | **Lecture**: Vanishing Gradients Analysis

**Correct Answer**: b) Sigmoid

### Why b) is CORRECT:
Sigmoid's derivative σ'(x) = σ(x)(1 - σ(x)) has maximum value of 0.25, causing gradients to shrink exponentially in deep networks. This mathematical analysis was covered in Week 5.

### Why other options are WRONG:
- **a) ReLU**: WRONG - ReLU derivative = 1 for positive inputs, preventing vanishing gradients
- **c) Leaky ReLU**: WRONG - Has small positive derivative for negative inputs, better than sigmoid
- **d) Swish**: WRONG - Modern activation function designed to avoid vanishing gradients

---

## Q32. The Adam optimizer combines:
**Module Coverage**: Module 2 | **Week Coverage**: Week 4 | **Lecture**: Advanced Optimizers

**Correct Answer**: a) Momentum and RMSprop

### Why a) is CORRECT:
Adam (Adaptive Moment Estimation) combines momentum's velocity tracking with RMSprop's adaptive learning rates for each parameter. This combination was covered in Week 4 advanced optimizers.

### Why other options are WRONG:
- **b) SGD and batch gradient descent**: WRONG - These are batch size variants, not what Adam combines
- **c) L1 and L2 regularization**: WRONG - These are regularization techniques, not optimization methods
- **d) Dropout and batch normalization**: WRONG - These are regularization/normalization techniques

---

## Q33. L2 regularization adds which penalty to the loss function?
**Module Coverage**: Module 2 | **Week Coverage**: Week 6 | **Lecture**: L2 Regularization Mathematics

**Correct Answer**: b) Sum of squared weights

### Why b) is CORRECT:
L2 regularization adds λ∑w_i² to the loss function, penalizing large weights quadratically. This mathematical formulation was derived in Week 6 L2 regularization lecture.

### Why other options are WRONG:
- **a) Sum of absolute values of weights**: WRONG - This describes L1 regularization (λ∑|w_i|)
- **c) Product of weights**: WRONG - Not a standard regularization penalty
- **d) Maximum weight value**: WRONG - Not how L2 regularization works

---

## Q34. The exploding gradient problem can be mitigated by:
**Module Coverage**: Module 2 | **Week Coverage**: Week 5 | **Lecture**: Exploding Gradients Solutions

**Correct Answer**: b) Gradient clipping

### Why b) is CORRECT:
Gradient clipping limits gradient norms to a maximum threshold, preventing gradients from becoming too large. This solution was implemented in Week 5 exploding gradients lecture.

### Why other options are WRONG:
- **a) Using smaller learning rates**: WRONG - Helps but doesn't solve the fundamental problem
- **c) Adding more layers**: WRONG - Would make the problem worse
- **d) Using sigmoid activation**: WRONG - Sigmoid causes vanishing gradients, not exploding

---

## Q35. Which optimizer adapts the learning rate for each parameter individually?
**Module Coverage**: Module 2 | **Week Coverage**: Week 4 | **Lecture**: Adaptive Optimizers

**Correct Answer**: c) AdaGrad

### Why c) is CORRECT:
AdaGrad maintains per-parameter learning rates based on historical gradients, adapting each parameter's learning rate individually. This was covered in Week 4 advanced optimizers.

### Why other options are WRONG:
- **a) SGD**: WRONG - Uses fixed learning rate for all parameters
- **b) Momentum**: WRONG - Still uses global learning rate
- **d) Standard gradient descent**: WRONG - Uses same learning rate for all parameters

---

## Q36. The momentum parameter in gradient descent:
**Module Coverage**: Module 2 | **Week Coverage**: Week 4 | **Lecture**: Momentum in Gradient Descent

**Correct Answer**: b) Accelerates convergence in relevant directions

### Why b) is CORRECT:
Momentum accumulates velocity in consistent gradient directions while dampening oscillations, leading to faster convergence. This was demonstrated in Week 4 momentum analysis.

### Why other options are WRONG:
- **a) Increases the learning rate**: WRONG - Momentum is separate from learning rate
- **c) Prevents overfitting**: WRONG - Momentum affects optimization, not regularization
- **d) Normalizes the inputs**: WRONG - This describes normalization techniques

---

## Q37. Batch normalization is applied:
**Module Coverage**: Module 2 | **Week Coverage**: Week 5 | **Lecture**: Batch Normalization Implementation

**Correct Answer**: c) Between layers during training

### Why c) is CORRECT:
Batch normalization normalizes layer inputs during training to maintain stable distributions throughout the network. This was covered in Week 5 gradient solutions.

### Why other options are WRONG:
- **a) Only at the input layer**: WRONG - Can be applied between any layers
- **b) Only at the output layer**: WRONG - Typically not applied to final output
- **d) Only during testing**: WRONG - Applied during training (uses running statistics during testing)

---

## Q38. In mini-batch gradient descent, the batch size affects:
**Module Coverage**: Module 2 | **Week Coverage**: Week 4 | **Lecture**: Mini-batch Analysis

**Correct Answer**: c) Both computational efficiency and gradient noise

### Why c) is CORRECT:
Larger batches are more computationally efficient (better vectorization) but have less gradient noise. Smaller batches have more noise but less efficiency. This tradeoff was covered in Week 4.

### Why other options are WRONG:
- **a) Only computational efficiency**: WRONG - Also affects gradient noise
- **b) Only gradient noise**: WRONG - Also affects computational efficiency
- **d) Neither efficiency nor noise**: WRONG - Batch size affects both

---

## Q39. L1 regularization tends to produce:
**Module Coverage**: Module 2 | **Week Coverage**: Week 6 | **Lecture**: L1 Regularization Properties

**Correct Answer**: b) Sparse weight matrices

### Why b) is CORRECT:
L1 regularization's diamond-shaped constraint drives many weights to exactly zero, creating sparse matrices that perform automatic feature selection. This was demonstrated in Week 6 L1 regularization.

### Why other options are WRONG:
- **a) Dense weight matrices**: WRONG - L1 creates sparsity, not density
- **c) Larger weight values**: WRONG - L1 penalizes large weights
- **d) Negative weight values**: WRONG - L1 doesn't specifically create negative weights

---

## Q40. Which problem is addressed by batch normalization?
**Module Coverage**: Module 2 | **Week Coverage**: Week 5 | **Lecture**: Internal Covariate Shift

**Correct Answer**: b) Internal covariate shift

### Why b) is CORRECT:
Batch normalization addresses internal covariate shift - the change in distribution of layer inputs during training, which slows learning. This concept was explained in Week 5.

### Why other options are WRONG:
- **a) Overfitting only**: WRONG - Primary purpose is not overfitting prevention
- **c) Underfitting only**: WRONG - Not specifically designed to address underfitting
- **d) Memory optimization**: WRONG - BatchNorm actually increases memory usage

---

## Q41. Weight initialization using Xavier/Glorot method aims to:
**Module Coverage**: Module 2 | **Week Coverage**: Week 5 | **Lecture**: Weight Initialization Strategies

**Correct Answer**: b) Maintain gradient flow through layers

### Why b) is CORRECT:
Xavier/Glorot initialization (Var(W) = 2/(n_in + n_out)) maintains variance of activations and gradients across layers, preventing vanishing/exploding gradients. This was covered in Week 5.

### Why other options are WRONG:
- **a) Minimize the loss function**: WRONG - Initialization doesn't directly minimize loss
- **c) Increase computational speed**: WRONG - Doesn't affect computational speed
- **d) Reduce memory usage**: WRONG - Doesn't affect memory usage

---

## Q42. The RMSprop optimizer addresses which problem of AdaGrad?
**Module Coverage**: Module 2 | **Week Coverage**: Week 4 | **Lecture**: Optimizer Comparison

**Correct Answer**: c) Aggressive learning rate decay

### Why c) is CORRECT:
AdaGrad accumulates all past gradients, causing learning rates to decay too aggressively. RMSprop uses exponential moving average to prevent this problem. This was covered in Week 4 optimizer comparison.

### Why other options are WRONG:
- **a) Slow convergence**: WRONG - AdaGrad can converge quickly initially
- **b) High memory usage**: WRONG - Both have similar memory requirements
- **d) Poor generalization**: WRONG - This isn't AdaGrad's main problem

---

## Q43. The bias-variance tradeoff is related to:
**Module Coverage**: Module 2 | **Week Coverage**: Week 6 | **Lecture**: Bias-Variance Decomposition

**Correct Answer**: b) Overfitting and underfitting

### Why b) is CORRECT:
Bias-variance tradeoff: high bias leads to underfitting, high variance leads to overfitting. Total Error = Bias² + Variance + Irreducible Error. This mathematical relationship was covered in Week 6.

### Why other options are WRONG:
- **a) Computational complexity**: WRONG - Not directly related to bias-variance
- **c) Memory usage**: WRONG - Not related to bias-variance tradeoff
- **d) Training time**: WRONG - Not the primary concern of bias-variance analysis

---

## Q44. Which technique can help with both vanishing and exploding gradients?
**Module Coverage**: Module 2 | **Week Coverage**: Week 5 | **Lecture**: Comprehensive Gradient Solutions

**Correct Answer**: b) Residual connections

### Why b) is CORRECT:
Residual connections (skip connections) allow gradients to flow directly through shortcuts, helping with vanishing gradients, while the direct path can also help stabilize exploding gradients. This was covered in Week 5.

### Why other options are WRONG:
- **a) Dropout**: WRONG - Primarily for regularization, not gradient problems
- **c) L2 regularization**: WRONG - For overfitting, not gradient flow
- **d) Early stopping**: WRONG - For overfitting, not gradient problems

---

## Q45. The learning rate finder technique helps to:
**Module Coverage**: Module 2 | **Week Coverage**: Week 4 | **Lecture**: Learning Rate Selection

**Correct Answer**: b) Determine optimal learning rate range

### Why b) is CORRECT:
Learning rate finder gradually increases learning rate and monitors loss to find the optimal range where loss decreases most rapidly. This technique was mentioned in Week 4 learning rate analysis.

### Why other options are WRONG:
- **a) Find optimal architecture**: WRONG - Learning rate finder is only for learning rate selection
- **c) Prevent overfitting**: WRONG - Not its primary purpose
- **d) Reduce training time**: WRONG - While good learning rates help, this isn't the technique's main goal

---

## 📊 Learning Insights Summary

### Key Learning Points:
1. **Module 1 Focus**: Fundamental concepts, activation functions, TensorFlow basics
2. **Module 2 Focus**: Optimization algorithms, regularization techniques, practical solutions

### Study Strategy:
- **Mathematical Understanding**: Know derivations and properties
- **Practical Implementation**: Understand TensorFlow/Keras syntax
- **Problem-Solution Mapping**: Connect problems to appropriate solutions
- **Comparative Analysis**: Understand why certain choices are better than others

This comprehensive answer key helps you understand not just the correct answers, but the reasoning behind each choice, preparing you for deeper understanding and application of deep neural network concepts.