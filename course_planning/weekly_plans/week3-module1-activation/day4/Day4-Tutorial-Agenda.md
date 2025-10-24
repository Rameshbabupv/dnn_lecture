# Day 4 Tutorial Agenda - 1 Hour
**Week 3: Practical Implementation Session**  
**Date:** Aug 26, 2025 | **Duration:** 1 Hour | **Format:** Hands-On Tutorial

---

**© 2025 Prof. Ramesh Babu | SRM University | Data Science and Business Systems (DSBS)**  
*Course Materials for 21CSE558T - Deep Neural Network Architectures*

---

## Tutorial T3: Building Programs to Perform Basic Operations in Tensors

### **Session Overview**
**Primary Focus:** Convert Day 3 theory into practical code implementation  
**Learning Style:** Code → Test → Debug → Understand  
**Tools:** Python, TensorFlow/Keras, NumPy, Jupyter Notebook/Colab

---

## **Pre-Session Setup** (5 minutes)
- **Environment Check:** TensorFlow 2.x, NumPy, Matplotlib installed
- **Platform:** Google Colab recommended for consistency
- **Code Repository:** Download T3 starter templates
- **Quick Review:** Activation function formulas from Day 3

---

## **Detailed Implementation Timeline**

### **Segment 1: Custom Activation Functions** (20 minutes)

#### **Task 1A: Implement Basic Activation Functions** (12 minutes)
```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Students will implement:
def sigmoid(x):
    """Sigmoid activation: σ(x) = 1/(1+e^(-x))"""
    return 1 / (1 + np.exp(-x))
    
def tanh_custom(x):  
    """Hyperbolic tangent: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))"""
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
def relu_custom(x):
    """ReLU activation: max(0,x)"""
    return np.maximum(0, x)
    
def leaky_relu_custom(x, alpha=0.01):
    """Leaky ReLU: x if x>0, alpha*x otherwise"""
    return np.where(x > 0, x, alpha * x)

# Advanced activations (bonus)
def elu_custom(x, alpha=1.0):
    """ELU activation: x if x>0, alpha*(e^x - 1) otherwise"""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def swish_custom(x, beta=1.0):
    """Swish activation: x * sigmoid(beta*x)"""
    return x * sigmoid(beta * x)
```

#### **Task 1B: Gradient Computation** (8 minutes)
```python
# Implement derivatives learned in Day 3
def sigmoid_gradient(x):
    """Sigmoid derivative: σ(x) * (1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)

def tanh_gradient(x):  
    """Tanh derivative: 1 - tanh²(x)"""
    t = tanh_custom(x)
    return 1 - t**2

def relu_gradient(x):
    """ReLU derivative: 1 if x>0, 0 otherwise"""
    return np.where(x > 0, 1.0, 0.0)

def leaky_relu_gradient(x, alpha=0.01):
    """Leaky ReLU derivative: 1 if x>0, alpha otherwise"""
    return np.where(x > 0, 1.0, alpha)

# Visualization of activation functions and their gradients
def plot_activation_and_gradient(func, grad_func, name):
    x = np.linspace(-5, 5, 100)
    y = func(x)
    dy = grad_func(x)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(x, y, 'b-', linewidth=2)
    ax1.set_title(f'{name} Activation')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    
    ax2.plot(x, dy, 'r-', linewidth=2)
    ax2.set_title(f'{name} Gradient')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    
    plt.show()
```

**Learning Checkpoint:** Students can implement and test activation functions

---

### **Segment 2: Tensor Operations & Layer Construction** (25 minutes)

#### **Task 2A: Basic Tensor Operations** (10 minutes)
```python
# Essential tensor manipulations
import tensorflow as tf
import numpy as np

# 1. Create tensors of different dimensions
scalar = tf.constant(5.0)
vector = tf.constant([1, 2, 3], dtype=tf.float32)
matrix = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.float32)

# 2. Matrix multiplication exercises
A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
B = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

# Different multiplication operations
element_wise = tf.multiply(A, B)  # Element-wise multiplication
matrix_mult = tf.matmul(A, B)     # Matrix multiplication
dot_product = tf.tensordot(A, B, axes=1)  # Tensor dot product

# 3. Broadcasting operations
vec = tf.constant([1, 2], dtype=tf.float32)
mat = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)

# Broadcasting vector to matrix
broadcasted_add = mat + vec  # Shape (3,2) + (2,) -> (3,2)
broadcasted_mult = mat * tf.reshape(vec, [1, 2])  # Explicit reshape

# 4. Shape manipulation and reshaping
original = tf.constant([[1, 2, 3], [4, 5, 6]])
reshaped = tf.reshape(original, [3, 2])  # Reshape to 3x2
flattened = tf.reshape(original, [-1])   # Flatten to 1D
transposed = tf.transpose(original)      # Transpose

# Advanced operations
reduced_sum = tf.reduce_sum(mat, axis=1)  # Sum along rows
reduced_mean = tf.reduce_mean(mat, axis=0)  # Mean along columns
expanded = tf.expand_dims(vector, axis=0)  # Add batch dimension

print(f"Original shape: {original.shape}")
print(f"Reshaped: {reshaped.shape}")
print(f"Flattened: {flattened.shape}")
print(f"Transposed: {transposed.shape}")
```

#### **Task 2B: Dense Layer from Scratch** (15 minutes)
```python
# Implement a simple dense layer
class SimpleDenseLayer:
    def __init__(self, input_dim, output_dim, activation='relu'):
        """Initialize a dense layer with weights, bias, and activation"""
        # Xavier/Glorot initialization for weights
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.bias = np.zeros((1, output_dim))
        
        # Activation function selection
        self.activation_name = activation
        self.activation_funcs = {
            'relu': relu_custom,
            'sigmoid': sigmoid,
            'tanh': tanh_custom,
            'leaky_relu': leaky_relu_custom,
            'linear': lambda x: x,
            'softmax': lambda x: np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
        }
        
        self.gradient_funcs = {
            'relu': relu_gradient,
            'sigmoid': sigmoid_gradient,
            'tanh': tanh_gradient,
            'leaky_relu': leaky_relu_gradient,
            'linear': lambda x: np.ones_like(x),
            'softmax': None  # Softmax gradient is handled differently
        }
    
    def forward(self, inputs):
        """Forward pass: output = activation(inputs @ weights + bias)"""
        # Store for backward pass (if needed)
        self.inputs = inputs
        
        # Linear transformation
        self.z = np.dot(inputs, self.weights) + self.bias
        
        # Apply activation
        activation_func = self.activation_funcs.get(self.activation_name, lambda x: x)
        self.output = activation_func(self.z)
        
        return self.output
    
    def backward(self, grad_output, learning_rate=0.01):
        """Backward pass for gradient computation (bonus)"""
        # Gradient of activation function
        grad_func = self.gradient_funcs.get(self.activation_name)
        if grad_func:
            grad_activation = grad_func(self.z)
            grad_z = grad_output * grad_activation
        else:
            grad_z = grad_output
        
        # Gradient w.r.t weights and bias
        grad_weights = np.dot(self.inputs.T, grad_z)
        grad_bias = np.sum(grad_z, axis=0, keepdims=True)
        
        # Gradient w.r.t input (for backpropagation to previous layer)
        grad_input = np.dot(grad_z, self.weights.T)
        
        # Update weights and bias (simple SGD)
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        
        return grad_input
        
    def __call__(self, inputs):
        """Make layer callable"""
        return self.forward(inputs)

# Test the layer
layer = SimpleDenseLayer(input_dim=784, output_dim=128, activation='relu')
test_input = np.random.randn(32, 784)  # Batch of 32 samples
output = layer(test_input)
print(f"Input shape: {test_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Weights shape: {layer.weights.shape}")
print(f"Bias shape: {layer.bias.shape}")
```

**Learning Checkpoint:** Students understand layer mathematics practically

---

### **Segment 3: Complete Neural Network Construction** (10 minutes)

#### **Task 3: Build Multi-Layer Network**
```python
# Combine multiple layers into a complete network
class SimpleNeuralNetwork:
    def __init__(self, layer_sizes, activations):
        """Build a multi-layer neural network
        Args:
            layer_sizes: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
            activations: List of activation functions for each layer
        """
        assert len(layer_sizes) >= 2, "Need at least input and output dimensions"
        assert len(activations) == len(layer_sizes) - 1, "Need activation for each layer"
        
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = SimpleDenseLayer(
                input_dim=layer_sizes[i],
                output_dim=layer_sizes[i+1],
                activation=activations[i]
            )
            self.layers.append(layer)
    
    def forward(self, x):
        """Forward pass through all layers"""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)

# Build and test the network
network = SimpleNeuralNetwork(
    layer_sizes=[784, 128, 64, 10],
    activations=['relu', 'relu', 'softmax']
)

# Test with dummy MNIST-like data
batch_size = 32
test_data = np.random.randn(batch_size, 784)
output = network(test_data)

print(f"Network architecture:")
print(f"  Input: 784 dimensions")
print(f"  Hidden 1: 128 neurons (ReLU)")
print(f"  Hidden 2: 64 neurons (ReLU)")
print(f"  Output: 10 classes (Softmax)")
print(f"\nOutput shape: {output.shape}")
print(f"Output sum per sample (should be ~1 for softmax): {output.sum(axis=1)[:5]}")

# Compare with TensorFlow/Keras equivalent
def build_keras_equivalent():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

keras_model = build_keras_equivalent()
keras_output = keras_model(test_data)

print(f"\nKeras model output shape: {keras_output.shape}")
print(f"Parameter count comparison:")
print(f"  Our implementation: {sum([l.weights.size + l.bias.size for l in network.layers])}")
print(f"  Keras model: {keras_model.count_params()}")
```

**Integration Test:** Forward pass through complete network

---

## **Practical Exercises & Verification**

### **Exercise 1: Activation Function Comparison** (Throughout)
- Plot activation functions and their derivatives
- Test with various input ranges
- Compare custom implementations with TensorFlow built-ins

### **Exercise 2: Gradient Flow Analysis**
- Implement simple gradient computation
- Observe vanishing gradient behavior in deep sigmoid networks
- Compare ReLU vs sigmoid gradient propagation

### **Exercise 3: Layer Parameter Impact**
- Test different weight initialization strategies
- Observe impact of bias terms
- Understand why proper initialization matters

---

## **Code Verification & Testing**

### **Unit Tests for Students:**
```python
def run_unit_tests():
    """Comprehensive testing suite for student implementations"""
    print("Running Unit Tests...\n")
    
    # Test 1: Activation functions
    print("Testing activation functions:")
    assert abs(sigmoid(0) - 0.5) < 1e-6, "Sigmoid(0) should be 0.5"
    assert abs(sigmoid(100) - 1.0) < 0.01, "Sigmoid(large) should approach 1"
    assert abs(sigmoid(-100) - 0.0) < 0.01, "Sigmoid(-large) should approach 0"
    
    assert relu_custom(-1) == 0, "ReLU(-1) should be 0"
    assert relu_custom(1) == 1, "ReLU(1) should be 1"
    
    assert abs(leaky_relu_custom(-1, 0.01) - (-0.01)) < 1e-6, "Leaky ReLU test failed"
    
    assert abs(tanh_custom(0) - 0) < 1e-6, "Tanh(0) should be 0"
    assert abs(tanh_custom(100) - 1.0) < 0.01, "Tanh(large) should approach 1"
    print("✓ Activation functions pass all tests\n")
    
    # Test 2: Gradient functions
    print("Testing gradient functions:")
    assert abs(sigmoid_gradient(0) - 0.25) < 1e-6, "Sigmoid gradient at 0 should be 0.25"
    assert abs(tanh_gradient(0) - 1.0) < 1e-6, "Tanh gradient at 0 should be 1"
    assert relu_gradient(1) == 1, "ReLU gradient for positive should be 1"
    assert relu_gradient(-1) == 0, "ReLU gradient for negative should be 0"
    print("✓ Gradient functions pass all tests\n")
    
    # Test 3: Tensor operations
    print("Testing tensor operations:")
    A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    B = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
    C = tf.matmul(A, B)
    expected = tf.constant([[19, 22], [43, 50]], dtype=tf.float32)
    assert tf.reduce_all(tf.equal(C, expected)), "Matrix multiplication failed"
    print("✓ Tensor operations pass all tests\n")
    
    # Test 4: Layer construction
    print("Testing layer construction:")
    layer = SimpleDenseLayer(10, 5, activation='relu')
    test_input = np.random.randn(2, 10)
    output = layer(test_input)
    assert output.shape == (2, 5), f"Layer output shape mismatch: {output.shape}"
    assert np.all(output >= 0), "ReLU layer should have non-negative outputs"
    print("✓ Layer construction passes all tests\n")
    
    # Test 5: Complete network
    print("Testing complete network:")
    net = SimpleNeuralNetwork([10, 8, 4, 2], ['relu', 'relu', 'softmax'])
    test_input = np.random.randn(3, 10)
    output = net(test_input)
    assert output.shape == (3, 2), f"Network output shape mismatch: {output.shape}"
    assert np.allclose(output.sum(axis=1), 1.0, rtol=1e-5), "Softmax outputs should sum to 1"
    print("✓ Complete network passes all tests\n")
    
    print("\n🎉 All tests passed successfully!")

# Run the tests
run_unit_tests()
```

### **Expected Outputs:**
- Custom activation functions match TensorFlow implementations
- Dense layer produces correct output dimensions
- Complete network processes input correctly

---

## **Troubleshooting & Common Issues**

### **Typical Student Challenges:**
1. **Tensor dimension mismatches** - Review broadcasting rules
2. **Activation function implementation errors** - Check mathematical formulas  
3. **Layer weight initialization problems** - Verify initialization strategies
4. **Import errors** - Environment setup assistance

### **Debugging Strategies:**
- Print tensor shapes at each step
- Verify mathematical computations with small examples
- Compare outputs with TensorFlow/Keras implementations

---

## **Assessment & Submission**

### **Deliverables:**
1. **Working activation functions** with gradients
2. **Custom dense layer implementation**
3. **Complete multi-layer network** 
4. **Test results** showing correctness

### **Evaluation Criteria:**
- **Correctness** (40%): Functions produce expected outputs
- **Code Quality** (30%): Clean, readable, documented code  
- **Understanding** (20%): Can explain implementation choices
- **Testing** (10%): Adequate verification of results

---

## **Connection to Upcoming Content**

### **Module 2 Preparation:**
- Understanding gradient computation prepares for optimization algorithms
- Layer implementation knowledge enables advanced architectures
- Custom activation functions support regularization concepts

### **Next Week Preview:**
- Gradient descent algorithms will use these mathematical foundations
- Optimization techniques will build on gradient computation skills
- Regularization methods will extend activation function concepts

---

## **Resources & Support**

---

## CO-PO Achievement Through Practice

### **Course Outcomes Implementation**
- **CO-1**: Creating simple neural networks with mathematical understanding
- **CO-2**: Building multi-layer networks with appropriate activation selection

### **Programme Outcomes Development**
- **PO-1**: Applied engineering knowledge through tensor mathematics *(Level 3)*
- **PO-2**: Problem analysis in activation function implementation *(Level 2)*  
- **PO-3**: Design solutions through custom layer construction *(Level 1)*

---

## Resources & Support

### **Code References:**
- **TensorFlow Documentation:** Custom layers and activation functions
- **NumPy Documentation:** Array operations and broadcasting  
- **Course Repository:** T3 starter code and solutions

### **Reading Support:**
- **Chollet Ch. 2-3**: Implementation guidance for concepts
- **Manaswi Ch. 2**: TensorFlow practical examples
- **Goodfellow Ch. 6**: Mathematical verification of implementations

### **Help Channels:**
- **In-Session:** Immediate instructor/TA support
- **Post-Session:** Course forum, office hours
- **Peer Support:** Study group formation encouraged

---

## **Success Metrics**

**By End of Session:**
✅ **Functional Code:** All activation functions work correctly  
✅ **Practical Understanding:** Can build layers from mathematical principles  
✅ **Integration Skills:** Can combine components into complete networks  
✅ **Debugging Ability:** Can identify and fix common tensor operation errors  
✅ **Foundation Ready:** Prepared for Module 2 optimization challenges