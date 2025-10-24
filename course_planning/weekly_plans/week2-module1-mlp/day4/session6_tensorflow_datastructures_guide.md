# Week 2 - Session 6: Data Structures in TensorFlow
## Neural Network Building Blocks - Lab Session

---

## 🎯 **Session Overview**

**Session Number**: 6  
**Week**: 2 (August 18-22, 2025)  
**Duration**: 1 hour  
**Format**: Lab Session  
**Tutorial Task**: T3 - Building Programs Basic Operations in Tensors  
**Assessment**: Implementation assessment

---

## 📋 **Learning Objectives**

By the end of this session, students will be able to:
- Understand fundamental TensorFlow data structures
- Work with tensors, variables, and constants
- Implement basic tensor operations for neural networks
- Apply tensor manipulation in neural network contexts
- Debug and troubleshoot tensor operations

---

## 🔗 **Connection to Previous Sessions**

### **Building on Session 5**
- ✅ TensorFlow environment setup
- ✅ Basic tensor creation and manipulation
- ✅ Understanding tensor properties (shape, dtype, rank)

### **Today's Focus**
- 🎯 **Advanced tensor operations** for neural network building blocks
- 🎯 **Data structures** that form the foundation of neural networks
- 🎯 **Practical implementation** of concepts learned in Sessions 4-5

---

## 🏗️ **Part 1: TensorFlow Data Structures Foundation (15 minutes)**

### **1.1 Core TensorFlow Objects**

#### **Tensors - The Building Blocks**
```python
import tensorflow as tf
import numpy as np

# Different ways to create tensors
scalar = tf.constant(42)                    # 0D tensor (scalar)
vector = tf.constant([1, 2, 3, 4])         # 1D tensor (vector)
matrix = tf.constant([[1, 2], [3, 4]])     # 2D tensor (matrix)
tensor_3d = tf.zeros((2, 3, 4))            # 3D tensor

print(f"Scalar: {scalar}, shape: {scalar.shape}, rank: {tf.rank(scalar)}")
print(f"Vector: {vector}, shape: {vector.shape}, rank: {tf.rank(vector)}")
print(f"Matrix: {matrix}, shape: {matrix.shape}, rank: {tf.rank(matrix)}")
```

#### **Variables vs Constants**
```python
# Constants - immutable
weights_const = tf.constant([[0.1, 0.2], [0.3, 0.4]])

# Variables - mutable (used for trainable parameters)
weights_var = tf.Variable([[0.1, 0.2], [0.3, 0.4]], name="weights")
bias_var = tf.Variable([0.1, 0.2], name="bias")

print(f"Constant: {weights_const}")
print(f"Variable: {weights_var}")
print(f"Variable name: {weights_var.name}")
```

### **1.2 Data Types and Shapes**

#### **Understanding Data Types**
```python
# Common data types in neural networks
float_tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
int_tensor = tf.constant([1, 2, 3], dtype=tf.int32)
bool_tensor = tf.constant([True, False, True], dtype=tf.bool)

# Type conversion
converted = tf.cast(int_tensor, tf.float32)
print(f"Original: {int_tensor.dtype}, Converted: {converted.dtype}")
```

#### **Shape Manipulation**
```python
# Shape operations crucial for neural networks
original = tf.constant([[1, 2, 3], [4, 5, 6]])
print(f"Original shape: {original.shape}")

# Reshape for different layer inputs
reshaped = tf.reshape(original, (3, 2))
flattened = tf.reshape(original, (-1,))  # Flatten to 1D
print(f"Reshaped: {reshaped.shape}, Flattened: {flattened.shape}")
```

---

## 🧮 **Part 2: Essential Tensor Operations for Neural Networks (20 minutes)**

### **2.1 Mathematical Operations**

#### **Element-wise Operations**
```python
# Basic arithmetic operations
a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
b = tf.constant([[2, 1], [4, 3]], dtype=tf.float32)

# Element-wise operations
addition = tf.add(a, b)        # or a + b
subtraction = tf.subtract(a, b) # or a - b
multiplication = tf.multiply(a, b) # or a * b
division = tf.divide(a, b)     # or a / b

print(f"Addition:\n{addition}")
print(f"Element-wise multiplication:\n{multiplication}")
```

#### **Matrix Operations (Neural Network Core)**
```python
# Matrix multiplication (crucial for neural networks)
inputs = tf.constant([[1.0, 2.0, 3.0]])  # 1x3 input
weights = tf.constant([[0.1, 0.4],       # 3x2 weight matrix
                       [0.2, 0.5],
                       [0.3, 0.6]])

# Forward pass calculation: y = X * W
output = tf.matmul(inputs, weights)
print(f"Neural network forward pass: {output}")

# Transpose operations
weights_t = tf.transpose(weights)
print(f"Original weights shape: {weights.shape}")
print(f"Transposed weights shape: {weights_t.shape}")
```

### **2.2 Activation Function Implementation**

#### **Common Activation Functions**
```python
# Input values for testing activations
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])

# ReLU activation
relu_output = tf.nn.relu(x)
print(f"ReLU: {relu_output}")

# Sigmoid activation
sigmoid_output = tf.nn.sigmoid(x)
print(f"Sigmoid: {sigmoid_output}")

# Tanh activation
tanh_output = tf.nn.tanh(x)
print(f"Tanh: {tanh_output}")

# Softmax (for classification)
logits = tf.constant([2.0, 1.0, 0.1])
softmax_output = tf.nn.softmax(logits)
print(f"Softmax: {softmax_output}, Sum: {tf.reduce_sum(softmax_output)}")
```

### **2.3 Reduction Operations**

#### **Aggregation Functions**
```python
# Data for reduction operations
data = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)

# Common reductions
sum_all = tf.reduce_sum(data)           # Sum all elements
sum_rows = tf.reduce_sum(data, axis=1)  # Sum along rows
sum_cols = tf.reduce_sum(data, axis=0)  # Sum along columns

mean_val = tf.reduce_mean(data)
max_val = tf.reduce_max(data)
min_val = tf.reduce_min(data)

print(f"Sum all: {sum_all}")
print(f"Sum rows: {sum_rows}")
print(f"Sum columns: {sum_cols}")
print(f"Mean: {mean_val}, Max: {max_val}, Min: {min_val}")
```

---

## 🔨 **Part 3: Tutorial T3 - Building Programs with Basic Operations (20 minutes)**

### **3.1 Mini Neural Network Implementation**

#### **Complete Forward Pass Example**
```python
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases as Variables
        self.W1 = tf.Variable(tf.random.normal([input_size, hidden_size], stddev=0.1), name="W1")
        self.b1 = tf.Variable(tf.zeros([hidden_size]), name="b1")
        self.W2 = tf.Variable(tf.random.normal([hidden_size, output_size], stddev=0.1), name="W2")
        self.b2 = tf.Variable(tf.zeros([output_size]), name="b2")
    
    def forward_pass(self, inputs):
        # Hidden layer calculation
        hidden = tf.matmul(inputs, self.W1) + self.b1
        hidden_activated = tf.nn.relu(hidden)
        
        # Output layer calculation
        outputs = tf.matmul(hidden_activated, self.W2) + self.b2
        return outputs
    
    def print_parameters(self):
        print(f"W1 shape: {self.W1.shape}, W2 shape: {self.W2.shape}")
        print(f"b1 shape: {self.b1.shape}, b2 shape: {self.b2.shape}")

# Create and test the network
nn = SimpleNeuralNetwork(input_size=3, hidden_size=4, output_size=2)
nn.print_parameters()

# Test with sample data
sample_input = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
output = nn.forward_pass(sample_input)
print(f"Network output: {output}")
```

### **3.2 XOR Problem Implementation**

#### **Solving XOR with TensorFlow Data Structures**
```python
# XOR problem data
X = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float32)
y = tf.constant([[0], [1], [1], [0]], dtype=tf.float32)

# Pre-trained weights for XOR solution
W1 = tf.Variable([[10.0, -10.0], [10.0, -10.0]], name="hidden_weights")
b1 = tf.Variable([-5.0, 15.0], name="hidden_bias")
W2 = tf.Variable([[10.0], [10.0]], name="output_weights")
b2 = tf.Variable([-15.0], name="output_bias")

def xor_network(inputs):
    # Hidden layer
    hidden = tf.matmul(inputs, W1) + b1
    hidden_activated = tf.nn.sigmoid(hidden)
    
    # Output layer
    output = tf.matmul(hidden_activated, W2) + b2
    output_activated = tf.nn.sigmoid(output)
    
    return hidden_activated, output_activated

# Test XOR network
hidden_vals, predictions = xor_network(X)
print("XOR Results:")
for i in range(4):
    print(f"Input: {X[i].numpy()}, Hidden: {hidden_vals[i].numpy()}, Output: {predictions[i].numpy()}")
```

### **3.3 Data Preprocessing Operations**

#### **Common Data Preparation Tasks**
```python
# Sample dataset preparation
raw_data = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=tf.float32)

# Normalization (zero mean, unit variance)
mean = tf.reduce_mean(raw_data, axis=0)
std = tf.math.reduce_std(raw_data, axis=0)
normalized_data = (raw_data - mean) / std

print(f"Original data:\n{raw_data}")
print(f"Mean: {mean}, Std: {std}")
print(f"Normalized data:\n{normalized_data}")

# Min-Max scaling
min_val = tf.reduce_min(raw_data, axis=0)
max_val = tf.reduce_max(raw_data, axis=0)
scaled_data = (raw_data - min_val) / (max_val - min_val)

print(f"Min-Max scaled data:\n{scaled_data}")

# One-hot encoding for categorical data
labels = tf.constant([0, 1, 2, 1, 0])
one_hot = tf.one_hot(labels, depth=3)
print(f"Labels: {labels}")
print(f"One-hot encoded:\n{one_hot}")
```

---

## 💡 **Part 4: Practical Exercises and Debugging (5 minutes)**

### **4.1 Common Debugging Techniques**

#### **Shape Debugging**
```python
def debug_tensor_shapes(tensor, name):
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Rank: {tf.rank(tensor)}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Size: {tf.size(tensor)}")
    print()

# Debug example tensors
input_data = tf.random.normal([32, 784])  # Batch of flattened MNIST images
debug_tensor_shapes(input_data, "Input Data")

weights = tf.random.normal([784, 128])
debug_tensor_shapes(weights, "Hidden Layer Weights")
```

#### **Value Inspection**
```python
# Check for common issues
problematic_tensor = tf.constant([1.0, float('inf'), -float('inf'), float('nan')])

print(f"Has inf: {tf.reduce_any(tf.math.is_inf(problematic_tensor))}")
print(f"Has nan: {tf.reduce_any(tf.math.is_nan(problematic_tensor))}")
print(f"All finite: {tf.reduce_all(tf.math.is_finite(problematic_tensor[:-3]))}")
```

---

## 📝 **Assessment Criteria**

### **Implementation Assessment Rubric**

**Excellent (90-100%):**
- Correct implementation of all tensor operations
- Proper use of TensorFlow data structures
- Clean, well-commented code
- Successful neural network forward pass
- Effective debugging and error handling

**Good (80-89%):**
- Most tensor operations implemented correctly
- Minor issues with data structure usage
- Code mostly functional with some improvements needed

**Satisfactory (70-79%):**
- Basic tensor operations working
- Some conceptual understanding demonstrated
- Code runs but may have logical errors

**Needs Improvement (<70%):**
- Significant errors in implementation
- Limited understanding of TensorFlow concepts
- Code doesn't execute properly

### **Specific Assessment Points**
1. **Tensor Creation**: Proper use of constants, variables, and different data types
2. **Shape Management**: Correct reshaping and dimension handling
3. **Mathematical Operations**: Accurate implementation of matrix operations
4. **Neural Network Components**: Working forward pass implementation
5. **Code Quality**: Readable, commented, and debuggable code

---

## 🔄 **Session Summary**

### **Key Accomplishments**
✅ Mastered TensorFlow data structures (tensors, variables, constants)  
✅ Implemented essential tensor operations for neural networks  
✅ Built complete mini neural network using TensorFlow  
✅ Solved XOR problem with proper tensor operations  
✅ Learned debugging techniques for tensor operations  

### **Skills Developed**
- Tensor manipulation and shape handling
- Matrix operations for neural network computations
- Activation function implementation
- Data preprocessing with TensorFlow
- Debugging and troubleshooting tensor code

### **Real-world Applications**
*"These tensor operations form the foundation of every modern deep learning model - from simple MLPs to advanced transformers!"*

---

## 🚀 **Preparation for Next Session**

### **Week 3 Preview: Activation Functions**
- Understanding the need for non-linearity
- Mathematical properties of different activations
- Choosing appropriate activation functions
- Impact on network training and performance

### **Recommended Practice**
- Experiment with different tensor shapes and operations
- Try implementing other simple neural network architectures
- Practice debugging tensor dimension mismatches
- Explore TensorFlow documentation for additional operations

---

## 📚 **Additional Resources**

### **TensorFlow Documentation**
- [Tensor Operations Guide](https://www.tensorflow.org/guide/tensor)
- [Variables Guide](https://www.tensorflow.org/guide/variable)
- [Math Operations](https://www.tensorflow.org/api_docs/python/tf/math)

### **Practice Datasets**
- Use small synthetic datasets for practicing tensor operations
- MNIST digit dataset for more complex examples
- XOR and logic gate datasets for boolean operations

---

## 🎯 **Instructor Notes**

### **Teaching Tips**
- **Hands-on Focus**: Encourage students to type and run every code example
- **Shape Awareness**: Emphasize the importance of tensor shapes in neural networks
- **Error Discussion**: Show common errors and how to fix them
- **Visual Aids**: Use diagrams to show tensor operations and transformations

### **Common Student Issues**
- **Shape Mismatches**: Most common error in tensor operations
- **Data Type Confusion**: Mixing int32 and float32 operations
- **Broadcasting Rules**: Understanding how TensorFlow handles different shapes

### **Equipment Needed**
- Computers with TensorFlow 2.x installed
- Google Colab access (recommended)
- Code editor or Jupyter notebooks
- Sample datasets for practice

### **Time Management**
- Part 1 (Foundation): 15 minutes
- Part 2 (Operations): 20 minutes
- Part 3 (Tutorial): 20 minutes
- Part 4 (Assessment): 5 minutes

**Total: 60 minutes**