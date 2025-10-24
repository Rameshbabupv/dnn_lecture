# Instructor Solutions Guide - Day 4 Tutorial
**Week 3: Tutorial T3 - Tensor Operations & Neural Network Implementation**

---

## Quick Reference Solutions

### Part 1: Activation Functions

```python
# Complete implementations
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh_custom(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def relu_custom(x):
    return np.maximum(0, x)

def leaky_relu_custom(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Gradient functions
def sigmoid_gradient(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_gradient(x):
    t = tanh_custom(x)
    return 1 - t**2

def relu_gradient(x):
    return np.where(x > 0, 1.0, 0.0)

def leaky_relu_gradient(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)
```

### Part 2: Tensor Operations

```python
# Matrix multiplication
element_wise = tf.multiply(A, B)
matrix_mult = tf.matmul(A, B)

# Broadcasting
broadcasted_add = mat + vec

# Shape manipulation
reshaped = tf.reshape(original, [3, 2])
flattened = tf.reshape(original, [-1])
transposed = tf.transpose(original)
```

### Part 3: Layer Implementation

```python
class SimpleDenseLayer:
    def __init__(self, input_dim, output_dim, activation='relu'):
        # Xavier initialization
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.bias = np.zeros((1, output_dim))
        # ... rest of initialization
    
    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        activation_func = self.activation_funcs.get(self.activation_name, lambda x: x)
        self.output = activation_func(self.z)
        return self.output
```

### Part 4: Network Construction

```python
class SimpleNeuralNetwork:
    def __init__(self, layer_sizes, activations):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = SimpleDenseLayer(
                input_dim=layer_sizes[i],
                output_dim=layer_sizes[i+1],
                activation=activations[i]
            )
            self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
```

---

## Common Student Issues & Solutions

### Issue 1: Sigmoid Overflow
**Problem:** `RuntimeWarning: overflow encountered in exp`
**Solution:** Use `np.clip(x, -500, 500)` before exp to prevent overflow

### Issue 2: Matrix Dimension Mismatch
**Problem:** Shape errors in matrix multiplication
**Solution:** Remind students about @ operator vs * operator, and proper weight dimensions

### Issue 3: Broadcasting Confusion
**Problem:** Unexpected output shapes
**Solution:** Demonstrate with small examples, visualize broadcasting rules

### Issue 4: Softmax Numerical Stability
**Problem:** NaN values in softmax
**Solution:** Subtract max before exp: `exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))`

---

## Assessment Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| Activation Functions | 20% | All 4 functions correctly implemented |
| Gradients | 20% | Correct derivative formulas |
| Tensor Operations | 20% | Proper use of TF operations |
| Layer Implementation | 20% | Working forward pass |
| Network Construction | 20% | Correct layer stacking |

---

## Time Management Tips

1. **Minutes 0-5:** Environment setup, imports
2. **Minutes 5-20:** Activation functions (allow students to struggle briefly)
3. **Minutes 20-25:** Quick demo if students are stuck
4. **Minutes 25-45:** Tensor operations and layer building
5. **Minutes 45-55:** Network construction
6. **Minutes 55-60:** Testing and wrap-up

---

## Extension Activities (If Time Permits)

1. **Vanishing Gradient Demo:**
   ```python
   # Show gradient magnitude through deep sigmoid network
   x = np.random.randn(100)
   for i in range(10):
       x = sigmoid(x)
       grad = sigmoid_gradient(x)
       print(f"Layer {i+1}: Mean gradient = {np.mean(grad):.6f}")
   ```

2. **Weight Initialization Comparison:**
   - Compare random vs Xavier vs He initialization
   - Show impact on activation distributions

3. **Mini-Batch Processing:**
   - Modify network to handle different batch sizes
   - Discuss memory efficiency

---

## Key Learning Verification Questions

Ask students to explain:
1. Why does ReLU help with vanishing gradients?
2. What happens to gradients in deep sigmoid networks?
3. Why is weight initialization important?
4. How does broadcasting work in neural networks?
5. What's the difference between element-wise and matrix multiplication?

---

## Post-Tutorial Assignment Ideas

1. Implement ELU and Swish activations
2. Add dropout to the layer implementation
3. Implement backward pass for one layer
4. Compare custom implementation with Keras on MNIST
5. Visualize activation distributions for different functions

---

## Connection to Next Topics

**Module 2 Preview:**
- These gradients will be used in optimization algorithms
- Weight initialization strategies affect convergence
- Activation choice impacts training dynamics
- Understanding tensor operations essential for advanced architectures

---

## Quick Debugging Checklist

✓ Import statements correct?
✓ NumPy arrays vs TensorFlow tensors consistent?
✓ Shape assertions in place?
✓ Activation function returning correct shape?
✓ Weight dimensions match input/output?
✓ Batch dimension handled correctly?

---

## Sample Test Cases for Validation

```python
# Test 1: Activation boundary conditions
assert sigmoid(0) == 0.5
assert abs(tanh_custom(0)) < 1e-10
assert relu_custom(0) == 0

# Test 2: Shape preservation
x = np.random.randn(10, 20)
assert sigmoid(x).shape == x.shape

# Test 3: Layer dimension check
layer = SimpleDenseLayer(10, 5)
assert layer.weights.shape == (10, 5)
assert layer.bias.shape == (1, 5)

# Test 4: Network output
net = SimpleNeuralNetwork([10, 5, 2], ['relu', 'softmax'])
out = net(np.random.randn(3, 10))
assert out.shape == (3, 2)
assert np.allclose(out.sum(axis=1), 1.0)
```

---

## Notes for Next Session

- Students should now understand activation functions deeply
- Ready for gradient descent and optimization in Module 2
- Can build simple networks from scratch
- Foundation laid for understanding backpropagation

**Homework:** Read Goodfellow Chapter 8 on optimization before next class