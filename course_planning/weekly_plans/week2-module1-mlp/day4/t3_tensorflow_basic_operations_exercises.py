"""
Tutorial T3: Building Programs with Basic Operations in Tensors
Week 2, Session 6 - Neural Network Building Blocks
Course: Deep Neural Network Architectures (21CSE558T)

This file contains practical exercises for students to implement
basic tensor operations that form the building blocks of neural networks.

===============================================================================
ENVIRONMENT SETUP INSTRUCTIONS
===============================================================================

🐍 PYTHON REQUIREMENTS:
- Python 3.7 or higher (recommended: 3.8-3.11)
- Operating System: Windows 10+, macOS 10.14+, Ubuntu 18.04+

📦 ENVIRONMENT CREATION & INSTALLATION METHODS:

🅰️ METHOD 1: PYTHON VENV (Built-in, Recommended for most users)
   # Step 1: Create virtual environment
   python -m venv tensorflow_env
   
   # Step 2: Activate environment
   # On Windows:
   tensorflow_env\\Scripts\\activate
   # On macOS/Linux:
   source tensorflow_env/bin/activate
   
   # Step 3: Upgrade pip and install packages
   python -m pip install --upgrade pip
   pip install tensorflow>=2.10.0 numpy>=1.21.0 matplotlib>=3.5.0
   
   # Step 4: Verify installation
   python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"

🅱️ METHOD 2: CONDA/MINICONDA (Good for data science)
   # Step 1: Create conda environment
   conda create -n tensorflow_env python=3.9
   
   # Step 2: Activate environment
   conda activate tensorflow_env
   
   # Step 3: Install packages
   conda install tensorflow numpy matplotlib
   # OR use pip within conda:
   # pip install tensorflow>=2.10.0 numpy>=1.21.0 matplotlib>=3.5.0
   
   # Step 4: Verify installation
   python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"

🅲️ METHOD 3: VIRTUALENV (Alternative to venv)
   # Step 1: Install virtualenv (if not already installed)
   pip install virtualenv
   
   # Step 2: Create virtual environment
   virtualenv tensorflow_env
   
   # Step 3: Activate environment
   # On Windows:
   tensorflow_env\\Scripts\\activate
   # On macOS/Linux:
   source tensorflow_env/bin/activate
   
   # Step 4: Install packages
   pip install tensorflow>=2.10.0 numpy>=1.21.0 matplotlib>=3.5.0

🅳️ METHOD 4: PIPENV (Modern dependency management)
   # Step 1: Install pipenv (if not already installed)
   pip install pipenv
   
   # Step 2: Create Pipfile and install dependencies
   pipenv install tensorflow>=2.10.0 numpy>=1.21.0 matplotlib>=3.5.0
   
   # Step 3: Activate environment
   pipenv shell
   
   # Step 4: Run the exercises
   python t3_tensorflow_basic_operations_exercises.py

🅴️ METHOD 5: GOOGLE COLAB (Easiest for students - No local setup needed)
   # Step 1: Go to https://colab.research.google.com/
   # Step 2: Create new notebook
   # Step 3: TensorFlow is pre-installed
   # Step 4: Upload this file or copy-paste the code
   
   # In Colab cell, run:
   # !python t3_tensorflow_basic_operations_exercises.py --list

📋 REQUIREMENTS.txt FILE (For reproducibility):
   Create a requirements.txt file with:
   
   tensorflow>=2.10.0
   numpy>=1.21.0
   matplotlib>=3.5.0
   
   Then install with: pip install -r requirements.txt

🔄 ENVIRONMENT MANAGEMENT COMMANDS:

For VENV/VIRTUALENV:
   # Activate:   source tensorflow_env/bin/activate (macOS/Linux)
   #             tensorflow_env\\Scripts\\activate (Windows)
   # Deactivate: deactivate
   # Remove:     rm -rf tensorflow_env (macOS/Linux)
   #             rmdir /s tensorflow_env (Windows)

For CONDA:
   # Activate:   conda activate tensorflow_env
   # Deactivate: conda deactivate
   # List envs:  conda env list
   # Remove:     conda env remove -n tensorflow_env

For PIPENV:
   # Activate:   pipenv shell
   # Deactivate: exit
   # Remove:     pipenv --rm

🔧 VERIFICATION:
Run this file without parameters to check if everything is working:
   python t3_tensorflow_basic_operations_exercises.py

💡 USAGE EXAMPLES:
   python t3_tensorflow_basic_operations_exercises.py --all        # Run all exercises
   python t3_tensorflow_basic_operations_exercises.py ex-1        # Run Exercise 1 only
   python t3_tensorflow_basic_operations_exercises.py exercise-3  # Run Exercise 3 only
   python t3_tensorflow_basic_operations_exercises.py 5          # Run Exercise 5 only
   python t3_tensorflow_basic_operations_exercises.py --list     # List all exercises
   python t3_tensorflow_basic_operations_exercises.py --help     # Show help

🚨 TROUBLESHOOTING:
- If TensorFlow installation fails, try: pip install --upgrade pip
- For M1/M2 Macs: pip install tensorflow-macos tensorflow-metal
- For older GPUs: Check TensorFlow compatibility at tensorflow.org/install
- Memory issues: Reduce batch sizes or use Google Colab

===============================================================================
"""

import tensorflow as tf
import numpy as np
import argparse
import sys

# Check TensorFlow installation and version
try:
    print("✅ TensorFlow version:", tf.__version__)
    print("✅ NumPy version:", np.__version__)
    print("✅ Python version:", sys.version.split()[0])
    
    # Check if GPU is available (optional)
    if tf.config.list_physical_devices('GPU'):
        print("🚀 GPU acceleration available")
    else:
        print("💻 Running on CPU (this is fine for these exercises)")
    
except ImportError as e:
    print("❌ Import Error:", e)
    print("Please install TensorFlow: pip install tensorflow>=2.10.0")
    sys.exit(1)

print("=" * 60)

# =============================================================================
# EXERCISE 1: TENSOR CREATION AND MANIPULATION
# =============================================================================

def exercise_1_tensor_basics():
    """
    Exercise 1: Create different types of tensors and understand their properties
    """
    print("EXERCISE 1: TENSOR CREATION AND MANIPULATION")
    print("-" * 40)
    
    # TODO: Create the following tensors
    # 1. A scalar tensor with value 42
    scalar = tf.constant(42)
    
    # 2. A 1D tensor (vector) with values [1, 2, 3, 4, 5]
    vector = tf.constant([1, 2, 3, 4, 5])
    
    # 3. A 2D tensor (matrix) with shape (3, 3) filled with ones
    matrix = tf.ones((3, 3))
    
    # 4. A 3D tensor with shape (2, 3, 4) filled with random normal values
    tensor_3d = tf.random.normal((2, 3, 4))
    
    # 5. A variable tensor for weights (2x3 matrix with random initialization)
    weights = tf.Variable(tf.random.normal((2, 3), stddev=0.1), name="weights")
    
    # Print tensor properties
    tensors = [
        ("Scalar", scalar),
        ("Vector", vector),
        ("Matrix", matrix),
        ("3D Tensor", tensor_3d),
        ("Weights Variable", weights)
    ]
    
    for name, tensor in tensors:
        print(f"{name}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Rank: {tf.rank(tensor)}")
        print(f"  Data type: {tensor.dtype}")
        print(f"  Size: {tf.size(tensor)}")
        print()
    
    # Challenge: Create a tensor and reshape it
    original = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
    reshaped = tf.reshape(original, (4, 2))
    flattened = tf.reshape(original, (-1,))
    
    print("Reshape Challenge:")
    print(f"Original {original.shape}: {original}")
    print(f"Reshaped {reshaped.shape}: {reshaped}")
    print(f"Flattened {flattened.shape}: {flattened}")
    print()

# =============================================================================
# EXERCISE 2: MATHEMATICAL OPERATIONS
# =============================================================================

def exercise_2_math_operations():
    """
    Exercise 2: Implement basic mathematical operations on tensors
    """
    print("EXERCISE 2: MATHEMATICAL OPERATIONS")
    print("-" * 40)
    
    # Create sample tensors
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[2.0, 1.0], [4.0, 3.0]])
    
    print("Tensor A:")
    print(a)
    print("Tensor B:")
    print(b)
    print()
    
    # TODO: Implement the following operations
    # 1. Element-wise addition
    addition = tf.add(a, b)  # or a + b
    
    # 2. Element-wise multiplication
    element_mult = tf.multiply(a, b)  # or a * b
    
    # 3. Matrix multiplication
    matrix_mult = tf.matmul(a, b)
    
    # 4. Transpose of matrix A
    transpose_a = tf.transpose(a)
    
    # 5. Element-wise square root
    sqrt_a = tf.sqrt(a)
    
    # 6. Exponential
    exp_a = tf.exp(a)
    
    print("Results:")
    print(f"Addition (A + B):\n{addition}\n")
    print(f"Element-wise multiplication (A * B):\n{element_mult}\n")
    print(f"Matrix multiplication (A @ B):\n{matrix_mult}\n")
    print(f"Transpose of A:\n{transpose_a}\n")
    print(f"Square root of A:\n{sqrt_a}\n")
    print(f"Exponential of A:\n{exp_a}\n")

# =============================================================================
# EXERCISE 3: ACTIVATION FUNCTIONS
# =============================================================================

def exercise_3_activation_functions():
    """
    Exercise 3: Implement and test various activation functions
    """
    print("EXERCISE 3: ACTIVATION FUNCTIONS")
    print("-" * 40)
    
    # Test input values
    x = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0])
    print(f"Input values: {x}")
    print()
    
    # TODO: Apply the following activation functions
    # 1. ReLU (Rectified Linear Unit)
    relu_output = tf.nn.relu(x)
    
    # 2. Sigmoid
    sigmoid_output = tf.nn.sigmoid(x)
    
    # 3. Tanh (Hyperbolic tangent)
    tanh_output = tf.nn.tanh(x)
    
    # 4. Leaky ReLU
    leaky_relu_output = tf.nn.leaky_relu(x, alpha=0.01)
    
    # 5. Softmax (for classification - use positive values)
    softmax_input = tf.constant([2.0, 1.0, 0.1])
    softmax_output = tf.nn.softmax(softmax_input)
    
    print("Activation Function Results:")
    print(f"ReLU: {relu_output}")
    print(f"Sigmoid: {sigmoid_output}")
    print(f"Tanh: {tanh_output}")
    print(f"Leaky ReLU: {leaky_relu_output}")
    print(f"Softmax input: {softmax_input}")
    print(f"Softmax output: {softmax_output}")
    print(f"Softmax sum: {tf.reduce_sum(softmax_output)}")
    print()

# =============================================================================
# EXERCISE 4: REDUCTION OPERATIONS
# =============================================================================

def exercise_4_reduction_operations():
    """
    Exercise 4: Practice reduction operations (sum, mean, max, min)
    """
    print("EXERCISE 4: REDUCTION OPERATIONS")
    print("-" * 40)
    
    # Create a sample matrix
    data = tf.constant([[1, 2, 3], 
                        [4, 5, 6], 
                        [7, 8, 9]], dtype=tf.float32)
    
    print(f"Sample data:\n{data}\n")
    
    # TODO: Calculate the following reductions
    # 1. Sum of all elements
    sum_all = tf.reduce_sum(data)
    
    # 2. Sum along rows (axis=1)
    sum_rows = tf.reduce_sum(data, axis=1)
    
    # 3. Sum along columns (axis=0)  
    sum_cols = tf.reduce_sum(data, axis=0)
    
    # 4. Mean of all elements
    mean_all = tf.reduce_mean(data)
    
    # 5. Maximum value
    max_val = tf.reduce_max(data)
    
    # 6. Minimum value  
    min_val = tf.reduce_min(data)
    
    # 7. Standard deviation
    std_val = tf.math.reduce_std(data)
    
    print("Reduction Results:")
    print(f"Sum all elements: {sum_all}")
    print(f"Sum along rows: {sum_rows}")
    print(f"Sum along columns: {sum_cols}")
    print(f"Mean: {mean_all}")
    print(f"Maximum: {max_val}")
    print(f"Minimum: {min_val}")
    print(f"Standard deviation: {std_val}")
    print()

# =============================================================================
# EXERCISE 5: SIMPLE NEURAL NETWORK FORWARD PASS
# =============================================================================

def exercise_5_neural_network_forward_pass():
    """
    Exercise 5: Implement a simple neural network forward pass
    """
    print("EXERCISE 5: NEURAL NETWORK FORWARD PASS")
    print("-" * 40)
    
    # Network parameters
    input_size = 3
    hidden_size = 4
    output_size = 2
    
    # TODO: Initialize weights and biases
    # 1. Input to hidden layer weights (3x4)
    W1 = tf.Variable(tf.random.normal([input_size, hidden_size], stddev=0.1), name="W1")
    
    # 2. Hidden layer bias (4,)
    b1 = tf.Variable(tf.zeros([hidden_size]), name="b1")
    
    # 3. Hidden to output layer weights (4x2)
    W2 = tf.Variable(tf.random.normal([hidden_size, output_size], stddev=0.1), name="W2")
    
    # 4. Output layer bias (2,)
    b2 = tf.Variable(tf.zeros([output_size]), name="b2")
    
    # Sample input data (batch_size=2, features=3)
    inputs = tf.constant([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]])
    
    print(f"Input shape: {inputs.shape}")
    print(f"W1 shape: {W1.shape}, b1 shape: {b1.shape}")
    print(f"W2 shape: {W2.shape}, b2 shape: {b2.shape}")
    print()
    
    # TODO: Implement forward pass
    # Step 1: Calculate hidden layer pre-activation
    hidden_pre = tf.matmul(inputs, W1) + b1
    
    # Step 2: Apply activation function (ReLU)
    hidden_activated = tf.nn.relu(hidden_pre)
    
    # Step 3: Calculate output layer pre-activation
    output_pre = tf.matmul(hidden_activated, W2) + b2
    
    # Step 4: Apply output activation (sigmoid for binary classification)
    outputs = tf.nn.sigmoid(output_pre)
    
    print("Forward Pass Results:")
    print(f"Hidden pre-activation: {hidden_pre}")
    print(f"Hidden activated: {hidden_activated}")
    print(f"Output pre-activation: {output_pre}")
    print(f"Final outputs: {outputs}")
    print()

# =============================================================================
# EXERCISE 6: XOR PROBLEM IMPLEMENTATION
# =============================================================================

def exercise_6_xor_problem():
    """
    Exercise 6: Solve the XOR problem using TensorFlow operations
    """
    print("EXERCISE 6: XOR PROBLEM IMPLEMENTATION")
    print("-" * 40)
    
    # XOR truth table
    X = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float32)
    y_true = tf.constant([[0], [1], [1], [0]], dtype=tf.float32)
    
    print("XOR Truth Table:")
    for i in range(4):
        print(f"Input: {X[i].numpy()}, Expected Output: {y_true[i].numpy()}")
    print()
    
    # TODO: Define XOR network with pre-trained weights
    # These weights solve the XOR problem
    W1 = tf.Variable([[10.0, -10.0], 
                      [10.0, -10.0]], name="hidden_weights")
    b1 = tf.Variable([-5.0, 15.0], name="hidden_bias")
    W2 = tf.Variable([[10.0], 
                      [10.0]], name="output_weights")
    b2 = tf.Variable([-15.0], name="output_bias")
    
    # Forward pass function
    def xor_forward_pass(inputs):
        # Hidden layer
        hidden_pre = tf.matmul(inputs, W1) + b1
        hidden_activated = tf.nn.sigmoid(hidden_pre)
        
        # Output layer
        output_pre = tf.matmul(hidden_activated, W2) + b2
        output_activated = tf.nn.sigmoid(output_pre)
        
        return hidden_activated, output_activated
    
    # Test the XOR network
    hidden_vals, predictions = xor_forward_pass(X)
    
    print("XOR Network Results:")
    print("Input\t\tHidden Values\t\tPrediction\tExpected")
    print("-" * 60)
    for i in range(4):
        pred_rounded = tf.round(predictions[i]).numpy()
        print(f"{X[i].numpy()}\t{hidden_vals[i].numpy()}\t{predictions[i].numpy()[0]:.3f}\t\t{y_true[i].numpy()[0]}")
    
    # Calculate accuracy
    predictions_rounded = tf.round(predictions)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions_rounded, y_true), tf.float32))
    print(f"\nAccuracy: {accuracy.numpy():.3f}")
    print()

# =============================================================================
# EXERCISE 7: DATA PREPROCESSING
# =============================================================================

def exercise_7_data_preprocessing():
    """
    Exercise 7: Practice common data preprocessing operations
    """
    print("EXERCISE 7: DATA PREPROCESSING")
    print("-" * 40)
    
    # Sample dataset
    raw_data = tf.constant([[1, 2, 3], 
                           [4, 5, 6], 
                           [7, 8, 9], 
                           [10, 11, 12]], dtype=tf.float32)
    
    print(f"Raw data:\n{raw_data}\n")
    
    # TODO: Implement preprocessing operations
    # 1. Normalization (zero mean, unit variance)
    mean = tf.reduce_mean(raw_data, axis=0)
    std = tf.math.reduce_std(raw_data, axis=0)
    normalized = (raw_data - mean) / std
    
    # 2. Min-Max scaling (scale to [0, 1])
    min_val = tf.reduce_min(raw_data, axis=0)
    max_val = tf.reduce_max(raw_data, axis=0)
    min_max_scaled = (raw_data - min_val) / (max_val - min_val)
    
    # 3. One-hot encoding for categorical labels
    labels = tf.constant([0, 1, 2, 1, 0])
    one_hot = tf.one_hot(labels, depth=3)
    
    print("Preprocessing Results:")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    print(f"Normalized data:\n{normalized}\n")
    
    print(f"Min values: {min_val}")
    print(f"Max values: {max_val}")
    print(f"Min-Max scaled data:\n{min_max_scaled}\n")
    
    print(f"Original labels: {labels}")
    print(f"One-hot encoded:\n{one_hot}\n")

# =============================================================================
# EXERCISE 8: DEBUGGING AND ERROR HANDLING
# =============================================================================

def exercise_8_debugging():
    """
    Exercise 8: Practice debugging common tensor operations
    """
    print("EXERCISE 8: DEBUGGING AND ERROR HANDLING")
    print("-" * 40)
    
    # Helper function to debug tensor properties
    def debug_tensor(tensor, name):
        print(f"{name}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Rank: {tf.rank(tensor).numpy()}")
        print(f"  Data type: {tensor.dtype}")
        print(f"  Size: {tf.size(tensor).numpy()}")
        print()
    
    # Sample tensors for debugging
    tensor_a = tf.random.normal([32, 784])  # Simulating flattened MNIST images
    tensor_b = tf.random.normal([784, 128])  # Hidden layer weights
    tensor_c = tf.random.normal([128, 10])   # Output layer weights
    
    debug_tensor(tensor_a, "Input Batch")
    debug_tensor(tensor_b, "Hidden Weights")
    debug_tensor(tensor_c, "Output Weights")
    
    # Demonstrate shape checking before operations
    print("Shape Compatibility Check:")
    print(f"Can multiply {tensor_a.shape} × {tensor_b.shape}? {tensor_a.shape[-1] == tensor_b.shape[0]}")
    print(f"Result shape would be: ({tensor_a.shape[0]}, {tensor_b.shape[1]})")
    print()
    
    # Check for problematic values
    problematic = tf.constant([1.0, float('inf'), -float('inf'), float('nan')])
    
    print("Value Checks:")
    print(f"Tensor: {problematic}")
    print(f"Has infinity: {tf.reduce_any(tf.math.is_inf(problematic)).numpy()}")
    print(f"Has NaN: {tf.reduce_any(tf.math.is_nan(problematic)).numpy()}")
    print(f"All finite: {tf.reduce_all(tf.math.is_finite(problematic)).numpy()}")
    print()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Exercise registry for easy access
EXERCISE_REGISTRY = {
    '1': {'func': exercise_1_tensor_basics, 'name': 'Tensor Creation and Manipulation'},
    '2': {'func': exercise_2_math_operations, 'name': 'Mathematical Operations'},
    '3': {'func': exercise_3_activation_functions, 'name': 'Activation Functions'},
    '4': {'func': exercise_4_reduction_operations, 'name': 'Reduction Operations'},
    '5': {'func': exercise_5_neural_network_forward_pass, 'name': 'Neural Network Forward Pass'},
    '6': {'func': exercise_6_xor_problem, 'name': 'XOR Problem Implementation'},
    '7': {'func': exercise_7_data_preprocessing, 'name': 'Data Preprocessing'},
    '8': {'func': exercise_8_debugging, 'name': 'Debugging and Error Handling'}
}

def parse_exercise_number(exercise_arg):
    """
    Parse different exercise number formats
    Supports: ex-1, exercise-1, 1, etc.
    """
    exercise_arg = str(exercise_arg).lower().strip()
    
    # Handle different formats
    if exercise_arg.startswith('ex-'):
        return exercise_arg.split('-')[1]
    elif exercise_arg.startswith('exercise-'):
        return exercise_arg.split('-')[1]
    elif exercise_arg.startswith('ex'):
        return exercise_arg[2:]
    elif exercise_arg.startswith('exercise'):
        return exercise_arg[8:]
    else:
        return exercise_arg

def list_exercises():
    """
    Display all available exercises
    """
    print("📋 AVAILABLE EXERCISES:")
    print("=" * 50)
    for num, info in EXERCISE_REGISTRY.items():
        print(f"  Exercise {num}: {info['name']}")
    print("\n💡 Usage Examples:")
    print("  python t3_tensorflow_basic_operations_exercises.py 1")
    print("  python t3_tensorflow_basic_operations_exercises.py ex-3")
    print("  python t3_tensorflow_basic_operations_exercises.py exercise-5")
    print("  python t3_tensorflow_basic_operations_exercises.py --all")

def run_exercise(exercise_num):
    """
    Run a specific exercise by number
    """
    if exercise_num in EXERCISE_REGISTRY:
        exercise_info = EXERCISE_REGISTRY[exercise_num]
        print(f"🎯 RUNNING EXERCISE {exercise_num}: {exercise_info['name']}")
        print("=" * 60)
        print()
        
        try:
            exercise_info['func']()
            print(f"✅ Exercise {exercise_num} completed successfully!")
        except Exception as e:
            print(f"❌ Error in Exercise {exercise_num}: {e}")
            print("💡 Check your TensorFlow installation and try again.")
    else:
        print(f"❌ Exercise {exercise_num} not found!")
        print("Available exercises: 1-8")
        list_exercises()

def run_all_exercises():
    """
    Run all exercises in sequence
    """
    print("🚀 TUTORIAL T3: TENSORFLOW BASIC OPERATIONS")
    print("=" * 60)
    print("Running all exercises...")
    print()
    
    completed = 0
    total = len(EXERCISE_REGISTRY)
    
    for num in sorted(EXERCISE_REGISTRY.keys()):
        try:
            print(f"\n📍 Starting Exercise {num}...")
            EXERCISE_REGISTRY[num]['func']()
            completed += 1
            print(f"✅ Exercise {num} completed! ({completed}/{total})")
        except Exception as e:
            print(f"❌ Exercise {num} failed: {e}")
            continue
    
    print("\n" + "=" * 60)
    if completed == total:
        print("🎉 ALL EXERCISES COMPLETED SUCCESSFULLY!")
        print("You have successfully implemented the building blocks of neural networks!")
    else:
        print(f"📊 Completed {completed}/{total} exercises.")
        print("Some exercises had issues. Check error messages above.")

def setup_argument_parser():
    """
    Setup command line argument parser
    """
    parser = argparse.ArgumentParser(
        description='TensorFlow Basic Operations Exercises - Tutorial T3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  %(prog)s                    # Run all exercises
  %(prog)s --all              # Run all exercises  
  %(prog)s 1                  # Run Exercise 1 only
  %(prog)s ex-3               # Run Exercise 3 only
  %(prog)s exercise-5         # Run Exercise 5 only
  %(prog)s --list             # List all available exercises

SUPPORTED EXERCISE FORMATS:
  1, ex-1, exercise-1         # All refer to Exercise 1
  3, ex-3, exercise-3         # All refer to Exercise 3
        """
    )
    
    parser.add_argument(
        'exercise',
        nargs='?',
        help='Exercise number to run (1-8, ex-1, exercise-1, etc.)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all exercises (default behavior)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available exercises'
    )
    
    return parser

def main():
    """
    Main function with command line argument support
    """
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Handle different argument scenarios
    if args.list:
        list_exercises()
    elif args.exercise:
        # Run specific exercise
        exercise_num = parse_exercise_number(args.exercise)
        run_exercise(exercise_num)
    elif args.all:
        # Explicitly run all
        run_all_exercises()
    else:
        # Default behavior - run all exercises
        run_all_exercises()

if __name__ == "__main__":
    main()

# =============================================================================
# ADDITIONAL CHALLENGES (OPTIONAL)
# =============================================================================

"""
ADDITIONAL CHALLENGES FOR ADVANCED STUDENTS:

1. Implement a 3-layer neural network for multi-class classification
2. Create a function to visualize tensor shapes and operations
3. Implement batch normalization using tensor operations
4. Create a simple gradient calculation using tf.GradientTape
5. Implement a basic loss function (mean squared error, cross-entropy)
6. Create a function to save and load model weights
7. Implement dropout as a tensor operation
8. Create visualization of activation function behaviors

HINT: Use the following structure for challenge implementations:

def challenge_1_multilayer_network():
    # Implement 3-layer network here
    pass

def challenge_2_visualization():
    # Create visualization functions here
    pass

# Continue with other challenges...
"""