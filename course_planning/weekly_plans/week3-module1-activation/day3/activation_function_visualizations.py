#!/usr/bin/env python3
"""
Activation Function Visualizations for Day 3 Lecture
Week 3: Advanced Neural Network Fundamentals
Deep Neural Network Architectures (21CSE558T)

© 2025 Prof. Ramesh Babu | SRM University | DSBS
Run this script to generate all visual aids for the Day 3 lecture
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set up beautiful plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.alpha'] = 0.3

def save_and_show(filename, show=True):
    """Save figure and optionally display"""
    plt.tight_layout()
    plt.savefig(f"week3_day3_{filename}.png", dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.clf()

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Sigmoid derivative"""
    s = sigmoid(x)
    return s * (1 - s)

def tanh_derivative(x):
    """Tanh derivative"""
    t = np.tanh(x)
    return 1 - t**2

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU derivative"""
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function"""
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    """Leaky ReLU derivative"""
    return np.where(x > 0, 1, alpha)

def create_activation_comparison():
    """Create comprehensive activation function comparison"""
    x = np.linspace(-5, 5, 1000)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🎭 Classical Activation Functions & Their Derivatives', fontsize=16, fontweight='bold')
    
    # Sigmoid
    axes[0, 0].plot(x, sigmoid(x), 'purple', linewidth=3, label='Sigmoid')
    axes[0, 0].plot(x, sigmoid_derivative(x), 'purple', linestyle='--', linewidth=2, label='Sigmoid Derivative')
    axes[0, 0].set_title('Sigmoid: σ(x) = 1/(1 + e⁻ˣ)', fontweight='bold')
    axes[0, 0].set_ylabel('f(x)')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    axes[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Add annotations for sigmoid
    axes[0, 0].annotate('Saturation\n(Vanishing Gradients)', xy=(-4, 0.02), xytext=(-3, 0.3),
                       arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Tanh
    axes[0, 1].plot(x, np.tanh(x), 'teal', linewidth=3, label='Tanh')
    axes[0, 1].plot(x, tanh_derivative(x), 'teal', linestyle='--', linewidth=2, label='Tanh Derivative')
    axes[0, 1].set_title('Tanh: (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)', fontweight='bold')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Add annotations for tanh
    axes[0, 1].annotate('Zero-centered\n(Better than Sigmoid)', xy=(0, 0), xytext=(2, -0.7),
                       arrowprops=dict(arrowstyle='->', color='green'), fontsize=10, ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    # ReLU
    axes[1, 0].plot(x, relu(x), 'red', linewidth=3, label='ReLU')
    axes[1, 0].plot(x, relu_derivative(x), 'red', linestyle='--', linewidth=2, label='ReLU Derivative')
    axes[1, 0].set_title('ReLU: f(x) = max(0, x)', fontweight='bold')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('f(x)')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Add annotations for ReLU
    axes[1, 0].annotate('Dead Neurons\n(x < 0)', xy=(-2, 0), xytext=(-3, 2),
                       arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7))
    
    # Leaky ReLU
    axes[1, 1].plot(x, leaky_relu(x), 'orange', linewidth=3, label='Leaky ReLU')
    axes[1, 1].plot(x, leaky_relu_derivative(x), 'orange', linestyle='--', linewidth=2, label='Leaky ReLU Derivative')
    axes[1, 1].set_title('Leaky ReLU: f(x) = max(0.01x, x)', fontweight='bold')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Add annotations for Leaky ReLU
    axes[1, 1].annotate('Small gradient\n(Prevents dying)', xy=(-2, -0.02), xytext=(-3, 1.5),
                       arrowprops=dict(arrowstyle='->', color='green'), fontsize=10, ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    save_and_show("activation_functions_comparison")

def create_gradient_flow_demo():
    """Demonstrate vanishing gradient problem"""
    x = np.linspace(-5, 5, 1000)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('🌊 Gradient Flow Analysis: Why ReLU Won the Battle', fontsize=16, fontweight='bold')
    
    # Sigmoid gradients
    axes[0].plot(x, sigmoid_derivative(x), 'purple', linewidth=3, label='Sigmoid Gradient')
    axes[0].fill_between(x, 0, sigmoid_derivative(x), alpha=0.3, color='purple')
    axes[0].set_title('Sigmoid Gradients\n(Vanishing Problem)', fontweight='bold')
    axes[0].set_ylabel('Gradient Magnitude')
    axes[0].grid(True)
    axes[0].axhline(y=0.25, color='red', linestyle='--', alpha=0.7, label='Max Gradient = 0.25')
    axes[0].legend()
    
    # Tanh gradients
    axes[1].plot(x, tanh_derivative(x), 'teal', linewidth=3, label='Tanh Gradient')
    axes[1].fill_between(x, 0, tanh_derivative(x), alpha=0.3, color='teal')
    axes[1].set_title('Tanh Gradients\n(Better, but still saturates)', fontweight='bold')
    axes[1].set_xlabel('x')
    axes[1].grid(True)
    axes[1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Max Gradient = 1.0')
    axes[1].legend()
    
    # ReLU gradients
    relu_grad = relu_derivative(x)
    axes[2].plot(x, relu_grad, 'red', linewidth=3, label='ReLU Gradient')
    axes[2].fill_between(x, 0, relu_grad, alpha=0.3, color='red')
    axes[2].set_title('ReLU Gradients\n(No Saturation!)', fontweight='bold')
    axes[2].grid(True)
    axes[2].set_ylim(-0.1, 1.2)
    axes[2].axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Constant Gradient = 1.0')
    axes[2].legend()
    
    save_and_show("gradient_flow_analysis")

def create_neural_layer_architecture():
    """Visualize neural network layer architecture with matrix operations"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('🧠 Neural Network Layer Architecture & Matrix Operations', fontsize=16, fontweight='bold')
    
    # Layer architecture diagram
    ax1 = axes[0]
    
    # Input layer
    input_y = [1, 2, 3]
    for i, y in enumerate(input_y):
        circle = plt.Circle((1, y), 0.3, color='lightblue', ec='black', linewidth=2)
        ax1.add_patch(circle)
        ax1.text(1, y, f'x{i+1}', ha='center', va='center', fontweight='bold')
    
    ax1.text(1, 0.2, 'Input Layer\n(3 neurons)', ha='center', fontweight='bold')
    
    # Hidden layer
    hidden_y = [0.5, 1.5, 2.5, 3.5]
    for i, y in enumerate(hidden_y):
        circle = plt.Circle((3, y), 0.3, color='lightgreen', ec='black', linewidth=2)
        ax1.add_patch(circle)
        ax1.text(3, y, f'h{i+1}', ha='center', va='center', fontweight='bold')
        # Add activation function symbol
        ax1.text(3.5, y, 'σ', ha='center', va='center', fontsize=16, color='red', fontweight='bold')
    
    ax1.text(3, -0.3, 'Hidden Layer\n(4 neurons)', ha='center', fontweight='bold')
    
    # Output layer
    output_y = [1.5, 2.5]
    for i, y in enumerate(output_y):
        circle = plt.Circle((5, y), 0.3, color='lightcoral', ec='black', linewidth=2)
        ax1.add_patch(circle)
        ax1.text(5, y, f'y{i+1}', ha='center', va='center', fontweight='bold')
    
    ax1.text(5, 0.7, 'Output Layer\n(2 neurons)', ha='center', fontweight='bold')
    
    # Connections
    for i in input_y:
        for j in hidden_y:
            ax1.plot([1.3, 2.7], [i, j], 'k-', alpha=0.3, linewidth=1)
    
    for i in hidden_y:
        for j in output_y:
            ax1.plot([3.3, 4.7], [i, j], 'k-', alpha=0.3, linewidth=1)
    
    # Weight matrices labels
    ax1.text(2, 4, 'W₁ (3×4)\nb₁ (4×1)', ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    ax1.text(4, 4, 'W₂ (4×2)\nb₂ (2×1)', ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax1.set_xlim(0, 6)
    ax1.set_ylim(-0.5, 4.5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Network Architecture', fontweight='bold')
    
    # Matrix operations visualization
    ax2 = axes[1]
    
    # Create matrix visualization
    ax2.text(0.5, 0.9, 'Forward Pass Mathematics:', fontsize=14, fontweight='bold', transform=ax2.transAxes)
    
    equations = [
        'Layer 1: z₁ = x·W₁ + b₁',
        '         a₁ = σ(z₁)',
        '',
        'Layer 2: z₂ = a₁·W₂ + b₂', 
        '         a₂ = σ(z₂)',
        '',
        'Matrix Dimensions:',
        '• Input: (batch_size, 3)',
        '• W₁: (3, 4), b₁: (4,)',
        '• Hidden: (batch_size, 4)',
        '• W₂: (4, 2), b₂: (2,)',
        '• Output: (batch_size, 2)'
    ]
    
    for i, eq in enumerate(equations):
        ax2.text(0.05, 0.8 - i*0.06, eq, fontsize=12, transform=ax2.transAxes,
                fontweight='bold' if '•' in eq or 'Layer' in eq else 'normal',
                color='red' if 'σ' in eq else 'black')
    
    # Add activation function examples
    ax2.text(0.05, 0.25, 'Common Activation Choices:', fontsize=12, fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.05, 0.20, '• Hidden layers: ReLU, Tanh, Leaky ReLU', fontsize=11, transform=ax2.transAxes)
    ax2.text(0.05, 0.15, '• Binary output: Sigmoid', fontsize=11, transform=ax2.transAxes)
    ax2.text(0.05, 0.10, '• Multi-class: Softmax', fontsize=11, transform=ax2.transAxes)
    ax2.text(0.05, 0.05, '• Regression: Linear (no activation)', fontsize=11, transform=ax2.transAxes)
    
    ax2.axis('off')
    ax2.set_title('Mathematical Operations', fontweight='bold')
    
    save_and_show("neural_layer_architecture")

def create_initialization_demo():
    """Demonstrate weight initialization strategies"""
    np.random.seed(42)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('⚖️ Weight Initialization Strategies & Their Impact', fontsize=16, fontweight='bold')
    
    n_inputs, n_outputs = 100, 50
    n_samples = 1000
    
    # Different initialization strategies
    strategies = [
        ('Random Normal (σ=1)', np.random.normal(0, 1, (n_inputs, n_outputs))),
        ('Random Normal (σ=0.01)', np.random.normal(0, 0.01, (n_inputs, n_outputs))),
        ('Xavier/Glorot', np.random.normal(0, np.sqrt(2.0/(n_inputs + n_outputs)), (n_inputs, n_outputs))),
        ('He Initialization', np.random.normal(0, np.sqrt(2.0/n_inputs), (n_inputs, n_outputs))),
        ('Random Uniform [-1,1]', np.random.uniform(-1, 1, (n_inputs, n_outputs))),
        ('Zeros (Bad!)', np.zeros((n_inputs, n_outputs)))
    ]
    
    # Simulate forward pass with different initializations
    input_data = np.random.normal(0, 1, (n_samples, n_inputs))
    
    for idx, (name, weights) in enumerate(strategies):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Forward pass
        output = np.dot(input_data, weights)
        
        # Apply ReLU activation
        activated_output = np.maximum(0, output)
        
        # Plot distribution of activations
        ax.hist(activated_output.flatten(), bins=50, alpha=0.7, density=True, color=plt.cm.Set3(idx))
        ax.set_title(f'{name}\n(Mean: {activated_output.mean():.3f}, Std: {activated_output.std():.3f})', 
                    fontweight='bold')
        ax.set_xlabel('Activation Values')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        
        # Add annotations for good/bad cases
        if 'Xavier' in name or 'He' in name:
            ax.text(0.02, 0.95, '✅ Good!', transform=ax.transAxes, fontsize=12, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
        elif 'Zeros' in name or 'σ=1' in name:
            ax.text(0.02, 0.95, '❌ Bad!', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8))
        else:
            ax.text(0.02, 0.95, '⚠️ Okay', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    save_and_show("initialization_strategies")

def create_backpropagation_visual():
    """Create backpropagation visualization"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle('🔄 Backpropagation: The Learning Engine', fontsize=16, fontweight='bold')
    
    # Forward pass visualization
    ax1 = axes[0]
    
    # Network layers
    layers = ['Input', 'Hidden 1', 'Hidden 2', 'Output']
    layer_x = [0, 2, 4, 6]
    
    for i, (layer, x) in enumerate(zip(layers, layer_x)):
        # Draw layer
        rect = Rectangle((x-0.3, 1), 0.6, 2, facecolor=plt.cm.Set3(i), alpha=0.7, ec='black')
        ax1.add_patch(rect)
        ax1.text(x, 2, layer, ha='center', va='center', fontweight='bold')
        
        # Add arrows for forward pass
        if i < len(layers) - 1:
            ax1.arrow(x+0.3, 2, 1.4, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
            ax1.text(x+1, 2.3, 'Forward', ha='center', color='blue', fontweight='bold')
    
    # Add mathematical operations
    operations = ['W₁·x + b₁\nσ(z₁)', 'W₂·a₁ + b₂\nσ(z₂)', 'W₃·a₂ + b₃\nσ(z₃)']
    for i, op in enumerate(operations):
        ax1.text(layer_x[i] + 1, 1.2, op, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7))
    
    ax1.set_xlim(-1, 7)
    ax1.set_ylim(0.5, 3.5)
    ax1.set_title('Forward Pass: Computing Predictions', fontweight='bold')
    ax1.axis('off')
    
    # Backward pass visualization
    ax2 = axes[1]
    
    for i, (layer, x) in enumerate(zip(layers, layer_x)):
        # Draw layer
        rect = Rectangle((x-0.3, 1), 0.6, 2, facecolor=plt.cm.Set3(i), alpha=0.7, ec='black')
        ax2.add_patch(rect)
        ax2.text(x, 2, layer, ha='center', va='center', fontweight='bold')
        
        # Add arrows for backward pass
        if i > 0:
            ax2.arrow(x-0.3, 2, -1.4, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')
            ax2.text(x-1, 1.7, 'Backward', ha='center', color='red', fontweight='bold')
    
    # Add gradient computations
    gradients = ['∂L/∂W₃\n∂L/∂b₃', '∂L/∂W₂\n∂L/∂b₂', '∂L/∂W₁\n∂L/∂b₁']
    for i, grad in enumerate(gradients):
        ax2.text(layer_x[3-i] - 1, 2.8, grad, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcoral', alpha=0.7))
    
    # Loss function
    ax2.text(6.5, 2, 'Loss\nL(y, ŷ)', ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    ax2.set_xlim(-1, 7.5)
    ax2.set_ylim(0.5, 3.5)
    ax2.set_title('Backward Pass: Computing Gradients for Learning', fontweight='bold')
    ax2.axis('off')
    
    save_and_show("backpropagation_visual")

def create_summary_slide():
    """Create a comprehensive summary slide"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    fig.suptitle('📚 Week 3 Day 3: Neural Network Fundamentals Summary', fontsize=20, fontweight='bold')
    
    # Remove axes
    ax.axis('off')
    
    # Create summary content
    summary_content = """
🎭 ACTIVATION FUNCTIONS MASTERY
┌─────────────────────────────────────────────────────────────────┐
│ Classic Functions:                                              │
│ • Sigmoid: σ(x) = 1/(1+e⁻ˣ) → Range: (0,1) → Binary outputs   │
│ • Tanh: (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) → Range: (-1,1) → Zero-centered     │
│                                                                 │
│ Modern Powerhouses:                                             │
│ • ReLU: f(x) = max(0,x) → Simple, fast, no saturation         │
│ • Leaky ReLU: f(x) = max(αx,x) → Prevents dying neurons       │
│ • ELU: Smooth, negative saturation → Better gradient flow      │
└─────────────────────────────────────────────────────────────────┘

🧠 NEURAL LAYER MATHEMATICS
┌─────────────────────────────────────────────────────────────────┐
│ Forward Pass: a⁽ˡ⁾ = σ(W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾)                      │
│                                                                 │
│ Key Components:                                                 │
│ • Weight Matrix W: Learnable parameters                        │
│ • Bias Vector b: Shifts the activation                         │
│ • Activation σ: Introduces non-linearity                       │
│                                                                 │
│ Initialization Matters:                                         │
│ • Xavier/Glorot: √(2/(fan_in + fan_out))                      │
│ • He: √(2/fan_in) → Better for ReLU                           │
└─────────────────────────────────────────────────────────────────┘

🔄 BACKPROPAGATION FRAMEWORK
┌─────────────────────────────────────────────────────────────────┐
│ Chain Rule Application:                                         │
│ ∂L/∂W⁽ˡ⁾ = ∂L/∂a⁽ˡ⁾ · ∂a⁽ˡ⁾/∂z⁽ˡ⁾ · ∂z⁽ˡ⁾/∂W⁽ˡ⁾                   │
│                                                                 │
│ Gradient Flow:                                                  │
│ • Good activations → Stable gradients                          │
│ • Bad activations → Vanishing/exploding gradients              │
│                                                                 │
│ Learning Update:                                                │
│ W⁽ˡ⁾ ← W⁽ˡ⁾ - η · ∂L/∂W⁽ˡ⁾                                    │
└─────────────────────────────────────────────────────────────────┘

🎯 TOMORROW'S PREVIEW: Hands-On Implementation
• Build custom activation functions from scratch
• Implement neural layers with TensorFlow
• Create complete neural networks
• Test different activation choices practically
    """
    
    # Add the text content
    ax.text(0.05, 0.95, summary_content, fontsize=12, fontfamily='monospace',
            verticalalignment='top', transform=ax.transAxes)
    
    # Add decorative elements
    from matplotlib.patches import FancyBboxPatch
    
    # Background box
    fancy_box = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.02",
                              facecolor='lightblue', alpha=0.1, transform=ax.transAxes)
    ax.add_patch(fancy_box)
    
    save_and_show("lecture_summary")

def main():
    """Generate all visualizations for Day 3 lecture"""
    print("🎭 Generating Week 3 Day 3 Visual Aids...")
    print("=" * 50)
    
    print("1. Creating activation function comparison...")
    create_activation_comparison()
    
    print("2. Creating gradient flow analysis...")
    create_gradient_flow_demo()
    
    print("3. Creating neural layer architecture...")
    create_neural_layer_architecture()
    
    print("4. Creating initialization demonstration...")
    create_initialization_demo()
    
    print("5. Creating backpropagation visualization...")
    create_backpropagation_visual()
    
    print("6. Creating summary slide...")
    create_summary_slide()
    
    print("\n✅ All visualizations generated successfully!")
    print("📁 Files saved with prefix 'week3_day3_'")
    print("🎨 Ready for your Day 3 lecture!")

if __name__ == "__main__":
    main()