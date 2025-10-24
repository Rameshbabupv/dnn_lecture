"""
Deep Neural Network Architecture Diagrams
Course: 21CSE558T - Deep Neural Network Architectures
Created for M.Tech students at SRM University
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import seaborn as sns

# Set style for academic presentations
plt.style.use('default')
sns.set_palette("husl")

def create_basic_dnn_diagram():
    """Create a comprehensive Deep Neural Network architecture diagram"""

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Define colors
    colors = {
        'input': '#3498db',      # Blue
        'hidden': '#e74c3c',     # Red
        'output': '#2ecc71',     # Green
        'weights': '#95a5a6',    # Gray
        'bias': '#f39c12',       # Orange
        'activation': '#9b59b6'  # Purple
    }

    # Layer positions
    layer_x = [1, 3, 5, 7, 9]  # x-positions for layers
    layer_sizes = [4, 6, 8, 6, 3]  # neurons per layer
    layer_names = ['Input\nLayer', 'Hidden\nLayer 1', 'Hidden\nLayer 2', 'Hidden\nLayer 3', 'Output\nLayer']
    layer_colors = [colors['input'], colors['hidden'], colors['hidden'], colors['hidden'], colors['output']]

    # Draw neurons and connections
    neuron_positions = {}

    for i, (x, size, name, color) in enumerate(zip(layer_x, layer_sizes, layer_names, layer_colors)):
        # Calculate y positions for neurons in this layer
        y_positions = np.linspace(1, 9, size)
        neuron_positions[i] = []

        for j, y in enumerate(y_positions):
            # Draw neuron
            circle = patches.Circle((x, y), 0.15, facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            neuron_positions[i].append((x, y))

            # Add neuron labels for input and output layers
            if i == 0:  # Input layer
                ax.text(x-0.4, y, f'x{j+1}', fontsize=10, ha='right', va='center', fontweight='bold')
            elif i == len(layer_x)-1:  # Output layer
                ax.text(x+0.4, y, f'y{j+1}', fontsize=10, ha='left', va='center', fontweight='bold')

        # Add layer label
        ax.text(x, 0.3, name, fontsize=12, ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))

    # Draw connections between layers
    for i in range(len(layer_x)-1):
        for pos1 in neuron_positions[i]:
            for pos2 in neuron_positions[i+1]:
                line = plt.Line2D([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                                color=colors['weights'], alpha=0.3, linewidth=0.8)
                ax.add_line(line)

    # Add weight and bias annotations
    ax.text(2, 9.5, 'Weights (W)', fontsize=12, ha='center', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['weights'], alpha=0.5))
    ax.text(2, 9.2, 'W₁₁, W₁₂, ...', fontsize=10, ha='center', style='italic')

    # Add activation function box
    activation_box = FancyBboxPatch((4.5, 10), 1, 0.6,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['activation'], alpha=0.7)
    ax.add_patch(activation_box)
    ax.text(5, 10.3, 'Activation\nFunction', fontsize=10, ha='center', va='center',
            fontweight='bold', color='white')

    # Add mathematical formulation
    ax.text(5, -0.5, 'Mathematical Formulation:', fontsize=14, ha='center', fontweight='bold')
    ax.text(5, -1.0, r'$h^{(l+1)} = f(W^{(l)}h^{(l)} + b^{(l)})$', fontsize=16, ha='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.5))
    ax.text(5, -1.4, 'where f() is activation function (ReLU, Sigmoid, Tanh)', fontsize=10, ha='center', style='italic')

    # Set axis properties
    ax.set_xlim(0, 10)
    ax.set_ylim(-2, 11)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    plt.title('Deep Neural Network Architecture\n(Multi-Layer Perceptron)',
              fontsize=18, fontweight='bold', pad=20)

    # Add legend
    legend_elements = [
        patches.Circle((0, 0), 0.15, facecolor=colors['input'], label='Input Neurons'),
        patches.Circle((0, 0), 0.15, facecolor=colors['hidden'], label='Hidden Neurons'),
        patches.Circle((0, 0), 0.15, facecolor=colors['output'], label='Output Neurons'),
        plt.Line2D([0], [0], color=colors['weights'], label='Weights/Connections')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))

    plt.tight_layout()
    return fig

def create_cnn_architecture_diagram():
    """Create CNN architecture diagram"""

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Define CNN components with positions and sizes
    components = [
        {'name': 'Input\nImage', 'pos': (0.5, 4), 'size': (1, 4), 'color': '#3498db'},
        {'name': 'Conv\n3x3', 'pos': (2.5, 4), 'size': (0.8, 3.5), 'color': '#e74c3c'},
        {'name': 'ReLU', 'pos': (3.8, 4), 'size': (0.6, 3.5), 'color': '#9b59b6'},
        {'name': 'MaxPool\n2x2', 'pos': (5, 4), 'size': (0.8, 2.5), 'color': '#f39c12'},
        {'name': 'Conv\n3x3', 'pos': (6.5, 4), 'size': (0.8, 2.2), 'color': '#e74c3c'},
        {'name': 'ReLU', 'pos': (7.8, 4), 'size': (0.6, 2.2), 'color': '#9b59b6'},
        {'name': 'MaxPool\n2x2', 'pos': (9, 4), 'size': (0.8, 1.5), 'color': '#f39c12'},
        {'name': 'Flatten', 'pos': (10.5, 4), 'size': (0.6, 1.5), 'color': '#95a5a6'},
        {'name': 'Dense\n128', 'pos': (12, 4), 'size': (0.8, 2), 'color': '#2ecc71'},
        {'name': 'Dropout\n0.5', 'pos': (13.3, 4), 'size': (0.8, 2), 'color': '#34495e'},
        {'name': 'Dense\n10', 'pos': (14.8, 4), 'size': (0.8, 1.5), 'color': '#2ecc71'}
    ]

    # Draw components
    for i, comp in enumerate(components):
        x, y = comp['pos']
        w, h = comp['size']

        # Create rectangle
        rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                             boxstyle="round,pad=0.05",
                             facecolor=comp['color'], alpha=0.7,
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)

        # Add text
        ax.text(x, y, comp['name'], fontsize=10, ha='center', va='center',
                fontweight='bold', color='white' if comp['color'] != '#f39c12' else 'black')

        # Add arrows between components
        if i < len(components) - 1:
            next_comp = components[i + 1]
            arrow = patches.FancyArrowPatch(
                (x + w/2 + 0.1, y), (next_comp['pos'][0] - next_comp['size'][0]/2 - 0.1, next_comp['pos'][1]),
                connectionstyle="arc3", arrowstyle='->', mutation_scale=15,
                color='black', linewidth=2
            )
            ax.add_patch(arrow)

    # Add feature map visualization
    # Input image representation
    ax.add_patch(patches.Rectangle((0.1, 2.2), 0.8, 0.8, facecolor='lightblue', alpha=0.5))
    ax.text(0.5, 1.8, '28x28x1', fontsize=9, ha='center', fontweight='bold')

    # Feature maps after first conv
    for i in range(3):
        ax.add_patch(patches.Rectangle((2.2 + i*0.1, 2.5 + i*0.1), 0.6, 0.6,
                                     facecolor='lightcoral', alpha=0.7))
    ax.text(2.5, 2.0, '26x26x32', fontsize=9, ha='center', fontweight='bold')

    # Feature maps after first pooling
    for i in range(3):
        ax.add_patch(patches.Rectangle((5.1 + i*0.08, 3.0 + i*0.08), 0.4, 0.4,
                                     facecolor='orange', alpha=0.7))
    ax.text(5.0, 2.5, '13x13x32', fontsize=9, ha='center', fontweight='bold')

    # Add mathematical operations
    ax.text(8, 0.5, 'Key Operations:', fontsize=14, fontweight='bold')
    operations = [
        'Convolution: Feature Detection',
        'ReLU: Non-linearity',
        'MaxPool: Dimensionality Reduction',
        'Dense: Classification'
    ]

    for i, op in enumerate(operations):
        ax.text(8, 0.2 - i*0.3, f'• {op}', fontsize=11)

    # Set axis properties
    ax.set_xlim(-0.5, 16)
    ax.set_ylim(-1, 7)
    ax.axis('off')

    # Title
    plt.title('Convolutional Neural Network (CNN) Architecture\nImage Classification Pipeline',
              fontsize=18, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig

def create_rnn_lstm_diagram():
    """Create RNN/LSTM architecture diagram"""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # RNN Diagram (top)
    ax1.set_title('Recurrent Neural Network (RNN) Architecture', fontsize=16, fontweight='bold', pad=20)

    # RNN cells
    time_steps = 5
    for t in range(time_steps):
        x = t * 2 + 1

        # RNN cell
        cell = FancyBboxPatch((x-0.4, 1.5), 0.8, 1, boxstyle="round,pad=0.1",
                             facecolor='#e74c3c', alpha=0.7, edgecolor='black', linewidth=2)
        ax1.add_patch(cell)
        ax1.text(x, 2, f'RNN\nt={t+1}', fontsize=10, ha='center', va='center',
                fontweight='bold', color='white')

        # Input
        ax1.arrow(x, 0.8, 0, 0.5, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        ax1.text(x, 0.5, f'x{t+1}', fontsize=11, ha='center', va='center', fontweight='bold')

        # Output
        ax1.arrow(x, 2.7, 0, 0.5, head_width=0.1, head_length=0.1, fc='green', ec='green')
        ax1.text(x, 3.5, f'y{t+1}', fontsize=11, ha='center', va='center', fontweight='bold')

        # Hidden state connections
        if t < time_steps - 1:
            ax1.arrow(x+0.5, 2, 1, 0, head_width=0.1, head_length=0.15, fc='red', ec='red', linewidth=2)
            ax1.text(x+1, 2.3, f'h{t+1}', fontsize=10, ha='center', va='center',
                    fontweight='bold', color='red')

    # Add equation
    ax1.text(5, 4.2, r'$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$', fontsize=14, ha='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.5))
    ax1.text(5, 3.8, r'$y_t = W_{hy}h_t + b_y$', fontsize=14, ha='center')

    ax1.set_xlim(-0.5, 10.5)
    ax1.set_ylim(0, 5)
    ax1.axis('off')

    # LSTM Diagram (bottom)
    ax2.set_title('Long Short-Term Memory (LSTM) Cell Architecture', fontsize=16, fontweight='bold', pad=20)

    # LSTM cell components
    cell_x, cell_y = 5, 3

    # Main cell body
    main_cell = FancyBboxPatch((cell_x-2, cell_y-1.5), 4, 3, boxstyle="round,pad=0.2",
                              facecolor='lightblue', alpha=0.3, edgecolor='black', linewidth=3)
    ax2.add_patch(main_cell)

    # Gates
    gates = [
        {'name': 'Forget\nGate', 'pos': (cell_x-1.3, cell_y+0.5), 'color': '#e74c3c'},
        {'name': 'Input\nGate', 'pos': (cell_x-0.3, cell_y+0.5), 'color': '#f39c12'},
        {'name': 'Output\nGate', 'pos': (cell_x+0.7, cell_y+0.5), 'color': '#2ecc71'},
        {'name': 'Cell\nUpdate', 'pos': (cell_x+0.2, cell_y-0.5), 'color': '#9b59b6'}
    ]

    for gate in gates:
        x, y = gate['pos']
        rect = FancyBboxPatch((x-0.3, y-0.3), 0.6, 0.6, boxstyle="round,pad=0.05",
                             facecolor=gate['color'], alpha=0.7, edgecolor='black')
        ax2.add_patch(rect)
        ax2.text(x, y, gate['name'], fontsize=9, ha='center', va='center',
                fontweight='bold', color='white')

    # Cell state line
    ax2.arrow(1, cell_y, 2.5, 0, head_width=0.1, head_length=0.15, fc='purple', ec='purple', linewidth=3)
    ax2.arrow(6.5, cell_y, 2.5, 0, head_width=0.1, head_length=0.15, fc='purple', ec='purple', linewidth=3)
    ax2.text(2.5, cell_y+0.3, 'C_{t-1}', fontsize=12, ha='center', fontweight='bold', color='purple')
    ax2.text(7.5, cell_y+0.3, 'C_t', fontsize=12, ha='center', fontweight='bold', color='purple')

    # Input and output
    ax2.arrow(cell_x, 0.5, 0, 1, head_width=0.15, head_length=0.15, fc='blue', ec='blue', linewidth=2)
    ax2.text(cell_x, 0.2, 'x_t, h_{t-1}', fontsize=12, ha='center', fontweight='bold')

    ax2.arrow(cell_x, 4.8, 0, 0.7, head_width=0.15, head_length=0.15, fc='green', ec='green', linewidth=2)
    ax2.text(cell_x, 5.8, 'h_t', fontsize=12, ha='center', fontweight='bold')

    # Key equations
    equations = [
        r'$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$  (Forget Gate)',
        r'$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$  (Input Gate)',
        r'$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$  (Candidate)',
        r'$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$  (Cell State)',
        r'$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$  (Output Gate)',
        r'$h_t = o_t * \tanh(C_t)$  (Hidden State)'
    ]

    for i, eq in enumerate(equations):
        ax2.text(0.5, 1.5 - i*0.25, eq, fontsize=10, ha='left', va='center')

    ax2.set_xlim(0, 10)
    ax2.set_ylim(-1, 6.5)
    ax2.axis('off')

    plt.tight_layout()
    return fig

def create_transfer_learning_diagram():
    """Create transfer learning architecture diagram"""

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Pre-trained model section
    ax.text(2, 9, 'Pre-trained Model (e.g., ResNet-50)', fontsize=16, fontweight='bold', ha='center')
    ax.text(2, 8.6, 'Trained on ImageNet (1000 classes)', fontsize=12, ha='center', style='italic')

    # Draw pre-trained layers
    pretrained_layers = [
        {'name': 'Conv Block 1', 'pos': (1, 7.5), 'frozen': True},
        {'name': 'Conv Block 2', 'pos': (1, 6.8), 'frozen': True},
        {'name': 'Conv Block 3', 'pos': (1, 6.1), 'frozen': True},
        {'name': 'Conv Block 4', 'pos': (1, 5.4), 'frozen': True},
        {'name': 'Conv Block 5', 'pos': (1, 4.7), 'frozen': False},
        {'name': 'Global Pool', 'pos': (1, 4.0), 'frozen': False}
    ]

    for layer in pretrained_layers:
        x, y = layer['pos']
        color = '#95a5a6' if layer['frozen'] else '#3498db'
        label = '❄️ Frozen' if layer['frozen'] else '🔥 Fine-tuned'

        rect = FancyBboxPatch((x-0.7, y-0.2), 1.4, 0.4, boxstyle="round,pad=0.05",
                             facecolor=color, alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, y, layer['name'], fontsize=10, ha='center', va='center', fontweight='bold', color='white')
        ax.text(x+1.2, y, label, fontsize=9, ha='left', va='center')

    # Arrow to new layers
    ax.arrow(2, 3.5, 2, 0, head_width=0.15, head_length=0.2, fc='red', ec='red', linewidth=3)
    ax.text(3, 3.8, 'Feature\nExtraction', fontsize=11, ha='center', fontweight='bold', color='red')

    # New classification layers
    ax.text(6, 9, 'New Classification Head', fontsize=16, fontweight='bold', ha='center')
    ax.text(6, 8.6, 'Adapted for your specific task', fontsize=12, ha='center', style='italic')

    new_layers = [
        {'name': 'Dense 512', 'pos': (6, 6.5), 'color': '#e74c3c'},
        {'name': 'Dropout 0.5', 'pos': (6, 5.8), 'color': '#34495e'},
        {'name': 'Dense 256', 'pos': (6, 5.1), 'color': '#e74c3c'},
        {'name': 'Dropout 0.3', 'pos': (6, 4.4), 'color': '#34495e'},
        {'name': 'Output Layer', 'pos': (6, 3.7), 'color': '#2ecc71'}
    ]

    for layer in new_layers:
        x, y = layer['pos']
        rect = FancyBboxPatch((x-0.7, y-0.2), 1.4, 0.4, boxstyle="round,pad=0.05",
                             facecolor=layer['color'], alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, y, layer['name'], fontsize=10, ha='center', va='center',
                fontweight='bold', color='white')
        ax.text(x+1.2, y, '🆕 New & Trainable', fontsize=9, ha='left', va='center')

    # Training strategies comparison
    strategies = [
        {'title': 'Feature Extraction', 'x': 1, 'y': 2.5, 'color': '#3498db'},
        {'title': 'Fine-tuning', 'x': 4, 'y': 2.5, 'color': '#e74c3c'},
        {'title': 'From Scratch', 'x': 7, 'y': 2.5, 'color': '#f39c12'}
    ]

    strategy_details = [
        ['• Freeze pre-trained layers', '• Train only new classifier', '• Fast & stable', '• Good for small datasets'],
        ['• Unfreeze some layers', '• Lower learning rate', '• Better performance', '• Needs more data'],
        ['• Train entire network', '• Random initialization', '• Maximum flexibility', '• Requires large dataset']
    ]

    for i, (strategy, details) in enumerate(zip(strategies, strategy_details)):
        x, y = strategy['x'], strategy['y']

        # Strategy box
        rect = FancyBboxPatch((x-0.8, y-1), 1.6, 2, boxstyle="round,pad=0.1",
                             facecolor=strategy['color'], alpha=0.2, edgecolor=strategy['color'], linewidth=2)
        ax.add_patch(rect)

        # Title
        ax.text(x, y+0.7, strategy['title'], fontsize=12, ha='center', va='center',
                fontweight='bold', color=strategy['color'])

        # Details
        for j, detail in enumerate(details):
            ax.text(x, y+0.3-j*0.25, detail, fontsize=9, ha='center', va='center')

    # Add performance comparison chart
    ax.text(9.5, 8, 'Performance Comparison', fontsize=14, fontweight='bold', ha='center')

    # Simple bar chart
    methods = ['Scratch', 'Feature\nExtraction', 'Fine-tuning']
    performance = [0.75, 0.85, 0.92]
    colors = ['#f39c12', '#3498db', '#e74c3c']

    for i, (method, perf, color) in enumerate(zip(methods, performance, colors)):
        x = 9 + i * 0.4
        bar_height = perf * 2
        rect = plt.Rectangle((x-0.15, 5.5), 0.3, bar_height, facecolor=color, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, 5.3, method, fontsize=9, ha='center', va='top', rotation=0)
        ax.text(x, 5.5 + bar_height + 0.1, f'{perf:.0%}', fontsize=10, ha='center', va='bottom', fontweight='bold')

    ax.text(9.4, 5.0, 'Typical Accuracy', fontsize=10, ha='center', style='italic')

    # Set axis properties
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(0, 10)
    ax.axis('off')

    plt.title('Transfer Learning Architecture & Strategies\nLeveraging Pre-trained Deep Networks',
              fontsize=18, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig

def save_all_diagrams():
    """Generate and save all architecture diagrams"""

    print("Generating Deep Neural Network Architecture Diagrams...")

    # Create output directory
    import os
    output_dir = "architecture_diagrams"
    os.makedirs(output_dir, exist_ok=True)

    # Generate all diagrams
    diagrams = [
        (create_basic_dnn_diagram, "basic_dnn_architecture.png"),
        (create_cnn_architecture_diagram, "cnn_architecture.png"),
        (create_rnn_lstm_diagram, "rnn_lstm_architecture.png"),
        (create_transfer_learning_diagram, "transfer_learning_architecture.png")
    ]

    for create_func, filename in diagrams:
        print(f"Creating {filename}...")
        fig = create_func()
        fig.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"✓ Saved: {output_dir}/{filename}")

    print(f"\nAll diagrams saved to '{output_dir}/' directory")
    print("Diagrams created:")
    print("1. basic_dnn_architecture.png - Multi-layer Perceptron")
    print("2. cnn_architecture.png - Convolutional Neural Network")
    print("3. rnn_lstm_architecture.png - Recurrent Networks")
    print("4. transfer_learning_architecture.png - Transfer Learning")

if __name__ == "__main__":
    save_all_diagrams()