"""
Week 5 - Tutorial Task T6: Gradient Descent Optimization
Module 2: Optimization and Regularization

This lab exercise implements various gradient descent optimization techniques
and demonstrates their effects on neural network training.

Author: Deep Neural Network Architectures Course
Date: September 2025
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time


class GradientDescentVisualizer:
    """
    Visualizer for different gradient descent optimization algorithms
    """

    def __init__(self, random_seed: int = 42):
        """Initialize the visualizer with reproducible results"""
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
        self.history = {}

    def create_dataset(self, n_samples: int = 1000) -> Tuple:
        """
        Create a synthetic dataset for binary classification

        Args:
            n_samples: Number of samples to generate

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        from sklearn.datasets import make_moons
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        # Generate moon-shaped data for interesting decision boundaries
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=42)

        # Split the data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def create_base_model(self, input_dim: int = 2) -> tf.keras.Model:
        """
        Create a base neural network model

        Args:
            input_dim: Input dimension

        Returns:
            Uncompiled Keras model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    def train_with_optimizer(
        self,
        optimizer_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32
    ) -> Dict:
        """
        Train model with specified optimizer

        Args:
            optimizer_name: Name of optimizer to use
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Training history dictionary
        """
        # Create fresh model
        model = self.create_base_model(X_train.shape[1])

        # Select optimizer
        if optimizer_name == 'SGD':
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        elif optimizer_name == 'SGD_Momentum':
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        elif optimizer_name == 'Adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        elif optimizer_name == 'RMSprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
        elif optimizer_name == 'Adagrad':
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Train model
        print(f"\nTraining with {optimizer_name}...")
        start_time = time.time()

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        training_time = time.time() - start_time

        # Store results
        self.history[optimizer_name] = {
            'history': history.history,
            'model': model,
            'time': training_time
        }

        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

        return history.history

    def compare_optimizers(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        optimizers: List[str] = None,
        epochs: int = 50
    ):
        """
        Compare multiple optimizers

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            optimizers: List of optimizer names to compare
            epochs: Number of training epochs
        """
        if optimizers is None:
            optimizers = ['SGD', 'SGD_Momentum', 'Adam', 'RMSprop', 'Adagrad']

        # Train with each optimizer
        for opt_name in optimizers:
            self.train_with_optimizer(
                opt_name, X_train, y_train, X_val, y_val, epochs
            )

        # Visualize comparison
        self._plot_comparison()

    def _plot_comparison(self):
        """Plot comparison of all trained optimizers"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot training loss
        ax = axes[0, 0]
        for name, data in self.history.items():
            ax.plot(data['history']['loss'], label=name)
        ax.set_title('Training Loss Comparison')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)

        # Plot validation loss
        ax = axes[0, 1]
        for name, data in self.history.items():
            ax.plot(data['history']['val_loss'], label=name)
        ax.set_title('Validation Loss Comparison')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)

        # Plot training accuracy
        ax = axes[1, 0]
        for name, data in self.history.items():
            ax.plot(data['history']['accuracy'], label=name)
        ax.set_title('Training Accuracy Comparison')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True)

        # Plot validation accuracy
        ax = axes[1, 1]
        for name, data in self.history.items():
            ax.plot(data['history']['val_accuracy'], label=name)
        ax.set_title('Validation Accuracy Comparison')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

        # Print summary table
        self._print_summary_table()

    def _print_summary_table(self):
        """Print summary table of optimizer performance"""
        print("\n" + "="*70)
        print("OPTIMIZER COMPARISON SUMMARY")
        print("="*70)
        print(f"{'Optimizer':<15} {'Final Loss':<12} {'Final Acc':<12} {'Time (s)':<10}")
        print("-"*70)

        for name, data in self.history.items():
            final_loss = data['history']['val_loss'][-1]
            final_acc = data['history']['val_accuracy'][-1]
            train_time = data['time']
            print(f"{name:<15} {final_loss:<12.4f} {final_acc:<12.4f} {train_time:<10.2f}")


class GradientProblemsDemonstrator:
    """
    Demonstrate and solve gradient vanishing/exploding problems
    """

    def __init__(self):
        self.models = {}

    def create_deep_network(
        self,
        depth: int = 10,
        activation: str = 'sigmoid',
        initialization: str = 'random_normal'
    ) -> tf.keras.Model:
        """
        Create a deep network to demonstrate gradient problems

        Args:
            depth: Number of layers
            activation: Activation function to use
            initialization: Weight initialization method

        Returns:
            Keras model
        """
        model = tf.keras.Sequential()

        # Input layer
        model.add(tf.keras.layers.Dense(
            64, activation=activation,
            kernel_initializer=initialization,
            input_shape=(10,)
        ))

        # Hidden layers
        for _ in range(depth - 2):
            model.add(tf.keras.layers.Dense(
                64, activation=activation,
                kernel_initializer=initialization
            ))

        # Output layer
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        return model

    def analyze_gradient_flow(self, model: tf.keras.Model, X_sample: tf.Tensor) -> List[float]:
        """
        Analyze gradient flow through the network

        Args:
            model: Keras model to analyze
            X_sample: Sample input data

        Returns:
            List of gradient norms for each layer
        """
        with tf.GradientTape() as tape:
            y_pred = model(X_sample)
            loss = tf.reduce_mean(tf.square(y_pred - 1.0))

        gradients = tape.gradient(loss, model.trainable_variables)

        gradient_norms = []
        layer_names = []

        for i, (grad, var) in enumerate(zip(gradients, model.trainable_variables)):
            if 'kernel' in var.name:  # Only weights, not biases
                norm = tf.norm(grad).numpy()
                gradient_norms.append(norm)
                layer_names.append(f"Layer {i//2}")

        return gradient_norms, layer_names

    def demonstrate_vanishing_gradients(self):
        """Demonstrate vanishing gradient problem"""
        print("\n" + "="*60)
        print("DEMONSTRATING VANISHING GRADIENTS")
        print("="*60)

        # Create networks with different activations
        sigmoid_model = self.create_deep_network(depth=10, activation='sigmoid')
        relu_model = self.create_deep_network(depth=10, activation='relu')

        # Generate sample data
        X_sample = tf.random.normal((32, 10))

        # Analyze gradients
        sigmoid_grads, layer_names = self.analyze_gradient_flow(sigmoid_model, X_sample)
        relu_grads, _ = self.analyze_gradient_flow(relu_model, X_sample)

        # Visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Sigmoid gradients
        ax1.semilogy(range(len(sigmoid_grads)), sigmoid_grads, 'ro-', linewidth=2)
        ax1.set_title('Vanishing Gradients with Sigmoid Activation')
        ax1.set_xlabel('Layer Depth')
        ax1.set_ylabel('Gradient Norm (log scale)')
        ax1.grid(True, alpha=0.3)

        # ReLU gradients
        ax2.semilogy(range(len(relu_grads)), relu_grads, 'bo-', linewidth=2)
        ax2.set_title('Gradient Flow with ReLU Activation')
        ax2.set_xlabel('Layer Depth')
        ax2.set_ylabel('Gradient Norm (log scale)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print analysis
        print("\nGradient Analysis:")
        print(f"Sigmoid - First layer gradient: {sigmoid_grads[0]:.6f}")
        print(f"Sigmoid - Last layer gradient: {sigmoid_grads[-1]:.6f}")
        print(f"Sigmoid - Gradient ratio: {sigmoid_grads[-1]/sigmoid_grads[0]:.6f}")
        print(f"\nReLU - First layer gradient: {relu_grads[0]:.6f}")
        print(f"ReLU - Last layer gradient: {relu_grads[-1]:.6f}")
        print(f"ReLU - Gradient ratio: {relu_grads[-1]/relu_grads[0]:.6f}")

    def demonstrate_gradient_clipping(self):
        """Demonstrate gradient clipping to prevent explosion"""
        print("\n" + "="*60)
        print("DEMONSTRATING GRADIENT CLIPPING")
        print("="*60)

        # Create a model prone to gradient explosion
        model = self.create_deep_network(
            depth=5,
            activation='relu',
            initialization=tf.keras.initializers.RandomNormal(mean=0, stddev=5)
        )

        # Generate sample data
        X = tf.random.normal((100, 10))
        y = tf.random.uniform((100, 1))

        # Training without clipping
        print("\nTraining WITHOUT gradient clipping:")
        optimizer_no_clip = tf.keras.optimizers.SGD(learning_rate=0.01)

        gradient_norms_no_clip = []
        for epoch in range(10):
            with tf.GradientTape() as tape:
                y_pred = model(X)
                loss = tf.reduce_mean(tf.square(y_pred - y))

            grads = tape.gradient(loss, model.trainable_variables)
            total_norm = tf.reduce_sum([tf.norm(g) for g in grads if g is not None])
            gradient_norms_no_clip.append(total_norm.numpy())

            if total_norm > 100:
                print(f"Epoch {epoch}: Gradient explosion! Norm = {total_norm:.2f}")

        # Reset model
        model = self.create_deep_network(
            depth=5,
            activation='relu',
            initialization=tf.keras.initializers.RandomNormal(mean=0, stddev=5)
        )

        # Training with clipping
        print("\nTraining WITH gradient clipping (max_norm=1.0):")
        optimizer_clip = tf.keras.optimizers.SGD(learning_rate=0.01, clipnorm=1.0)

        gradient_norms_clip = []
        for epoch in range(10):
            with tf.GradientTape() as tape:
                y_pred = model(X)
                loss = tf.reduce_mean(tf.square(y_pred - y))

            grads = tape.gradient(loss, model.trainable_variables)

            # Manual clipping for demonstration
            clipped_grads = []
            for grad in grads:
                if grad is not None:
                    clipped_grads.append(tf.clip_by_norm(grad, 1.0))
                else:
                    clipped_grads.append(grad)

            total_norm = tf.reduce_sum([tf.norm(g) for g in clipped_grads if g is not None])
            gradient_norms_clip.append(total_norm.numpy())
            print(f"Epoch {epoch}: Gradient norm (clipped) = {total_norm:.2f}")

        # Visualize the difference
        plt.figure(figsize=(10, 6))
        plt.plot(gradient_norms_no_clip, 'r-', label='Without Clipping', linewidth=2)
        plt.plot(gradient_norms_clip, 'b-', label='With Clipping', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Total Gradient Norm')
        plt.title('Effect of Gradient Clipping')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def main():
    """Main function to run all demonstrations"""
    print("="*70)
    print("TUTORIAL TASK T6: GRADIENT DESCENT OPTIMIZATION")
    print("Module 2: Optimization and Regularization")
    print("="*70)

    # Part 1: Compare optimizers
    print("\n" + "="*70)
    print("PART 1: COMPARING GRADIENT DESCENT OPTIMIZERS")
    print("="*70)

    visualizer = GradientDescentVisualizer()

    # Create dataset
    X_train, y_train, X_val, y_val, X_test, y_test = visualizer.create_dataset(1000)

    print(f"\nDataset created:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Compare optimizers
    visualizer.compare_optimizers(
        X_train, y_train, X_val, y_val,
        optimizers=['SGD', 'SGD_Momentum', 'Adam', 'RMSprop'],
        epochs=50
    )

    # Part 2: Demonstrate gradient problems
    print("\n" + "="*70)
    print("PART 2: GRADIENT PROBLEMS AND SOLUTIONS")
    print("="*70)

    demonstrator = GradientProblemsDemonstrator()

    # Demonstrate vanishing gradients
    demonstrator.demonstrate_vanishing_gradients()

    # Demonstrate gradient clipping
    demonstrator.demonstrate_gradient_clipping()

    print("\n" + "="*70)
    print("TUTORIAL COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Adam and RMSprop converge faster than vanilla SGD")
    print("2. Momentum helps SGD escape local minima")
    print("3. Sigmoid activation causes vanishing gradients in deep networks")
    print("4. ReLU activation maintains gradient flow better")
    print("5. Gradient clipping prevents training instability")
    print("\nNext steps: Implement these optimizations in your own neural networks!")


if __name__ == "__main__":
    main()