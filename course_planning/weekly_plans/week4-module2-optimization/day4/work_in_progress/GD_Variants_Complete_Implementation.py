"""
Week 4 - Day 4: Gradient Descent Variants - Complete Implementation
Course: Deep Neural Network Architectures (21CSE558T)
Module 2: Optimization and Regularization

This file contains complete, runnable implementations of all three
gradient descent variants with comprehensive analysis and visualization.

Author: Course Instructor
Date: Week 4, Day 4
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Tuple, List

# Set random seed for reproducibility
np.random.seed(42)

class GradientDescentComparison:
    """
    A comprehensive class to implement and compare all three gradient descent variants
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize with dataset
        
        Args:
            X: Input features (m, n)
            y: Target values (m,)
        """
        self.X = X
        self.y = y
        self.m = len(X)
        self.n = X.shape[1] if len(X.shape) > 1 else 1
        
        print(f"Dataset initialized:")
        print(f"  Examples (m): {self.m}")
        print(f"  Features (n): {self.n}")
        print(f"  X range: [{X.min():.2f}, {X.max():.2f}]")
        print(f"  y range: [{y.min():.2f}, {y.max():.2f}]")
    
    def _compute_cost(self, predictions: np.ndarray, y_true: np.ndarray) -> float:
        """Compute Mean Squared Error cost"""
        return np.mean((predictions - y_true)**2)
    
    def _initialize_parameters(self) -> Tuple[float, float]:
        """Initialize weight and bias with small random values"""
        w = np.random.randn() * 0.01
        b = np.random.randn() * 0.01
        return w, b
    
    def batch_gradient_descent(self, epochs: int = 100, learning_rate: float = 0.1, 
                             verbose: bool = True) -> Tuple[float, float, List[float], float]:
        """
        Batch Gradient Descent Implementation
        
        Uses the entire dataset to compute gradient at each step.
        Provides smooth, stable convergence but is slow for large datasets.
        
        Args:
            epochs: Number of training epochs
            learning_rate: Step size for parameter updates
            verbose: Whether to print progress updates
            
        Returns:
            w: Final weight parameter
            b: Final bias parameter
            costs: List of costs per epoch
            time_taken: Execution time in seconds
        """
        start_time = time.time()
        
        # Initialize parameters
        w, b = self._initialize_parameters()
        costs = []
        
        if verbose:
            print("\n" + "="*50)
            print("BATCH GRADIENT DESCENT")
            print("="*50)
            print(f"Initial: w = {w:.6f}, b = {b:.6f}")
            print(f"Learning rate: {learning_rate}")
            print(f"Dataset size: {self.m} examples")
            print("Updates per epoch: 1")
        
        for epoch in range(epochs):
            # Forward pass - compute predictions for ALL examples
            predictions = w * self.X.squeeze() + b
            
            # Compute cost using entire dataset
            cost = self._compute_cost(predictions, self.y)
            costs.append(cost)
            
            # Compute gradients using ALL examples
            # dJ/dw = (1/m) * Σ(predictions - y) * X
            # dJ/db = (1/m) * Σ(predictions - y)
            dw = np.mean((predictions - self.y) * self.X.squeeze())
            db = np.mean(predictions - self.y)
            
            # Parameter update - SINGLE update per epoch
            w -= learning_rate * dw
            b -= learning_rate * db
            
            # Progress logging
            if verbose and (epoch % 20 == 0 or epoch < 10):
                print(f"Epoch {epoch:3d}: Cost = {cost:.8f}, w = {w:.6f}, b = {b:.6f}")
        
        time_taken = time.time() - start_time
        
        if verbose:
            print(f"\nFinal Results:")
            print(f"  w = {w:.6f}, b = {b:.6f}")
            print(f"  Final cost = {costs[-1]:.8f}")
            print(f"  Time taken = {time_taken:.3f} seconds")
            print(f"  Total parameter updates: {epochs}")
        
        return w, b, costs, time_taken
    
    def stochastic_gradient_descent(self, epochs: int = 100, learning_rate: float = 0.01,
                                  verbose: bool = True) -> Tuple[float, float, List[float], float]:
        """
        Stochastic Gradient Descent Implementation
        
        Updates parameters after each individual training example.
        Fast updates but noisy convergence pattern.
        
        Args:
            epochs: Number of training epochs
            learning_rate: Step size (typically smaller than BGD)
            verbose: Whether to print progress updates
            
        Returns:
            w: Final weight parameter
            b: Final bias parameter
            costs: List of average costs per epoch
            time_taken: Execution time in seconds
        """
        start_time = time.time()
        
        # Initialize parameters
        w, b = self._initialize_parameters()
        costs = []
        
        if verbose:
            print("\n" + "="*50)
            print("STOCHASTIC GRADIENT DESCENT")
            print("="*50)
            print(f"Initial: w = {w:.6f}, b = {b:.6f}")
            print(f"Learning rate: {learning_rate} (smaller than BGD)")
            print(f"Dataset size: {self.m} examples")
            print(f"Updates per epoch: {self.m}")
        
        for epoch in range(epochs):
            epoch_cost = 0
            
            # Shuffle data each epoch for better convergence
            indices = np.random.permutation(self.m)
            
            # Process each example individually
            for idx in indices:
                # Forward pass - SINGLE example
                x_i = self.X[idx]
                y_i = self.y[idx]
                prediction = w * x_i + b
                
                # Accumulate cost for epoch average
                cost_i = (prediction - y_i)**2
                epoch_cost += cost_i
                
                # Compute gradients for SINGLE example
                # dJ/dw = (prediction - y_i) * x_i
                # dJ/db = (prediction - y_i)
                dw = (prediction - y_i) * x_i
                db = (prediction - y_i)
                
                # Immediate parameter update
                w -= learning_rate * dw
                b -= learning_rate * db
            
            # Average cost for the epoch
            avg_cost = epoch_cost / self.m
            costs.append(avg_cost)
            
            # Progress logging
            if verbose and (epoch % 20 == 0 or epoch < 10):
                print(f"Epoch {epoch:3d}: Cost = {avg_cost:.8f}, w = {w:.6f}, b = {b:.6f}")
        
        time_taken = time.time() - start_time
        
        if verbose:
            print(f"\nFinal Results:")
            print(f"  w = {w:.6f}, b = {b:.6f}")
            print(f"  Final cost = {costs[-1]:.8f}")
            print(f"  Time taken = {time_taken:.3f} seconds")
            print(f"  Total parameter updates: {epochs * self.m}")
        
        return w, b, costs, time_taken
    
    def mini_batch_gradient_descent(self, batch_size: int = 32, epochs: int = 100,
                                  learning_rate: float = 0.05, verbose: bool = True
                                  ) -> Tuple[float, float, List[float], float]:
        """
        Mini-batch Gradient Descent Implementation
        
        Updates parameters using small batches of examples.
        Balances the stability of BGD with the speed of SGD.
        
        Args:
            batch_size: Number of examples per batch
            epochs: Number of training epochs
            learning_rate: Step size (between BGD and SGD rates)
            verbose: Whether to print progress updates
            
        Returns:
            w: Final weight parameter
            b: Final bias parameter
            costs: List of average costs per epoch
            time_taken: Execution time in seconds
        """
        start_time = time.time()
        
        # Initialize parameters
        w, b = self._initialize_parameters()
        costs = []
        
        # Calculate number of batches per epoch
        num_batches = int(np.ceil(self.m / batch_size))
        
        if verbose:
            print("\n" + "="*50)
            print("MINI-BATCH GRADIENT DESCENT")
            print("="*50)
            print(f"Initial: w = {w:.6f}, b = {b:.6f}")
            print(f"Learning rate: {learning_rate}")
            print(f"Dataset size: {self.m} examples")
            print(f"Batch size: {batch_size}")
            print(f"Batches per epoch: {num_batches}")
            print(f"Updates per epoch: {num_batches}")
        
        for epoch in range(epochs):
            epoch_cost = 0
            
            # Shuffle data each epoch
            indices = np.random.permutation(self.m)
            
            # Process data in mini-batches
            for i in range(0, self.m, batch_size):
                # Create mini-batch
                end_idx = min(i + batch_size, self.m)
                batch_indices = indices[i:end_idx]
                
                X_batch = self.X[batch_indices]
                y_batch = self.y[batch_indices]
                batch_m = len(X_batch)
                
                # Forward pass - BATCH of examples
                predictions = w * X_batch.squeeze() + b
                
                # Compute cost for this batch
                batch_cost = self._compute_cost(predictions, y_batch)
                epoch_cost += batch_cost * batch_m
                
                # Compute gradients for BATCH
                # dJ/dw = (1/batch_m) * Σ(predictions - y_batch) * X_batch
                # dJ/db = (1/batch_m) * Σ(predictions - y_batch)
                dw = np.mean((predictions - y_batch) * X_batch.squeeze())
                db = np.mean(predictions - y_batch)
                
                # Parameter update after each batch
                w -= learning_rate * dw
                b -= learning_rate * db
            
            # Average cost for the epoch
            avg_cost = epoch_cost / self.m
            costs.append(avg_cost)
            
            # Progress logging
            if verbose and (epoch % 20 == 0 or epoch < 10):
                print(f"Epoch {epoch:3d}: Cost = {avg_cost:.8f}, w = {w:.6f}, b = {b:.6f}")
        
        time_taken = time.time() - start_time
        
        if verbose:
            print(f"\nFinal Results:")
            print(f"  w = {w:.6f}, b = {b:.6f}")
            print(f"  Final cost = {costs[-1]:.8f}")
            print(f"  Time taken = {time_taken:.3f} seconds")
            print(f"  Total parameter updates: {epochs * num_batches}")
        
        return w, b, costs, time_taken
    
    def compare_all_methods(self, epochs: int = 100, plot_results: bool = True):
        """
        Run all three gradient descent variants and compare results
        
        Args:
            epochs: Number of epochs for each algorithm
            plot_results: Whether to create comparison plots
        """
        print("Starting comprehensive comparison of gradient descent variants...")
        
        # Run all three algorithms
        w_batch, b_batch, costs_batch, time_batch = self.batch_gradient_descent(
            epochs=epochs, learning_rate=0.1, verbose=True
        )
        
        w_sgd, b_sgd, costs_sgd, time_sgd = self.stochastic_gradient_descent(
            epochs=epochs, learning_rate=0.01, verbose=True
        )
        
        w_mb, b_mb, costs_mb, time_mb = self.mini_batch_gradient_descent(
            batch_size=32, epochs=epochs, learning_rate=0.05, verbose=True
        )
        
        # Print comprehensive comparison
        print("\n" + "="*80)
        print("COMPREHENSIVE COMPARISON RESULTS")
        print("="*80)
        
        # Create comparison table
        methods = ["Batch GD", "Stochastic GD", "Mini-batch GD"]
        weights = [w_batch, w_sgd, w_mb]
        biases = [b_batch, b_sgd, b_mb]
        final_costs = [costs_batch[-1], costs_sgd[-1], costs_mb[-1]]
        times = [time_batch, time_sgd, time_mb]
        
        print(f"{'Method':<15} {'Weight':<10} {'Bias':<10} {'Final Cost':<12} {'Time (s)':<10}")
        print("-" * 65)
        for i in range(3):
            print(f"{methods[i]:<15} {weights[i]:<10.4f} {biases[i]:<10.4f} "
                  f"{final_costs[i]:<12.8f} {times[i]:<10.3f}")
        
        # True parameters for comparison (assuming y = 2x + 1 + noise)
        print(f"\nTrue parameters: w = 2.0000, b = 1.0000")
        
        # Error analysis
        print(f"\nError Analysis (|predicted - true|):")
        w_errors = [abs(w - 2.0) for w in weights]
        b_errors = [abs(b - 1.0) for b in biases]
        
        for i in range(3):
            print(f"{methods[i]:<15} Weight Error: {w_errors[i]:.6f}, "
                  f"Bias Error: {b_errors[i]:.6f}")
        
        if plot_results:
            self._plot_comparison(costs_batch, costs_sgd, costs_mb, methods)
        
        return {
            'batch': (w_batch, b_batch, costs_batch, time_batch),
            'sgd': (w_sgd, b_sgd, costs_sgd, time_sgd),
            'minibatch': (w_mb, b_mb, costs_mb, time_mb)
        }
    
    def _plot_comparison(self, costs_batch: List[float], costs_sgd: List[float], 
                        costs_mb: List[float], methods: List[str]):
        """Create comprehensive comparison plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Gradient Descent Variants Comparison', fontsize=16, fontweight='bold')
        
        # Individual convergence plots
        costs_data = [costs_batch, costs_sgd, costs_mb]
        colors = ['blue', 'red', 'green']
        
        for i, (costs, method, color) in enumerate(zip(costs_data, methods, colors)):
            row, col = i // 2, i % 2
            if i < 3:  # Only plot first 3 methods
                axes[row, col].plot(costs, color=color, linewidth=2, alpha=0.8)
                axes[row, col].set_title(f'{method}: Convergence Pattern')
                axes[row, col].set_xlabel('Epoch')
                axes[row, col].set_ylabel('Cost (MSE)')
                axes[row, col].grid(True, alpha=0.3)
                axes[row, col].set_yscale('log')  # Log scale for better visualization
        
        # Combined comparison plot
        axes[1, 1].plot(costs_batch, 'b-', linewidth=2, label='Batch GD', alpha=0.8)
        axes[1, 1].plot(costs_sgd, 'r-', linewidth=2, label='Stochastic GD', alpha=0.7)
        axes[1, 1].plot(costs_mb, 'g-', linewidth=2, label='Mini-batch GD', alpha=0.8)
        axes[1, 1].set_title('All Methods Combined')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Cost (MSE)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def experiment_batch_sizes(self, batch_sizes: List[int] = [1, 8, 16, 32, 64, 128],
                             epochs: int = 50):
        """
        Experiment with different batch sizes for mini-batch GD
        
        Args:
            batch_sizes: List of batch sizes to test
            epochs: Number of epochs for each experiment
        """
        print(f"\nExperimenting with different batch sizes...")
        print(f"Batch sizes to test: {batch_sizes}")
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch_size = {batch_size}")
            
            if batch_size >= self.m:
                # If batch size >= dataset size, this becomes batch GD
                w, b, costs, time_taken = self.batch_gradient_descent(
                    epochs=epochs, verbose=False
                )
                method_name = "Batch GD"
            else:
                w, b, costs, time_taken = self.mini_batch_gradient_descent(
                    batch_size=batch_size, epochs=epochs, verbose=False
                )
                method_name = f"Mini-batch (size={batch_size})"
            
            results[batch_size] = {
                'w': w, 'b': b, 'final_cost': costs[-1], 
                'time': time_taken, 'costs': costs,
                'method': method_name
            }
            
            print(f"{method_name}: w={w:.4f}, b={b:.4f}, "
                  f"final_cost={costs[-1]:.6f}, time={time_taken:.3f}s")
        
        # Plot batch size comparison
        self._plot_batch_size_comparison(results, batch_sizes)
        
        return results
    
    def _plot_batch_size_comparison(self, results: dict, batch_sizes: List[int]):
        """Plot comparison of different batch sizes"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot convergence curves
        for batch_size in batch_sizes:
            costs = results[batch_size]['costs']
            ax1.plot(costs, label=f'Batch size: {batch_size}', linewidth=2)
        
        ax1.set_title('Convergence vs Batch Size')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cost (MSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot final performance metrics
        final_costs = [results[bs]['final_cost'] for bs in batch_sizes]
        times = [results[bs]['time'] for bs in batch_sizes]
        
        ax2_twin = ax2.twinx()
        
        bars1 = ax2.bar([str(bs) for bs in batch_sizes], final_costs, 
                       alpha=0.7, color='blue', label='Final Cost')
        bars2 = ax2_twin.bar([str(bs) for bs in batch_sizes], times, 
                           alpha=0.7, color='red', label='Time (s)')
        
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Final Cost', color='blue')
        ax2_twin.set_ylabel('Time (seconds)', color='red')
        ax2.set_title('Performance vs Batch Size')
        
        # Add legends
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()


def create_synthetic_dataset(m: int = 1000, noise_std: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic linear regression dataset
    
    Args:
        m: Number of examples
        noise_std: Standard deviation of noise
        
    Returns:
        X: Input features (m, 1)
        y: Target values (m,) following y = 2x + 1 + noise
    """
    np.random.seed(42)  # For reproducibility
    X = np.random.randn(m, 1)
    y = 2 * X.squeeze() + 1 + noise_std * np.random.randn(m)
    
    print(f"Created synthetic dataset:")
    print(f"  Relationship: y = 2x + 1 + noise")
    print(f"  Examples: {m}")
    print(f"  Noise std: {noise_std}")
    print(f"  X range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"  y range: [{y.min():.2f}, {y.max():.2f}]")
    
    return X, y


def main_demonstration():
    """
    Main demonstration function to run complete gradient descent comparison
    """
    print("Week 4 - Day 4: Gradient Descent Variants Demonstration")
    print("="*60)
    
    # Create synthetic dataset
    X, y = create_synthetic_dataset(m=1000, noise_std=0.1)
    
    # Initialize comparison class
    gd_comparison = GradientDescentComparison(X, y)
    
    # Run comprehensive comparison
    results = gd_comparison.compare_all_methods(epochs=100, plot_results=True)
    
    # Experiment with different batch sizes
    batch_size_results = gd_comparison.experiment_batch_sizes(
        batch_sizes=[1, 8, 16, 32, 64, 128], epochs=50
    )
    
    print("\nDemonstration completed!")
    print("Key takeaways:")
    print("1. Batch GD: Stable but slow")
    print("2. Stochastic GD: Fast updates but noisy")
    print("3. Mini-batch GD: Best balance for most problems")
    print("4. Batch size affects both speed and stability")


# Interactive exercises for students
def student_exercises():
    """
    Interactive exercises for students to try
    """
    print("\nSTUDENT EXERCISES")
    print("="*40)
    
    # Exercise 1: Parameter sensitivity
    print("\nExercise 1: Learning Rate Sensitivity")
    print("Try different learning rates and observe convergence patterns")
    
    X, y = create_synthetic_dataset(m=500)
    gd = GradientDescentComparison(X, y)
    
    learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
    
    for lr in learning_rates:
        print(f"\nTesting learning_rate = {lr}")
        try:
            w, b, costs, _ = gd.mini_batch_gradient_descent(
                batch_size=32, epochs=50, learning_rate=lr, verbose=False
            )
            print(f"  Result: w={w:.4f}, b={b:.4f}, final_cost={costs[-1]:.6f}")
            if costs[-1] > 10:
                print("  WARNING: High final cost - learning rate might be too large!")
        except:
            print("  ERROR: Learning rate too large - diverged!")
    
    # Exercise 2: Dataset size impact
    print("\nExercise 2: Dataset Size Impact")
    dataset_sizes = [100, 500, 1000, 5000]
    
    for size in dataset_sizes:
        X, y = create_synthetic_dataset(m=size)
        gd = GradientDescentComparison(X, y)
        
        # Time different methods
        start = time.time()
        _, _, _, _ = gd.batch_gradient_descent(epochs=20, verbose=False)
        batch_time = time.time() - start
        
        start = time.time()
        _, _, _, _ = gd.mini_batch_gradient_descent(
            batch_size=32, epochs=20, verbose=False
        )
        mb_time = time.time() - start
        
        print(f"Dataset size {size}: Batch GD time = {batch_time:.3f}s, "
              f"Mini-batch time = {mb_time:.3f}s")


if __name__ == "__main__":
    # Run main demonstration
    main_demonstration()
    
    # Run student exercises
    student_exercises()
    
    print("\nComplete implementation finished!")
    print("Students can modify parameters and experiment with different settings.")