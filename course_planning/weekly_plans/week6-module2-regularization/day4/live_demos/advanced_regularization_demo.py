#!/usr/bin/env python3
"""
Week 6 Day 4: Advanced Regularization Live Demonstrations
Course: 21CSE558T - Deep Neural Network Architectures
Author: Prof. Ramesh Babu | SRM University

This file contains all live demonstration code for the 1-hour lecture on
advanced regularization techniques: Dropout, Batch Normalization, Early Stopping.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DropoutDemo:
    """
    Live Demonstration 1: Dropout Effectiveness
    Compare three models with different dropout strategies
    """

    def __init__(self):
        self.models = {}
        self.histories = {}
        print("🎯 Dropout Demo Initialized")

    def create_models(self):
        """Create three models with different dropout strategies"""

        # Model 1: No Dropout (Baseline)
        self.models['no_dropout'] = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ], name='NoDropout')

        # Model 2: Conservative Dropout (0.2)
        self.models['conservative'] = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ], name='Conservative_Dropout')

        # Model 3: Aggressive Dropout (0.5)
        self.models['aggressive'] = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax')
        ], name='Aggressive_Dropout')

        print("✅ Three dropout models created")
        return self.models

    def prepare_data(self):
        """Prepare MNIST data for training"""
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Normalize and reshape
        X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
        X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

        # Create validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        print(f"📊 Data prepared: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_all_models(self, X_train, y_train, X_val, y_val, epochs=15):
        """Train all three models and compare performance"""

        print("🚀 Starting training comparison...")

        for name, model in self.models.items():
            print(f"\n🔄 Training {name} model...")

            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            start_time = time.time()
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=128,
                verbose=1 if name == 'no_dropout' else 0  # Show progress for first model only
            )

            training_time = time.time() - start_time
            self.histories[name] = history

            # Print final metrics
            final_train_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            overfitting_gap = final_train_acc - final_val_acc

            print(f"✅ {name}: Train={final_train_acc:.3f}, Val={final_val_acc:.3f}, Gap={overfitting_gap:.3f}, Time={training_time:.1f}s")

    def visualize_results(self):
        """Create comprehensive visualization of dropout effects"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Training Accuracy
        axes[0, 0].set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
        for name, history in self.histories.items():
            axes[0, 0].plot(history.history['accuracy'], label=f'{name}', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Validation Accuracy
        axes[0, 1].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
        for name, history in self.histories.items():
            axes[0, 1].plot(history.history['val_accuracy'], label=f'{name}', linewidth=2, linestyle='--')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Loss Comparison
        axes[1, 0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
        for name, history in self.histories.items():
            axes[1, 0].plot(history.history['loss'], label=f'{name}', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')

        # Plot 4: Overfitting Gap Analysis
        axes[1, 1].set_title('Overfitting Gap (Train - Val Accuracy)', fontsize=14, fontweight='bold')
        for name, history in self.histories.items():
            train_acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            gap = [t - v for t, v in zip(train_acc, val_acc)]
            axes[1, 1].plot(gap, label=f'{name}', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Gap')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='red', linestyle='-', alpha=0.5)

        plt.tight_layout()
        plt.suptitle('🎯 Dropout Effectiveness Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.show()


class BatchNormalizationDemo:
    """
    Live Demonstration 2: Batch Normalization Impact
    Compare deep networks with and without batch normalization
    """

    def __init__(self):
        self.models = {}
        self.histories = {}
        print("⚡ Batch Normalization Demo Initialized")

    def create_deep_models(self):
        """Create deep models with and without batch normalization"""

        # Model 1: Deep network WITHOUT Batch Normalization
        self.models['without_bn'] = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ], name='Without_BatchNorm')

        # Model 2: Deep network WITH Batch Normalization
        self.models['with_bn'] = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(10, activation='softmax')
        ], name='With_BatchNorm')

        print("✅ Deep models (with/without BatchNorm) created")
        return self.models

    def train_and_compare(self, X_train, y_train, X_val, y_val, epochs=20):
        """Train both models and compare convergence"""

        print("🚀 Comparing BatchNorm impact on deep network training...")

        for name, model in self.models.items():
            print(f"\n🔄 Training {name} model...")

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            start_time = time.time()
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=128,
                verbose=1 if name == 'without_bn' else 0
            )

            training_time = time.time() - start_time
            self.histories[name] = history

            # Analyze convergence
            final_val_acc = history.history['val_accuracy'][-1]
            epochs_to_90 = self.find_convergence_epoch(history.history['val_accuracy'], 0.9)

            print(f"✅ {name}: Final Val Acc={final_val_acc:.3f}, 90% at epoch {epochs_to_90}, Time={training_time:.1f}s")

    def find_convergence_epoch(self, accuracy_history, target=0.9):
        """Find epoch when target accuracy was first reached"""
        for i, acc in enumerate(accuracy_history):
            if acc >= target:
                return i + 1
        return "Not reached"

    def visualize_convergence(self):
        """Visualize the impact of batch normalization on convergence"""

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Training Convergence
        axes[0].set_title('Training Accuracy Convergence', fontsize=14, fontweight='bold')
        for name, history in self.histories.items():
            color = 'blue' if 'without' in name else 'green'
            axes[0].plot(history.history['accuracy'], label=f'{name}', linewidth=3, color=color)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Training Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90% target')

        # Plot 2: Validation Convergence
        axes[1].set_title('Validation Accuracy Convergence', fontsize=14, fontweight='bold')
        for name, history in self.histories.items():
            color = 'blue' if 'without' in name else 'green'
            axes[1].plot(history.history['val_accuracy'], label=f'{name}', linewidth=3, color=color)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90% target')

        # Plot 3: Loss Convergence (Log Scale)
        axes[2].set_title('Loss Convergence (Log Scale)', fontsize=14, fontweight='bold')
        for name, history in self.histories.items():
            color = 'blue' if 'without' in name else 'green'
            axes[2].plot(history.history['loss'], label=f'{name} (train)', linewidth=2, color=color)
            axes[2].plot(history.history['val_loss'], label=f'{name} (val)', linewidth=2, color=color, linestyle='--')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss (Log Scale)')
        axes[2].set_yscale('log')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle('⚡ Batch Normalization Impact Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.show()


class EarlyStoppingDemo:
    """
    Live Demonstration 3: Early Stopping in Action
    Comprehensive callback suite with monitoring
    """

    def __init__(self):
        self.model = None
        self.history = None
        print("🛑 Early Stopping Demo Initialized")

    def create_model_with_callbacks(self):
        """Create model with comprehensive callback suite"""

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ], name='EarlyStop_Model')

        # Comprehensive callback suite
        self.callbacks = [
            # Early stopping with patience
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),

            # Learning rate reduction on plateau
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1,
                mode='min'
            ),

            # Model checkpointing (optional - commented out for demo)
            # tf.keras.callbacks.ModelCheckpoint(
            #     'best_model.h5',
            #     monitor='val_accuracy',
            #     save_best_only=True,
            #     verbose=1
            # )
        ]

        print("✅ Model with callback suite created")
        return self.model, self.callbacks

    def train_with_monitoring(self, X_train, y_train, X_val, y_val):
        """Train with comprehensive monitoring and early stopping"""

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("🚀 Training with early stopping and monitoring...")
        print("📊 Watch for automatic learning rate reductions and early stopping...")

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,  # Large number, but early stopping will intervene
            batch_size=128,
            callbacks=self.callbacks,
            verbose=1
        )

        return self.history

    def analyze_stopping_behavior(self):
        """Analyze why and when training stopped"""

        if self.history is None:
            print("❌ No training history available!")
            return

        epochs_trained = len(self.history.history['loss'])
        best_epoch = np.argmin(self.history.history['val_loss']) + 1

        print(f"\n📈 Training Analysis:")
        print(f"🔢 Total epochs trained: {epochs_trained}")
        print(f"🎯 Best validation loss at epoch: {best_epoch}")
        print(f"💰 Saved {epochs_trained - best_epoch} epochs of unnecessary training!")

        # Learning rate changes
        if 'lr' in self.history.history:
            lr_changes = []
            for i, lr in enumerate(self.history.history['lr']):
                if i > 0 and lr != self.history.history['lr'][i-1]:
                    lr_changes.append((i+1, lr))

            if lr_changes:
                print(f"📉 Learning rate changes: {lr_changes}")

    def visualize_early_stopping(self):
        """Visualize early stopping behavior"""

        if self.history is None:
            print("❌ No training history to visualize!")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Find best epoch
        best_epoch = np.argmin(self.history.history['val_loss'])

        # Plot 1: Loss Evolution
        axes[0].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].axvline(x=best_epoch, color='red', linestyle='--', linewidth=2, label='Best Model')
        axes[0].set_title('Loss Evolution with Early Stopping')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Accuracy Evolution
        axes[1].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[1].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[1].axvline(x=best_epoch, color='red', linestyle='--', linewidth=2, label='Best Model')
        axes[1].set_title('Accuracy Evolution with Early Stopping')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Learning Rate Schedule (if available)
        if 'lr' in self.history.history:
            axes[2].plot(self.history.history['lr'], label='Learning Rate', linewidth=2, color='orange')
            axes[2].set_title('Learning Rate Schedule')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Learning Rate')
            axes[2].set_yscale('log')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'Learning Rate\nHistory Not Available',
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[2].transAxes, fontsize=12)
            axes[2].set_title('Learning Rate Schedule')

        plt.tight_layout()
        plt.suptitle('🛑 Early Stopping Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.show()


# Main demonstration controller
class LiveDemoController:
    """
    Main controller for all live demonstrations
    Use this for the actual lecture
    """

    def __init__(self):
        print("🎓 Week 6 Day 4: Advanced Regularization Live Demos")
        print("=" * 60)

    def run_all_demos(self):
        """Run all three demonstrations in sequence"""

        # Prepare data once for all demos
        print("📊 Preparing MNIST data...")
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
        X_test = X_test.reshape(-1, 784).astype('float32') / 255.0
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Demo 1: Dropout
        print("\n" + "="*60)
        print("🎯 DEMO 1: DROPOUT EFFECTIVENESS")
        print("="*60)
        dropout_demo = DropoutDemo()
        dropout_demo.create_models()
        dropout_demo.train_all_models(X_train, y_train, X_val, y_val, epochs=10)
        dropout_demo.visualize_results()

        # Demo 2: Batch Normalization
        print("\n" + "="*60)
        print("⚡ DEMO 2: BATCH NORMALIZATION IMPACT")
        print("="*60)
        bn_demo = BatchNormalizationDemo()
        bn_demo.create_deep_models()
        bn_demo.train_and_compare(X_train, y_train, X_val, y_val, epochs=15)
        bn_demo.visualize_convergence()

        # Demo 3: Early Stopping
        print("\n" + "="*60)
        print("🛑 DEMO 3: EARLY STOPPING IN ACTION")
        print("="*60)
        es_demo = EarlyStoppingDemo()
        es_demo.create_model_with_callbacks()
        es_demo.train_with_monitoring(X_train, y_train, X_val, y_val)
        es_demo.analyze_stopping_behavior()
        es_demo.visualize_early_stopping()

        print("\n" + "="*60)
        print("✅ ALL DEMONSTRATIONS COMPLETED!")
        print("🎯 Key Takeaways:")
        print("   • Dropout prevents overfitting (use 0.2-0.5)")
        print("   • BatchNorm accelerates training convergence")
        print("   • Early stopping saves time and prevents overfitting")
        print("   • Combine techniques for best results!")
        print("="*60)


# Quick individual demo functions for flexible use during lecture
def quick_dropout_demo():
    """Quick 5-minute dropout demonstration"""
    print("🎯 Quick Dropout Demo (5 minutes)")

    # Load and prepare data
    (X_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
    X_train, X_val, y_train, y_val = train_test_split(X_train[:10000], y_train[:10000], test_size=0.2, random_state=42)

    # Two quick models
    model_no_dropout = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model_with_dropout = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Quick training
    for name, model in [('No Dropout', model_no_dropout), ('With Dropout', model_with_dropout)]:
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, verbose=0)

        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        print(f"{name}: Train={train_acc:.3f}, Val={val_acc:.3f}, Gap={train_acc-val_acc:.3f}")


def quick_batchnorm_demo():
    """Quick 5-minute batch normalization demonstration"""
    print("⚡ Quick BatchNorm Demo (5 minutes)")

    # Small dataset for quick demo
    (X_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
    X_train, X_val, y_train, y_val = train_test_split(X_train[:5000], y_train[:5000], test_size=0.2, random_state=42)

    # Two models
    models = {
        'Without BatchNorm': tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ]),
        'With BatchNorm': tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
    }

    # Quick comparison
    for name, model in models.items():
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        start_time = time.time()
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, verbose=0)
        train_time = time.time() - start_time

        final_acc = history.history['val_accuracy'][-1]
        print(f"{name}: Val Acc={final_acc:.3f}, Time={train_time:.1f}s")


if __name__ == "__main__":
    # For live lecture use
    print("🎓 Advanced Regularization Demonstrations Ready!")
    print("\nAvailable functions:")
    print("• LiveDemoController().run_all_demos() - Full demonstration suite")
    print("• quick_dropout_demo() - 5-minute dropout demo")
    print("• quick_batchnorm_demo() - 5-minute batchnorm demo")
    print("\nChoose based on available time during lecture!")