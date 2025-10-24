#!/usr/bin/env python3
"""
Week 6 Day 4: Advanced Regularization Practice Exercises
Course: 21CSE558T - Deep Neural Network Architectures

Practice problems for students to master advanced regularization techniques
before Unit Test 1. Complete these exercises to ensure you understand:
- Dropout implementation and effects
- Batch normalization mathematics and implementation
- Early stopping callback configuration
- Integration of multiple regularization techniques
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Set random seeds for reproducible results
np.random.seed(42)
tf.random.set_seed(42)

print("🎯 Advanced Regularization Practice Exercises")
print("=" * 60)
print("Complete all exercises to prepare for Unit Test 1!")
print("=" * 60)


# ==================== EXERCISE 1: DROPOUT ANALYSIS ====================

def exercise_1_dropout_analysis():
    """
    Exercise 1: Analyze the effect of different dropout rates

    Your Task:
    1. Create three models with dropout rates: 0.1, 0.3, 0.7
    2. Train all models on the same dataset
    3. Compare overfitting behavior
    4. Answer the analysis questions below
    """

    print("\n🎯 EXERCISE 1: DROPOUT RATE ANALYSIS")
    print("-" * 40)

    # Create synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                              n_redundant=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # TODO: Create three models with different dropout rates
    models = {}

    # Model 1: Low dropout (0.1)
    models['low_dropout'] = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
        tf.keras.layers.Dropout(0.1),  # TODO: Verify this is correct
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),  # TODO: Verify this is correct
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # TODO: Create Model 2 with 0.3 dropout rate
    models['medium_dropout'] = None  # Replace with your implementation

    # TODO: Create Model 3 with 0.7 dropout rate
    models['high_dropout'] = None  # Replace with your implementation

    # Training and evaluation code (complete this)
    histories = {}
    for name, model in models.items():
        if model is not None:
            print(f"Training {name} model...")
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # TODO: Train the model and store history
            history = None  # Replace with model.fit() call
            histories[name] = history

    # TODO: Create visualization comparing all three models
    # Plot training vs validation accuracy for each model
    # Your visualization code here

    print("\n📝 ANALYSIS QUESTIONS:")
    print("1. Which dropout rate performed best? Why?")
    print("2. What happened with 0.7 dropout rate? Explain the behavior.")
    print("3. How does dropout affect training speed?")
    print("4. When would you use each dropout rate?")

    return models, histories


# ==================== EXERCISE 2: BATCH NORMALIZATION DEEP DIVE ====================

def exercise_2_batch_normalization():
    """
    Exercise 2: Implement and analyze batch normalization effects

    Your Task:
    1. Create a deep network (5+ layers) without batch normalization
    2. Create the same network with batch normalization
    3. Compare training convergence and stability
    4. Implement batch normalization manually (bonus)
    """

    print("\n⚡ EXERCISE 2: BATCH NORMALIZATION DEEP DIVE")
    print("-" * 50)

    # Load MNIST for deep network testing
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

    # Use subset for faster training
    X_train, X_val, y_train, y_val = train_test_split(X_train[:5000], y_train[:5000],
                                                      test_size=0.2, random_state=42)

    # TODO: Create deep network WITHOUT batch normalization
    model_without_bn = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
        # TODO: Add 4 more dense layers (256, 128, 64, 32 units)
        # TODO: Add output layer (10 units, softmax activation)
    ])

    # TODO: Create deep network WITH batch normalization
    model_with_bn = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
        tf.keras.layers.BatchNormalization(),
        # TODO: Add 4 more dense layers with BatchNormalization after each
        # TODO: Add output layer (no BatchNorm after output)
    ])

    # TODO: Train both models and compare convergence
    print("Training models for comparison...")

    # Training code here
    history_without = None  # Train model_without_bn
    history_with = None     # Train model_with_bn

    # TODO: Create comparison plots
    # Compare: training loss, validation loss, training speed

    print("\n📝 ANALYSIS QUESTIONS:")
    print("1. Which model converged faster? By how many epochs?")
    print("2. Which model achieved better final accuracy?")
    print("3. How did batch normalization affect loss curve smoothness?")
    print("4. What happens if you remove batch normalization from the middle layers only?")

    # BONUS: Manual Batch Normalization Implementation
    def manual_batch_norm(x, gamma, beta, eps=1e-8):
        """
        TODO: Implement batch normalization manually

        Args:
            x: Input tensor (batch_size, features)
            gamma: Scale parameter (learnable)
            beta: Shift parameter (learnable)
            eps: Small constant for numerical stability

        Returns:
            Normalized output
        """
        # Your implementation here
        pass

    return model_without_bn, model_with_bn


# ==================== EXERCISE 3: EARLY STOPPING CONFIGURATION ====================

def exercise_3_early_stopping():
    """
    Exercise 3: Master early stopping and learning rate scheduling

    Your Task:
    1. Create a model that would normally overfit
    2. Configure early stopping with different patience values
    3. Add learning rate reduction on plateau
    4. Analyze the stopping behavior
    """

    print("\n🛑 EXERCISE 3: EARLY STOPPING MASTERY")
    print("-" * 40)

    # Create a small dataset that's easy to overfit
    X, y = make_classification(n_samples=500, n_features=50, n_informative=20,
                              n_redundant=30, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # Create a model that will overfit (too complex for the dataset)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(50,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # TODO: Configure early stopping callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,  # TODO: Experiment with different values (5, 10, 20)
        restore_best_weights=True,
        verbose=1
    )

    # TODO: Configure learning rate reduction
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # TODO: Experiment with different factors (0.2, 0.5, 0.8)
        patience=5,  # TODO: Experiment with different patience values
        min_lr=1e-7,
        verbose=1
    )

    # TODO: Train with callbacks and analyze results
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("Training with early stopping...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,  # Large number, early stopping will intervene
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # TODO: Analyze the stopping behavior
    epochs_trained = len(history.history['loss'])
    best_epoch = np.argmin(history.history['val_loss']) + 1

    print(f"\n📊 STOPPING ANALYSIS:")
    print(f"Total epochs trained: {epochs_trained}")
    print(f"Best model at epoch: {best_epoch}")
    print(f"Epochs saved: {epochs_trained - best_epoch}")

    # TODO: Plot training curves with stopping points
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.axvline(x=best_epoch-1, color='red', linestyle='--', label='Best Model')
    plt.title('Loss Evolution with Early Stopping')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.axvline(x=best_epoch-1, color='red', linestyle='--', label='Best Model')
    plt.title('Accuracy Evolution with Early Stopping')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n📝 ANALYSIS QUESTIONS:")
    print("1. How many epochs were saved by early stopping?")
    print("2. Did the learning rate get reduced? If so, when?")
    print("3. What would happen with patience=2 vs patience=20?")
    print("4. How do you choose optimal patience values?")

    return model, history


# ==================== EXERCISE 4: INTEGRATION CHALLENGE ====================

def exercise_4_integration_challenge():
    """
    Exercise 4: Combine all regularization techniques optimally

    Your Task:
    1. Create a comprehensive regularization strategy
    2. Use dropout, batch normalization, early stopping, and L2 regularization
    3. Tune hyperparameters for optimal performance
    4. Compare with baseline (no regularization)
    """

    print("\n🏆 EXERCISE 4: INTEGRATION CHALLENGE")
    print("-" * 45)

    # Create a challenging dataset
    X, y = make_classification(n_samples=2000, n_features=100, n_informative=50,
                              n_redundant=50, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # TODO: Create baseline model (no regularization)
    baseline_model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(100,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    # TODO: Create fully regularized model
    regularized_model = tf.keras.Sequential([
        # TODO: Add layers with L2 regularization, batch normalization, and dropout
        # Follow this pattern:
        # Dense layer with L2 regularization
        # BatchNormalization
        # Dropout
        # Repeat for multiple layers
        # Final output layer (no regularization)
    ])

    # TODO: Create comprehensive callback suite
    callbacks = [
        # TODO: Add early stopping
        # TODO: Add learning rate reduction
        # TODO: Add any other useful callbacks
    ]

    # TODO: Train both models and compare
    print("Training baseline model...")
    baseline_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # baseline_history = baseline_model.fit(...)

    print("Training regularized model...")
    regularized_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # regularized_history = regularized_model.fit(..., callbacks=callbacks)

    # TODO: Create comprehensive comparison
    # Compare: overfitting, final performance, training time, stability

    print("\n📝 FINAL CHALLENGE QUESTIONS:")
    print("1. Which model generalizes better? Provide evidence.")
    print("2. What's the trade-off between regularization and training time?")
    print("3. How would you adjust the regularization for a smaller dataset?")
    print("4. Design a regularization strategy for a 20-layer network.")

    return baseline_model, regularized_model


# ==================== UNIT TEST SIMULATION ====================

def unit_test_simulation():
    """
    Simulate Unit Test 1 conditions with practice problems
    """

    print("\n🎯 UNIT TEST 1 SIMULATION")
    print("=" * 40)
    print("Time limit: 50 minutes | Answer all questions")

    print("\n📋 MULTIPLE CHOICE QUESTIONS (15 minutes)")

    mcq_questions = [
        {
            "question": "What is the primary purpose of dropout in neural networks?",
            "options": ["A) Speed up training", "B) Prevent overfitting", "C) Improve accuracy", "D) Reduce memory usage"],
            "answer": "B"
        },
        {
            "question": "In batch normalization, what do γ and β parameters represent?",
            "options": ["A) Mean and variance", "B) Scale and shift", "C) Weights and biases", "D) Learning rates"],
            "answer": "B"
        },
        {
            "question": "Early stopping monitors which metric to prevent overfitting?",
            "options": ["A) Training loss", "B) Training accuracy", "C) Validation loss", "D) Test accuracy"],
            "answer": "C"
        }
    ]

    for i, q in enumerate(mcq_questions, 1):
        print(f"\nQ{i}. {q['question']}")
        for option in q['options']:
            print(f"   {option}")
        print(f"   Correct Answer: {q['answer']}")

    print("\n📝 SHORT ANSWER QUESTIONS (30 minutes)")

    saq_questions = [
        """
Q1. (5 marks) Explain why batch normalization accelerates training in deep neural networks.
Include the mathematical formulation and discuss the internal covariate shift problem.

Expected Answer Points:
- Internal covariate shift definition
- Mathematical formulation: (x-μ)/σ with γ and β
- Stabilization of layer inputs
- Allows higher learning rates
- Reduces dependency on initialization
""",
        """
Q2. (5 marks) Compare L1 and L2 regularization. When would you use each technique?
Provide mathematical formulations and explain their geometric interpretations.

Expected Answer Points:
- L1: λ∑|wi|, diamond constraint, sparsity
- L2: λ∑wi², circular constraint, weight decay
- Feature selection vs. weight smoothing
- Use cases for each technique
""",
        """
Q3. (5 marks) A model shows 95% training accuracy but 65% validation accuracy.
Diagnose the problem and suggest three specific solutions with TensorFlow implementation.

Expected Answer Points:
- Problem identification: overfitting
- Solution 1: Dropout with code
- Solution 2: L2 regularization with code
- Solution 3: Early stopping with code
- Expected improvement outcomes
"""
    ]

    for q in saq_questions:
        print(q)

    print("\n⏰ PRACTICE RECOMMENDATIONS:")
    print("1. Time yourself on these questions")
    print("2. Practice mathematical derivations")
    print("3. Memorize TensorFlow implementation patterns")
    print("4. Review common debugging scenarios")


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("🎓 Choose your practice exercise:")
    print("1. Dropout Analysis")
    print("2. Batch Normalization Deep Dive")
    print("3. Early Stopping Mastery")
    print("4. Integration Challenge")
    print("5. Unit Test Simulation")

    choice = input("\nEnter your choice (1-5): ")

    if choice == "1":
        exercise_1_dropout_analysis()
    elif choice == "2":
        exercise_2_batch_normalization()
    elif choice == "3":
        exercise_3_early_stopping()
    elif choice == "4":
        exercise_4_integration_challenge()
    elif choice == "5":
        unit_test_simulation()
    else:
        print("Invalid choice. Running all exercises...")
        exercise_1_dropout_analysis()
        exercise_2_batch_normalization()
        exercise_3_early_stopping()
        exercise_4_integration_challenge()
        unit_test_simulation()

    print("\n🎯 Exercise complete! Review your solutions and prepare for Unit Test 1!")
    print("📚 Don't forget to study the theoretical concepts alongside the code!")