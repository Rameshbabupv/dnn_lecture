# 🔢 Logic Gates with TensorFlow - Project Explanation

## 📋 **Project Overview**
This notebook demonstrates how to implement fundamental logic gates (AND, OR, XOR) using TensorFlow's neural network API. It's designed to show how neural networks can learn boolean logic operations.

## 🎯 **Project Plans & Structure**

### **1. Setup & Dependencies**
- **Libraries**: TensorFlow, NumPy, Matplotlib
- **Purpose**: Neural network training, numerical operations, and visualization

### **2. Data Preparation**
- **Input Data**: All possible 2-bit combinations `[[0,0], [0,1], [1,0], [1,1]]`
- **Output Targets**: 
  - **AND Gate**: `[0, 0, 0, 1]` (output 1 only when both inputs are 1)
  - **OR Gate**: `[0, 1, 1, 1]` (output 1 when at least one input is 1)
  - **XOR Gate**: `[0, 1, 1, 0]` (output 1 when inputs are different)

### **3. Model Architecture Strategy**

#### **For AND & OR Gates:**
- **Simple Single-Layer Perceptron**
- Architecture: `Dense(1, activation='sigmoid')`
- Optimizer: SGD (Stochastic Gradient Descent)
- Loss: Binary Crossentropy
- Training: 100 epochs

#### **For XOR Gate:**
- **Multi-Layer Perceptron (MLP)**
- Architecture: `Dense(4, sigmoid) → Dense(1, sigmoid)`
- Optimizer: Adam (more sophisticated than SGD)
- Loss: Binary Crossentropy
- Training: 1000 epochs (more complex problem)

### **4. Training Strategy**

#### **Phase 1: Linear Separable Gates (AND, OR)**
- Use simple single-layer networks
- These gates are linearly separable, so a single neuron can learn them
- Quick training with SGD optimizer

#### **Phase 2: Non-Linear Gate (XOR)**
- Requires hidden layer due to non-linearity
- XOR is not linearly separable, needs more complex architecture
- Longer training with Adam optimizer for better convergence

### **5. Evaluation Plan**
- **Prediction Display**: Show rounded predictions for all input combinations
- **Accuracy Metrics**: Track training accuracy during model fitting
- **Verification**: Compare predictions with expected truth tables

## 🔍 **Key Learning Objectives**

1. **Neural Network Fundamentals**: How neurons can learn boolean operations
2. **Linear vs Non-Linear Problems**: Why XOR needs hidden layers
3. **Architecture Selection**: Choosing appropriate network complexity
4. **Optimizer Comparison**: SGD vs Adam for different problem types
5. **Training Parameters**: Epochs, loss functions, and activation functions

## 🎯 **Expected Outcomes**

- **AND/OR Models**: Should achieve 100% accuracy with simple architecture
- **XOR Model**: Should learn the non-linear pattern with MLP architecture
- **Demonstration**: Proof that neural networks can learn logical operations

## 📊 **Truth Tables Reference**

| Input A | Input B | AND | OR | XOR |
|---------|---------|-----|----|-----|
| 0       | 0       | 0   | 0  | 0   |
| 0       | 1       | 0   | 1  | 1   |
| 1       | 0       | 0   | 1  | 1   |
| 1       | 1       | 1   | 1  | 0   |

## 🚀 **Implementation Details**

### **Code Structure:**
1. **Import Libraries**: TensorFlow, NumPy, Matplotlib
2. **Define Data**: Input combinations and target outputs
3. **Create Training Function**: Generic function for AND/OR gates
4. **Train Models**: Execute training for each gate type
5. **Special XOR Handling**: Separate function with MLP architecture

### **Key Functions:**
- `train_logic_gate()`: For AND/OR gates with single-layer perceptron
- `train_xor_gate()`: For XOR gate with multi-layer perceptron

This project serves as an excellent introduction to neural networks by starting with simple, interpretable problems that clearly demonstrate the difference between linearly separable and non-linearly separable problems.
