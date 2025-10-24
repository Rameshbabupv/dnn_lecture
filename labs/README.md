# MNIST Digit Recognition Application

## Overview

This repository contains a comprehensive MNIST digit recognition application built for the **Deep Neural Network Architectures (21CSE558T)** course at SRM University. The application demonstrates practical implementation of neural networks for handwritten digit classification.

## 🚀 Features

- **Interactive Drawing Interface**: Draw digits with your mouse/touchpad
- **Real-time Prediction**: Instant digit recognition from 0-9
- **Confidence Scoring**: See prediction confidence and top-3 results
- **Neural Network**: Multi-layer perceptron with dropout regularization
- **Automatic Training**: Trains on MNIST dataset if no pre-trained model exists

## 📁 Project Structure

```
labs/
├── gradioUI.py              # Gradio-based web interface (requires fix)
├── simple_mnist_ui.py       # Tkinter-based desktop interface
├── mnist_mlp.keras          # Trained model (auto-generated)
├── srnenv/                  # Python virtual environment
└── README.md               # This file
```

## 🔧 Technical Specifications

### Neural Network Architecture
- **Input Layer**: 784 neurons (28×28 flattened image)
- **Hidden Layer 1**: 128 neurons with ReLU activation + Dropout (0.2)
- **Hidden Layer 2**: 64 neurons with ReLU activation + Dropout (0.2)
- **Output Layer**: 10 neurons with softmax activation (digits 0-9)

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Epochs**: 3 (for quick demo)
- **Batch Size**: 128
- **Validation Split**: 10%

### Dependencies
```
tensorflow>=2.20.0
numpy>=2.2.6
opencv-python>=4.12.0
matplotlib>=3.10.5
gradio>=4.44.0 (for web interface)
tkinter (for desktop interface)
```

## 🛠️ Installation & Setup

### 1. Environment Setup
```bash
# Activate virtual environment
source srnenv/bin/activate

# Verify installation
pip list
```

### 2. Check Dependencies
All required packages are pre-installed in the virtual environment:
- TensorFlow 2.20.0
- OpenCV 4.12.0
- NumPy 2.2.6
- Matplotlib 3.10.5

## 🏃‍♂️ Running the Application

### Option 1: Desktop Interface (Tkinter)
```bash
source srnenv/bin/activate
python simple_mnist_ui.py
```

**Note**: Currently experiencing tkinter import issues. Alternative solutions below.

### Option 2: Web Interface (Gradio) - Requires Fix
```bash
source srnenv/bin/activate
python gradioUI.py
```

**Known Issue**: Gradio has compatibility issues with Python 3.13 due to missing `audioop`/`pyaudioop` modules.

### Option 3: Command Line Interface (Recommended)
Create a simple command-line version:

```python
# Run in Python environment
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model("mnist_mlp.keras")

# Test with MNIST test data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype('float32') / 255.0
x_test = x_test.reshape(-1, 28*28)

# Make predictions
predictions = model.predict(x_test[:10])
print("Predictions:", np.argmax(predictions, axis=1))
print("Actual:", y_test[:10])
```

## 🎯 How to Use

### Drawing Interface Features:
1. **Canvas**: 280×280 pixel drawing area
2. **Draw**: Click and drag to draw digits
3. **Predict Button**: Get AI prediction for your drawing
4. **Clear Button**: Reset canvas for new drawing
5. **Results Display**: Shows predicted digit and confidence scores

### Prediction Process:
1. **Preprocessing**: 
   - Convert drawing to grayscale
   - Normalize pixel values (0-1)
   - Resize to 28×28 pixels
   - Find and crop digit bounding box

2. **Model Inference**:
   - Feed preprocessed image to neural network
   - Get probability distribution over 10 classes
   - Return top prediction with confidence

3. **Results**:
   - Primary prediction with confidence score
   - Top-3 predictions with probabilities

## 🔍 Model Performance

### Training Results (3 epochs):
- **Training Accuracy**: ~95-97%
- **Validation Accuracy**: ~94-96%
- **Training Time**: ~2-3 minutes
- **Model Size**: ~2MB

### Evaluation Metrics:
- **Test Accuracy**: Expected 94-96% on MNIST test set
- **Inference Time**: <100ms per prediction
- **Memory Usage**: ~50MB model footprint

## 🐛 Troubleshooting

### Common Issues:

1. **Tkinter Import Error**:
   ```
   ModuleNotFoundError: No module named '_tkinter'
   ```
   **Solution**: Install tkinter system package or use web interface

2. **Gradio Audio Dependencies**:
   ```
   ModuleNotFoundError: No module named 'pyaudioop'
   ```
   **Solution**: Python 3.13 compatibility issue - use alternative interface

3. **TensorFlow Warnings**:
   ```
   Protobuf gencode version warnings
   ```
   **Solution**: These are non-critical version warnings, app functions normally

4. **Model Not Found**:
   ```
   No existing model found, creating new one...
   ```
   **Solution**: Normal behavior - app will train new model automatically

## 📚 Educational Context

This application serves as a practical demonstration for the Deep Neural Network Architectures course:

### Learning Objectives Covered:
- **CO-1**: Simple deep neural networks implementation
- **CO-2**: Multi-layer networks with activation functions
- **CO-3**: Deep learning for image processing
- **Module 1-2**: Perceptron → MLP → TensorFlow basics

### Key Concepts Demonstrated:
- Neural network architecture design
- Image preprocessing and normalization
- Model training and evaluation
- Real-time inference pipeline
- User interface development

## 🔮 Future Enhancements

### Planned Improvements:
1. **Web Interface Fix**: Resolve Gradio compatibility issues
2. **Model Optimization**: Implement CNN architecture
3. **Data Augmentation**: Add rotation, scaling, noise
4. **Transfer Learning**: Pre-trained model integration
5. **Deployment**: Docker containerization

### Advanced Features:
- Batch prediction capability
- Model comparison interface
- Real-time training visualization
- Export predictions to CSV
- Mobile-responsive web interface

## 📖 Course Integration

### Assessment Alignment:
- **Unit Test 1**: Neural network fundamentals (Sep 19)
- **Unit Test 2**: Advanced architectures (Oct 31)
- **Practical Evaluation**: Continuous assessment
- **Final Exam**: Comprehensive integration

### Tutorial Tasks (T1-T15):
This application demonstrates concepts from:
- **T1-T3**: Basic neural networks
- **T4-T6**: Multi-layer perceptrons
- **T7-T9**: Image processing basics
- **T10-T12**: Classification applications

## 📄 License & Attribution

**Course**: Deep Neural Network Architectures (21CSE558T)  
**Institution**: SRM University  
**Target**: M.Tech Students  
**Duration**: 15 weeks (Aug 11 - Nov 21, 2025)

## 🤝 Contributing

For course-related improvements:
1. Fork the repository
2. Create feature branch
3. Implement changes
4. Test thoroughly
5. Submit pull request

## 📞 Support

For technical issues or course questions:
- **Course Instructor**:Ramesh Babu
- **Lab Sessions**: Hands-on troubleshooting
- **Documentation**: Refer to course materials

---

*This application is part of the academic coursework for understanding deep learning fundamentals through practical implementation.*
