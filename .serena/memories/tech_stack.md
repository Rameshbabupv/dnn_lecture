# Technology Stack

## Primary Technologies
- **Deep Learning Framework**: TensorFlow 2.x with Keras API
- **Computer Vision**: OpenCV 4.x
- **Programming Language**: Python (primary), MATLAB (supplementary)
- **Development Platform**: Google Colab (recommended for students)

## Python Libraries
- **TensorFlow**: 2.20.0 - Primary deep learning framework
- **OpenCV**: Computer vision and image processing
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **Gradio**: Web interfaces (has Python 3.13 compatibility issues)
- **Tkinter**: Desktop GUI applications
- **Pandas**: Data manipulation (for course handouts)
- **openpyxl**: Excel file generation

## Virtual Environments
- **labs/srnenv/**: Primary lab environment with TensorFlow, OpenCV, NumPy
- **labs/exercise-1/mnist-env/**: MNIST-specific environment
- **course_planning/.../tensorflow_env/**: Tutorial-specific environments

## Standard Model Architecture
MLP with 128→64→10 architecture:
- Input layer: 784 neurons (28x28 flattened)
- Hidden layer 1: 128 neurons with ReLU and Dropout(0.2)
- Hidden layer 2: 64 neurons with ReLU and Dropout(0.2)
- Output layer: 10 neurons with softmax

## Development Tools
- Git for version control
- Jupyter notebooks for interactive coding
- Google Colab for cloud-based development