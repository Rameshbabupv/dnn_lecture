#!/bin/bash

# Week 5: Environment Setup Script
# Deep Neural Network Architectures - Gradient Problems & Regularization
# This script sets up a micromamba environment for Week 5 exercises

set -e  # Exit on error

echo "============================================"
echo "Week 5 Environment Setup"
echo "Deep Neural Network Architectures"
echo "============================================"
echo

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Environment name
ENV_NAME="week5-gradients"
PYTHON_VERSION="3.10"

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Check if micromamba is installed
if ! command -v micromamba &> /dev/null; then
    print_error "micromamba is not installed!"
    echo
    echo "Please install micromamba first:"
    echo "  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar xvj bin/micromamba"
    echo "  export MAMBA_ROOT_PREFIX=~/micromamba"
    echo "  eval \"\$(./bin/micromamba shell hook -s posix)\""
    echo
    echo "Or on macOS with Homebrew:"
    echo "  brew install micromamba"
    exit 1
fi

print_success "micromamba found"

# Check if environment already exists
if micromamba env list | grep -q "^${ENV_NAME}"; then
    print_info "Environment '${ENV_NAME}' already exists"
    read -p "Do you want to remove and recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing environment..."
        micromamba env remove -n ${ENV_NAME} -y
        print_success "Environment removed"
    else
        print_info "Activating existing environment..."
        echo "Run: micromamba activate ${ENV_NAME}"
        exit 0
    fi
fi

# Create new environment
print_info "Creating micromamba environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
micromamba create -n ${ENV_NAME} python=${PYTHON_VERSION} -c conda-forge -y
print_success "Environment created"

# Activate environment
print_info "Activating environment..."
eval "$(micromamba shell hook -s bash)"
micromamba activate ${ENV_NAME}
print_success "Environment activated"

# Upgrade pip
print_info "Upgrading pip..."
python -m pip install --upgrade pip
print_success "pip upgraded"

# Install packages from requirements.txt
if [ -f "requirements.txt" ]; then
    print_info "Installing packages from requirements.txt..."
    pip install -r requirements.txt
    print_success "All packages installed"
else
    print_error "requirements.txt not found in current directory"
    exit 1
fi

# Verify critical installations
print_info "Verifying installations..."
echo

python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import numpy as np; print(f'NumPy version: {np.__version__}')"
python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')"
python -c "import matplotlib; print(f'Matplotlib version: {matplotlib.__version__}')"

# Check GPU availability
echo
print_info "Checking GPU availability..."
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'✓ GPU found: {len(gpus)} device(s)')
    for gpu in gpus:
        print(f'  - {gpu}')
else:
    print('ℹ No GPU found (CPU mode)')
"

# Create Jupyter kernel
print_info "Creating Jupyter kernel..."
python -m ipykernel install --user --name=${ENV_NAME} --display-name="Week 5: Gradients (Python ${PYTHON_VERSION})"
print_success "Jupyter kernel created"

# Test the gradient demo script
echo
print_info "Testing gradient_problems_demo.py..."
if [ -f "gradient_problems_demo.py" ]; then
    python -c "
import sys
sys.path.append('.')
# Just test imports, don't run full script
exec(open('gradient_problems_demo.py').read().split('# Run vanishing gradient demonstration')[0])
print('✓ Script imports successful')
"
else
    print_info "gradient_problems_demo.py not found - skipping test"
fi

echo
echo "============================================"
print_success "Environment setup complete!"
echo "============================================"
echo
echo "To activate this environment, run:"
echo "  ${GREEN}micromamba activate ${ENV_NAME}${NC}"
echo
echo "To run the gradient demonstration:"
echo "  ${GREEN}python gradient_problems_demo.py${NC}"
echo
echo "To start Jupyter with this kernel:"
echo "  ${GREEN}jupyter notebook${NC}"
echo "  Then select kernel: 'Week 5: Gradients (Python ${PYTHON_VERSION})'"
echo
echo "To deactivate when done:"
echo "  ${GREEN}micromamba deactivate${NC}"
echo