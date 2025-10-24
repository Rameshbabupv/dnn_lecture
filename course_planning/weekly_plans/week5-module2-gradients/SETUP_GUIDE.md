# 🚀 Week 5 Environment Setup Guide

## Quick Setup Options

### Option 1: Using Micromamba (Recommended)

```bash
# Navigate to Week 5 directory
cd /Users/rameshbabu/data/projects/srm/lectures/Deep_Neural_Network_Architectures/course_planning/weekly_plans/week5-module2-gradients

# Run the automated setup script
./setup_environment.sh

# Or manually create environment
micromamba create -n week5-gradients -f environment.yml
micromamba activate week5-gradients
```

### Option 2: Using Conda/Mamba

```bash
# Using environment.yml
conda env create -f environment.yml
conda activate week5-gradients

# Or using mamba (faster)
mamba env create -f environment.yml
mamba activate week5-gradients
```

### Option 3: Using pip with Virtual Environment

```bash
# Create virtual environment
python3 -m venv week5-env
source week5-env/bin/activate  # On Windows: week5-env\Scripts\activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### Option 4: Quick Install (Current Environment)

```bash
# Just install the essentials
pip install tensorflow==2.15.0 numpy matplotlib scikit-learn seaborn jupyter
```

## 📦 Required Packages

### Essential (Must Have):
- `tensorflow` (2.15.0) - Deep learning framework
- `numpy` - Numerical computing
- `matplotlib` - Plotting
- `scikit-learn` - ML utilities

### Recommended (For Full Experience):
- `seaborn` - Better visualizations
- `jupyter` - Interactive notebooks
- `pandas` - Data manipulation
- `tqdm` - Progress bars

## 🧪 Testing Your Setup

### 1. Test Python Environment
```python
python -c "import sys; print(f'Python: {sys.version}')"
```

### 2. Test TensorFlow
```python
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
```

### 3. Test GPU (if available)
```python
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

### 4. Run the Demo Script
```bash
python gradient_problems_demo.py
```

## 🔧 Troubleshooting

### Issue: "No module named tensorflow"
```bash
pip install tensorflow==2.15.0
```

### Issue: "matplotlib backend issues"
```bash
# Set backend before importing
export MPLBACKEND=Agg  # For non-interactive
# or
export MPLBACKEND=TkAgg  # For interactive
```

### Issue: "micromamba not found"

**Install on macOS:**
```bash
brew install micromamba
```

**Install on Linux:**
```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar xvj bin/micromamba
export MAMBA_ROOT_PREFIX=~/micromamba
eval "$(./bin/micromamba shell hook -s posix)"
```

### Issue: "Permission denied on setup_environment.sh"
```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

## 📊 Expected Output

When you run `gradient_problems_demo.py`, you should see:
1. Package verification
2. Progress messages for each part
3. Four PNG files in `gradient_plots/` directory
4. Console output with gradient analysis

## 🎯 Minimal Quick Start

If you just want to run the demo quickly:

```bash
# Install only what's needed
pip install tensorflow numpy matplotlib scikit-learn

# Run the demo
python gradient_problems_demo.py
```

## 📚 For Google Colab

Upload these files to Colab:
1. `week5_gradient_problems_colab.ipynb`
2. `gradient_problems_demo.py`

Then run in Colab:
```python
!pip install -q tensorflow matplotlib numpy scikit-learn seaborn
```

## ✅ Verification Checklist

- [ ] Python 3.8+ installed
- [ ] TensorFlow 2.x installed
- [ ] NumPy installed
- [ ] Matplotlib installed
- [ ] Script runs without errors
- [ ] Plots are generated

## 💡 Tips

1. **For M1/M2 Macs**: Use `tensorflow-macos` instead of `tensorflow`
2. **For GPU support**: Install `tensorflow[and-cuda]` (TF 2.15+)
3. **For Jupyter**: Make sure to install ipykernel and create kernel
4. **For VSCode**: Select the correct Python interpreter

## 📧 Need Help?

If setup fails:
1. Check Python version: `python --version` (need 3.8+)
2. Try upgrading pip: `pip install --upgrade pip`
3. Install one package at a time to identify issues
4. Use `--no-cache-dir` flag if pip cache is corrupted

## 🚀 Ready to Go!

Once setup is complete:
1. Run `python gradient_problems_demo.py` for console version
2. Open `week5_gradient_problems_colab.ipynb` in Jupyter
3. Check `gradient_plots/` for generated visualizations