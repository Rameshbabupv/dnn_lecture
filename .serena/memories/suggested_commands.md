# Suggested Commands for Development

## Virtual Environment Activation
```bash
# Activate primary lab environment
source labs/srnenv/bin/activate

# Activate MNIST exercise environment
source labs/exercise-1/mnist-env/bin/activate
```

## Running Lab Applications
```bash
# Desktop MNIST application (Tkinter-based)
cd labs && python simple_mnist_ui.py

# Web interface (Gradio-based - has compatibility issues with Python 3.13)
cd labs && python gradioUI.py

# Generate course handout Excel file
python create_course_handout_excel.py
```

## Installing Dependencies
```bash
# Install from requirements file
pip install -r course_planning/weekly_plans/week2-module1-mlp/day4/requirements.txt

# Common packages
pip install tensorflow==2.20.0
pip install opencv-python
pip install numpy matplotlib
pip install gradio  # Note: compatibility issues with Python 3.13
```

## Development Commands
```bash
# Run Jupyter notebook
jupyter notebook

# Launch Google Colab (recommended for students)
# Visit: https://colab.research.google.com

# Python script execution
python <script_name>.py
```

## File System Navigation (macOS/Darwin)
```bash
# List files
ls -la

# Navigate directories
cd <directory>

# Find files
find . -name "*.py"

# Search in files (use ripgrep if available)
rg "pattern" --type py
grep -r "pattern" .
```

## Git Commands
```bash
# Check status
git status

# Stage changes
git add .

# Commit changes
git commit -m "message"

# View history
git log --oneline
```

## Testing & Validation
```bash
# No specific test framework defined yet
# Check with user for testing commands when needed
# Model training typically includes validation within scripts
```

## Important Notes
- Always activate appropriate virtual environment before running code
- Tkinter requires system-level installation on macOS
- Gradio has compatibility issues with Python 3.13
- TensorFlow protobuf warnings are non-critical
- Models auto-train if .keras file not found