# Week 10 Day 4 (DO4) - Tutorial T10 Materials

**Deep Neural Network Architectures (21CSE558T)**
**Tutorial T10: Building CNN for Fashion-MNIST Classification**
**Session Date: October 29, 2025 (Monday), 4:00-4:50 PM IST**

---

## 📋 Overview

This directory contains **complete materials** for Tutorial T10, the first hands-on CNN implementation session. Students will build, train, and analyze a convolutional neural network for Fashion-MNIST classification.

**Status:** ✅ **Production Ready - All Materials Complete**

---

## 📂 Directory Structure

```
do4-oct-29-Monday/
├── README.md                                    ← You are here
├── tutorial_t10_starter.py                      ← Student template (TODO markers)
├── tutorial_t10_solution.py                     ← Complete working solution
├── tutorial_t10_fashion_mnist_cnn.ipynb        ← Interactive Jupyter notebook
├── quick_reference_cheat_sheet.md              ← 2-page handout for students
├── architecture_design_worksheet.md            ← Pre-coding design exercise
├── troubleshooting_guide.md                    ← Common errors & solutions
├── week10_homework_assignment.md               ← 3 tasks, due Nov 3
├── week10_practice_questions.md                ← Unit Test 2 preparation
├── create_jupyter_notebook.py                  ← Script to regenerate notebook
└── output/                                      ← Directory for tutorial results
```

---

## 🎯 Learning Objectives

By the end of Tutorial T10, students will be able to:

1. ✅ Load and preprocess Fashion-MNIST dataset
2. ✅ Build a CNN using Keras Sequential API
3. ✅ Train CNN and visualize training history
4. ✅ Evaluate model performance on test data
5. ✅ Visualize learned filters and feature maps
6. ✅ Compare CNN performance with MLP baseline
7. ✅ Debug common CNN training errors

---

## 📚 Material Descriptions

### 1. **tutorial_t10_starter.py** (Student Template)
- **Purpose:** Template for 50-minute tutorial session
- **Format:** Python script with TODO markers
- **Features:**
  - Clear TODO comments for each step
  - Hints and guidance in comments
  - Example code commented out
  - Structured in 12 parts matching tutorial flow
- **Usage:** Students uncomment and complete TODOs during session
- **Lines:** ~600 lines (heavily commented)

---

### 2. **tutorial_t10_solution.py** (Complete Solution)
- **Purpose:** Instructor reference and student post-tutorial review
- **Format:** Fully implemented Python script
- **Features:**
  - Complete working code for all tutorial parts
  - Detailed comments explaining every step
  - Best practices demonstrated
  - Performance metrics and analysis
  - Comparison with MLP baseline
  - Model saving functionality
- **Expected Results:**
  - Training accuracy: ~95%
  - Validation accuracy: ~90-92%
  - Test accuracy: ~90%
  - Training time: 2-3 minutes (CPU), 30 seconds (GPU)
- **Lines:** ~650 lines
- **Outputs Generated:**
  - output/sample_images.png
  - output/training_history.png
  - output/predictions.png
  - output/learned_filters.png
  - output/feature_maps.png
  - output/fashion_mnist_cnn_model.keras
  - output/fashion_mnist_cnn_model.h5

---

### 3. **tutorial_t10_fashion_mnist_cnn.ipynb** (Interactive Notebook)
- **Purpose:** Interactive alternative to Python scripts
- **Format:** Jupyter notebook with 29 cells
- **Features:**
  - 15 markdown cells (explanations, instructions)
  - 14 code cells (progressive implementation)
  - Self-contained (no external files needed)
  - Can run in Google Colab
  - Includes extension exercises
- **Structure:**
  - Introduction & objectives
  - Part 1: Data loading (2 cells)
  - Part 2: Preprocessing (2 cells)
  - Part 3: Build CNN (3 cells)
  - Part 4: Compile (2 cells)
  - Part 5: Train (2 cells)
  - Part 6: Training history (2 cells)
  - Part 7: Evaluate (2 cells)
  - Part 8: Visualize filters (2 cells)
  - Part 9: Visualize feature maps (2 cells)
  - Summary & extensions (4 cells)
- **Usage:**
  - Run cells sequentially
  - Pause for student exercises
  - Use in flipped classroom or self-paced learning

---

### 4. **quick_reference_cheat_sheet.md** (Student Handout)
- **Purpose:** 2-page reference for tutorial session
- **Format:** Markdown (print or digital)
- **Sections:**
  - Dataset loading (1-liner)
  - Data preprocessing (normalize, reshape, one-hot)
  - CNN architecture code template
  - Compile & train code
  - Key formulas (output dimensions, parameters)
  - Visualization code snippets
  - Common errors & quick fixes
  - Decision tables (when to use what)
  - Complete workflow (copy-paste ready)
- **Length:** ~400 lines (fits on 2 pages when printed)
- **Usage:** Print and distribute, or share PDF

---

### 5. **architecture_design_worksheet.md** (Pre-Coding Exercise)
- **Purpose:** Plan architecture before coding
- **Format:** Fill-in-the-blank worksheet
- **Sections:**
  - Part 1: Architecture table (specify all layers)
  - Part 2: Output dimension calculations (show work)
  - Part 3: Parameter counting (detailed calculations)
  - Part 4: Design justification (5 questions)
  - Part 5: CNN vs MLP comparison
  - Part 6: Expected performance estimates
  - Part 7: Alternative design (challenge)
  - Part 8: Verification checklist
  - Part 9: Translate to code
  - Part 10: Reflection
- **Answer key included:** For self-check after completion
- **Length:** ~600 lines
- **Usage:**
  - Complete before Tutorial T10 (5-10 minutes)
  - Helps students understand architecture before coding
  - Optional instructor review before coding session

---

### 6. **troubleshooting_guide.md** (Error Reference)
- **Purpose:** Comprehensive guide to common CNN errors
- **Format:** Markdown with problem-solution pairs
- **Sections:**
  1. Installation & Environment (3 errors)
  2. Data Loading (2 errors)
  3. Model Architecture (4 errors)
  4. Training Problems (3 errors)
  5. Memory Errors (1 error)
  6. Visualization Errors (2 errors)
  7. Google Colab Issues (2 errors)
  8. Quick debugging checklist
- **Total Issues Covered:** 17 common errors
- **Format per error:**
  - Error message (exact text)
  - Cause explanation
  - Multiple solution options
  - Code examples
- **Length:** ~500 lines
- **Usage:**
  - Reference during tutorial when students hit errors
  - Share link for independent debugging
  - Review before tutorial to anticipate issues

---

### 7. **week10_homework_assignment.md** (Take-Home Work)
- **Purpose:** Reinforce concepts through practice
- **Format:** Structured assignment document
- **Due Date:** November 3, 2025 (before Week 11)
- **Total Points:** 100 (+5 bonus)
- **Tasks:**
  - **Task 1 (30 pts):** Manual convolution calculation
    - Calculate 2D convolution for 6×6 image, 3×3 kernel
    - Show all steps, interpret results
  - **Task 2 (40 pts):** CNN architecture design for MNIST
    - Complete architecture specification
    - Output dimension calculations
    - Design justifications
    - Training configuration
  - **Task 3 (30 pts):** Code experimentation
    - Experiment 1: Add third conv block
    - Experiment 2: Compare kernel sizes (3×3, 5×5, 7×7)
    - Experiment 3: Vary filter counts (16→32, 32→64, 64→128)
    - Document observations (2-3 paragraphs)
  - **Bonus (+5 pts):** Insightful analysis
- **Deliverables:**
  - task1_calculations.pdf
  - task2_architecture_design.pdf
  - task3_code.py
  - task3_observations.pdf
- **Length:** ~650 lines
- **Submission:** ZIP file via course management system

---

### 8. **week10_practice_questions.md** (Unit Test 2 Prep)
- **Purpose:** Prepare students for Unit Test 2 (Oct 31)
- **Format:** Question bank with answers
- **Content:**
  - **Part A: MCQ (12 questions, 1 mark each)**
    - Output dimension calculations
    - Parameter counting
    - Conceptual understanding
    - Layer purposes
  - **Part B: Short Answer (5 questions, 5 marks each)**
    - SAQ 1: Multi-layer output calculation
    - SAQ 2: Complete parameter counting
    - SAQ 3: CNN vs MLP comparison
    - SAQ 4: Pooling layer purpose
    - SAQ 5: Architecture design
  - **Part C: Long Answer (2 questions, 10 marks each)**
    - LAQ 1: Complete convolution calculation + NumPy
    - LAQ 2: CNN implementation in Keras + analysis
- **Features:**
  - Full solutions provided
  - Marking schemes included
  - Study tips section
  - Formula reference
- **Length:** ~800 lines
- **Usage:**
  - Self-study for Unit Test 2
  - Review session material
  - Practice test simulation

---

## 🚀 Usage Instructions

### For Instructors:

#### Before Tutorial Session:
1. **Test code:** Run `tutorial_t10_solution.py` in clean environment
   ```bash
   python tutorial_t10_solution.py
   ```
2. **Verify environment:** Check TensorFlow, NumPy, Matplotlib installed
3. **Print handout:** Print `quick_reference_cheat_sheet.md` (2 pages/student)
4. **Share worksheet:** Distribute `architecture_design_worksheet.md` (optional)
5. **Review errors:** Familiarize with `troubleshooting_guide.md`
6. **Test notebook:** Run `tutorial_t10_fashion_mnist_cnn.ipynb` cell-by-cell

#### During Tutorial Session (50 minutes):
- **0-5 min:** Introduction, setup verification, distribute handouts
- **5-15 min:** Part 1-3 (load data, preprocess, explore)
- **15-30 min:** Part 4 (build CNN architecture layer-by-layer)
- **30-40 min:** Part 5-6 (compile, train, plot history)
- **40-50 min:** Part 7-9 (evaluate, visualize filters/feature maps)
- **50 min:** Wrap-up, homework assignment, Q&A

**Teaching Tips:**
- Use `tutorial_t10_solution.py` as reference
- Live code using `tutorial_t10_starter.py` with students
- Pause for checkpoints every 10 minutes
- Have `troubleshooting_guide.md` open for quick debugging
- Show visualizations from solution if running short on time

#### After Tutorial Session:
1. **Share solution:** Upload `tutorial_t10_solution.py` to course portal
2. **Share notebook:** Upload `.ipynb` for students to review
3. **Assign homework:** Release `week10_homework_assignment.md`
4. **Share practice questions:** Release `week10_practice_questions.md` for Unit Test 2 prep
5. **Collect feedback:** Ask students what worked/didn't work

---

### For Students:

#### Before Tutorial:
1. **Review Week 10 DO3 materials:** Jupyter notebooks 00-09
2. **Install TensorFlow:**
   ```bash
   pip install tensorflow numpy matplotlib
   ```
3. **Test installation:**
   ```python
   import tensorflow as tf
   print(tf.__version__)
   ```
4. **Complete worksheet (optional):** `architecture_design_worksheet.md`
5. **Print cheat sheet:** `quick_reference_cheat_sheet.md`

#### During Tutorial:
1. Open `tutorial_t10_starter.py`
2. Follow instructor, uncomment TODOs as you go
3. Run code frequently to verify
4. Use cheat sheet for syntax reference
5. Ask questions when stuck

#### After Tutorial:
1. **Review solution:** Compare your code with `tutorial_t10_solution.py`
2. **Run notebook:** Execute `tutorial_t10_fashion_mnist_cnn.ipynb` cell-by-cell
3. **Complete homework:** Work on 3 tasks in `week10_homework_assignment.md`
4. **Study for test:** Practice questions in `week10_practice_questions.md`
5. **Experiment:** Try extension exercises in notebook

---

## 🔧 Technical Requirements

### Software:
- Python 3.8+
- TensorFlow 2.15+ (with Keras)
- NumPy 1.24+
- Matplotlib 3.7+

### Hardware:
- **Minimum:** CPU with 4GB RAM (2-3 min training time)
- **Recommended:** GPU with 8GB VRAM (30 sec training time)
- **Alternative:** Google Colab (free GPU access)

### Installation:
```bash
# Create virtual environment (optional)
python -m venv cnn_tutorial
source cnn_tutorial/bin/activate  # On Windows: cnn_tutorial\Scripts\activate

# Install dependencies
pip install tensorflow numpy matplotlib jupyter

# Verify installation
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

---

## 📊 Expected Outcomes

### Model Performance:
- **Training Accuracy:** 95-97% (after 10 epochs)
- **Validation Accuracy:** 90-92%
- **Test Accuracy:** 90-92%
- **Training Time:** 2-3 minutes (CPU), 30 seconds (GPU)
- **Model Size:** ~122,000 parameters
- **Memory Usage:** ~500 MB

### Student Learning:
After completing Tutorial T10, students should be able to:
- ✅ Build CNN from scratch in Keras
- ✅ Calculate output dimensions and parameters manually
- ✅ Train CNN and interpret training curves
- ✅ Visualize and interpret learned filters
- ✅ Debug common CNN errors independently
- ✅ Compare CNN vs MLP architectures

---

## 📁 Output Files Generated

When running `tutorial_t10_solution.py`, the following files are created in `output/`:

| File | Size | Description |
|------|------|-------------|
| sample_images.png | ~50KB | 10 Fashion-MNIST samples with labels |
| training_history.png | ~100KB | Accuracy and loss curves over 10 epochs |
| predictions.png | ~150KB | Model predictions on 10 test images |
| learned_filters.png | ~200KB | 16 learned 3×3 filters from first conv layer |
| feature_maps.png | ~300KB | Feature maps at 4 layers for sample image |
| fashion_mnist_cnn_model.keras | ~500KB | Saved model (TensorFlow format) |
| fashion_mnist_cnn_model.h5 | ~500KB | Saved model (HDF5 format) |

**Total:** ~1.8 MB

---

## 🎓 Assessment Integration

### Tutorial T10 Contribution:
- **FT-IV (Continuous Assessment):** 15% of total grade
- **Evaluation Criteria:**
  - Code completeness (40%)
  - Model performance (30%)
  - Visualizations (15%)
  - Code quality/comments (15%)

### Homework Weight:
- **Week 10 Homework:** 5% of total grade
- Due: November 3, 2025
- Late penalty: -10%/day

### Unit Test 2 Coverage:
- **Date:** October 31, 2025 (2 days after tutorial!)
- **Week 10 Content Weight:** ~30% of Unit Test 2
- **Expected Question Types:**
  - MCQ: Convolution output dimensions, parameter counting
  - 5-mark: CNN architecture design, layer explanations
  - 10-mark: Complete convolution calculation, Keras implementation

---

## 🔗 Related Materials

### Week 10 Day 3 (DO3) Materials:
- Location: `do3-oct-27-Saturday/notebooks/`
- 10 Jupyter notebooks (00-09) covering CNN theory
- README.md with notebook guide

### Week 9 Connection:
- Week 9 DO4: CNN Introduction (WHY CNNs exist)
- Manual features (LBP, GLCM) → Learned features (CNNs)
- Motivation for automatic feature learning

### Week 11 Preview:
- Famous CNN architectures (LeNet, AlexNet, VGG, ResNet)
- Advanced techniques (Dropout, Batch Normalization, Data Augmentation)
- Transfer learning fundamentals

---

## ❓ Common Questions

### Q: Can I use MNIST instead of Fashion-MNIST?
**A:** Yes, but Fashion-MNIST is preferred because:
- More challenging (~90% vs ~98% accuracy)
- More realistic (actual image classification)
- Same format/API as MNIST
- Better prepares for real-world datasets

### Q: Why is my training slow?
**A:**
- **CPU training:** Normal, 2-3 minutes expected
- **Speed up:** Use Google Colab GPU (Runtime → Change runtime type → GPU)
- **Alternative:** Reduce batch size or train on subset

### Q: Model accuracy stuck at 10%?
**A:** Common causes (see troubleshooting_guide.md):
1. Forgot to normalize data (divide by 255)
2. Wrong loss function for label format
3. Learning rate too high
Check troubleshooting guide Section 4, Error 10.

### Q: Shape mismatch error?
**A:** Most likely forgot to reshape:
```python
X_train = X_train.reshape(-1, 28, 28, 1)  # Add channel dimension
```
See troubleshooting_guide.md Section 3, Error 6.

### Q: Can I modify the architecture?
**A:** Yes! That's the point of Homework Task 3. Experiment with:
- Different numbers of filters
- Different kernel sizes
- Additional conv blocks
- Dropout layers
- Batch normalization

### Q: How do I run the notebook in Google Colab?
**A:**
1. Upload `.ipynb` to Google Drive
2. Right-click → Open with → Google Colaboratory
3. Runtime → Change runtime type → GPU (optional)
4. Run cells sequentially

---

## 📞 Support

### During Tutorial:
- Raise hand for instructor assistance
- Consult `quick_reference_cheat_sheet.md`
- Check `troubleshooting_guide.md`
- Ask classmates (pair programming encouraged)

### After Tutorial:
- **Office Hours:** [Insert times]
- **Email:** [Insert instructor email]
- **Course Forum:** [Insert link]
- **Study Groups:** Form groups for homework

### When Asking for Help:
Include:
1. Full error message (copy-paste)
2. Code snippet causing error
3. What you've already tried
4. Environment (local/Colab, TensorFlow version)

---

## 🎯 Success Metrics

### Individual Student Success:
- ✅ Completed all tutorial parts during session
- ✅ Model trains without errors
- ✅ Achieves >85% test accuracy
- ✅ Generates all visualization outputs
- ✅ Understands architecture design choices

### Class Success Indicators:
- 80%+ students achieve >88% accuracy
- <5% students stuck on environment issues
- 90%+ students complete during 50-minute session
- Positive feedback on tutorial structure

---

## 📝 Changelog

### Version 1.0 (October 29, 2025)
- ✅ Initial release of all materials
- ✅ Starter code with TODO markers
- ✅ Complete solution code
- ✅ Interactive Jupyter notebook (29 cells)
- ✅ Quick reference cheat sheet
- ✅ Architecture design worksheet
- ✅ Comprehensive troubleshooting guide
- ✅ Homework assignment (3 tasks)
- ✅ Practice questions (MCQ + SAQ + LAQ)
- ✅ README documentation

### Future Enhancements (Week 11+):
- Video walkthrough of solution
- Additional architecture templates
- Data augmentation examples
- Advanced visualization techniques

---

## 📄 License & Attribution

**Course:** Deep Neural Network Architectures (21CSE558T)
**Institution:** SRM University
**Instructor:** [Insert instructor name]
**Academic Year:** 2025-2026
**Semester:** I (August-December 2025)

**Dataset:**
- Fashion-MNIST: Zalando Research (CC0: Public Domain)
- Citation: Xiao, H., Rasul, K., & Vollgraf, R. (2017)

**Framework:**
- TensorFlow/Keras: Apache License 2.0

---

## ✅ Final Checklist for Instructor

Before Tutorial T10 session:

- [ ] All files present in do4-oct-29-Monday/
- [ ] Solution code tested and runs successfully
- [ ] Jupyter notebook verified (all cells execute)
- [ ] Cheat sheets printed for all students
- [ ] Troubleshooting guide reviewed
- [ ] Homework assignment uploaded to portal
- [ ] Practice questions shared for Unit Test 2 prep
- [ ] Output directory created
- [ ] Projector/screen tested
- [ ] Google Colab backup plan ready

---

**Materials Status: ✅ Production Ready**
**Last Updated: October 29, 2025**
**Next Review: After tutorial delivery**

---

**Happy Teaching! Happy Learning! 🚀🎓**
