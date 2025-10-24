# Week 10 CNN Lecture - Jupyter Notebook Series

## 📚 Complete Series Overview

This directory contains **9 interactive Jupyter notebooks** that break down the 2-hour CNN lecture into digestible, hands-on learning modules.

---

## 📖 Notebook Series

### ✅ **Notebook 0: Setup & Prerequisites** (CREATED)
**File:** `00_setup_and_prerequisites.ipynb`
**Status:** ✅ Complete
**Duration:** ~10 minutes

**What you'll learn:**
- Environment setup and package installation
- NumPy array operations essential for CNNs
- Helper visualization functions
- Week 9 recap (manual features: LBP, GLCM)

**Key outputs:**
- Test 1D and 2D convolutions working
- Ready for the series!

---

### **Notebook 1: Convolution Concept & Intuition** (TO CREATE)
**File:** `01_convolution_concept_intuition.ipynb`
**Duration:** ~15 minutes

**What you'll learn:**
- Coffee filter analogy (pure intuition)
- Character: Dr. Priya's ECG pattern matching
- Sliding window visualization
- Pattern matching without math

**Key concepts:**
- Convolution = sliding pattern matcher
- Similarity detection across signals
- Foundation for understanding math

**Interactive elements:**
- Sliding window animation
- Pattern matching game
- "Where does pattern appear?" quiz

---

### **Notebook 2: 1D Convolution - Math & Code** (TO CREATE - HIGH PRIORITY)
**File:** `02_1d_convolution_math_code.ipynb`
**Duration:** ~20 minutes

**What you'll learn:**
- Complete 1D convolution calculation
- Element-wise multiply → sum
- Why outputs can exceed "perfect match"
- NumPy implementation

**Key examples:**
- ECG signal [1,1,2,1,3,2,1,2] with kernel [1,2,1]
- Manual calculation at 3 positions
- Complete output array
- Visualization of sliding process

**Code snippets:**
```python
signal = np.array([1, 1, 2, 1, 3, 2, 1, 2])
kernel = np.array([1, 2, 1])
output = np.convolve(signal, kernel, mode='valid')
```

**Exercises:**
1. Calculate convolution at position 0, 1, 2 manually
2. Design kernel to detect "falling edges" 
3. Interpret output values

---

### **Notebook 3: 2D Convolution - Images** (TO CREATE - HIGH PRIORITY)
**File:** `03_2d_convolution_images.ipynb`
**Duration:** ~25 minutes

**What you'll learn:**
- Extend 1D → 2D convolution
- Edge detection on real images
- Feature map interpretation
- Different kernel types

**Key examples:**
- Character: Arjun's photography edge detection
- 5×5 image with 3×3 kernel
- Vertical edge detector: `[[-1,0,1], [-1,0,1], [-1,0,1]]`
- Horizontal edge detector
- Apply to real photos

**Code snippets:**
```python
from scipy.signal import convolve2d

vertical_edge_kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

feature_map = convolve2d(image, vertical_edge_kernel, mode='valid')
```

**Exercises:**
1. Calculate 2D conv for top-left position
2. Design custom kernel for diagonal edges
3. Apply multiple kernels to same image

---

### **Notebook 4: Convolution Parameters** (TO CREATE)
**File:** `04_convolution_parameters.ipynb`
**Duration:** ~25 minutes

**What you'll learn:**
- **Kernel size:** 3×3 vs 5×5 vs 7×7
- **Stride:** 1 vs 2 (speed/detail trade-off)
- **Padding:** VALID vs SAME
- **Num filters:** Why 32→64→128

**Character stories:**
- Meera: Kernel size (quality control)
- Dr. Rajesh: Stride (CT scan speed)
- Detective Kavya: Multiple filters

**Interactive elements:**
- Stride slider (see output size change)
- Padding calculator
- Filter count visualizer

**The formula:**
```
Output Size = (Input - Kernel + 2×Padding) / Stride + 1
```

**Exercises:**
1. Calculate output dimensions for given params
2. Design layer to achieve specific output size
3. Count parameters in conv layer

---

### **Notebook 5: Hierarchical Feature Learning** (TO CREATE)
**File:** `05_hierarchical_feature_learning.ipynb`
**Duration:** ~20 minutes

**What you'll learn:**
- Why stack convolution layers?
- Hierarchical learning (edges→shapes→objects)
- Receptive field growth
- LeNet-5 architecture walkthrough

**Key visualizations:**
- Layer 1: Edges (vertical, horizontal, diagonal)
- Layer 2: Shapes (corners, curves)
- Layer 3: Parts (eyes, wheels)
- Layer 4: Objects (faces, cars)

**Receptive field demo:**
- Single 3×3: sees 3×3
- Two 3×3: sees 5×5
- Three 3×3: sees 7×7

**LeNet-5 trace:**
- 32×32 input through all layers
- Show feature maps at each stage
- Parameter counting

**Exercises:**
1. Calculate receptive field for N layers
2. Design 3-layer CNN for MNIST
3. Trace image dimensions through network

---

### **Notebook 6: Pooling Mechanisms** (TO CREATE)
**File:** `06_pooling_layers.ipynb`
**Duration:** ~20 minutes

**What you'll learn:**
- Why pooling? (3 purposes)
- Max pooling calculation
- Average pooling
- Global Average Pooling (GAP)
- Translation invariance

**Character: Detective Kavya's security cameras**
- Max pooling: Keep peak suspicious moments
- Average pooling: Overall activity level

**Max pooling example:**
```
Input 4×4:          Output 2×2:
[1, 3, 2, 4]        [6, 8]
[5, 6, 1, 2]   →    [4, 9]
[3, 2, 8, 7]
[1, 4, 3, 9]
```

**GAP comparison:**
```
Traditional: 7×7×512 → Flatten → 25,088 → Dense → 25M params
Modern (GAP): 7×7×512 → GAP → 512 → Dense → 512K params
```

**Exercises:**
1. Calculate max pool for 4×4 region
2. Compare max vs average outputs
3. Calculate parameter savings with GAP

---

### **Notebook 7: Complete CNN Architecture** (TO CREATE - HIGH PRIORITY)
**File:** `07_complete_cnn_architecture.ipynb`
**Duration:** ~25 minutes

**What you'll learn:**
- Standard CNN pipeline pattern
- Build complete CIFAR-10 classifier
- CNN vs MLP parameter comparison
- Train mini CNN on MNIST

**The pipeline:**
```
Input → [Conv→ReLU→Pool]×N → Flatten → FC → Softmax → Output
```

**CIFAR-10 example:**
```python
model = Sequential([
    Conv2D(32, 3, padding='same', activation='relu', input_shape=(32,32,3)),
    MaxPooling2D(2),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(2),
    Conv2D(128, 3, padding='same', activation='relu'),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

**Parameter counting:**
- Conv layers: ~93K parameters
- FC layers: ~1M parameters
- Total: ~1.3M (vs MLP: 1.7M with worse performance!)

**Quick MNIST demo:**
- Load Fashion-MNIST
- Train for 3 epochs
- Plot accuracy curves
- Show predictions

**Exercises:**
1. Design CNN for 28×28 images
2. Calculate parameters, verify with summary()
3. Modify architecture, observe changes

---

### **Notebook 8: 3D Convolution Preview** (TO CREATE)
**File:** `08_3d_convolution_preview.ipynb`
**Duration:** ~15 minutes

**What you'll learn:**
- 2D → 3D extension (add time dimension)
- 3D kernels for video/volumes
- Medical imaging (Dr. Rajesh's MRI)
- Video action recognition

**3D convolution:**
```
2D: Height × Width
3D: Height × Width × Time (or Depth)
```

**Applications:**
- Video understanding (action recognition)
- Medical imaging (CT/MRI volumes)
- Self-driving cars (temporal prediction)

**Code example:**
```python
from tensorflow.keras.layers import Conv3D

Conv3D(32, (3,3,3), input_shape=(16, 112, 112, 3))
# 16 frames, 112×112 RGB video
```

**Trade-offs:**
- 3D conv: 27× more computation than 2D
- Modern alternatives: (2+1)D factorization

**Exercises:**
1. Calculate 3D conv output size
2. List 3D conv applications
3. Compare 2D vs 3D parameter counts

---

### **Notebook 9: Review & Tutorial Preview** (TO CREATE)
**File:** `09_review_and_tutorial_preview.ipynb`
**Duration:** ~10 minutes

**What you'll learn:**
- Consolidate all 8 notebooks
- Self-assessment quiz
- Tutorial T10 preview (Monday)
- Homework assignment

**The 5 Big Ideas:**
1. Convolution = sliding pattern matcher
2. Hierarchical feature learning (automatic!)
3. 4 control parameters (kernel, stride, padding, filters)
4. Complete pipeline (Conv→Pool→...→FC→Softmax)
5. CNNs are efficient (weight sharing)

**Self-assessment:**
- 10 conceptual questions (auto-graded)
- 5 calculation problems
- Immediate feedback with explanations

**Tutorial T10 Preview (Oct 29, Monday):**
- Fashion-MNIST dataset
- Build CNN from scratch
- Expected: CNN 92% vs MLP 88%

**Homework (Due Week 11):**
- Task 1: Manual 2D convolution calculation
- Task 2: Design CNN for MNIST
- Task 3: Modify Tutorial T10 code

**Resources:**
- LeNet-5 paper (Yann LeCun, 1998)
- Stanford CS231n notes
- 3Blue1Brown convolution video

---

## 🎯 Learning Path

```
Start → Notebook 0 (Setup)
          ↓
      Notebook 1 (Intuition) ← Build foundation
          ↓
      Notebook 2 (1D Conv) ← Master mechanics
          ↓
      Notebook 3 (2D Conv) ← Apply to images
          ↓
      Notebook 4 (Parameters) ← Control behavior
          ↓
      Notebook 5 (Hierarchy) ← Understand depth
          ↓
      Notebook 6 (Pooling) ← Add efficiency
          ↓
      Notebook 7 (Complete CNN) ← Put together!
          ↓
      Notebook 8 (3D Conv) ← Preview advanced
          ↓
      Notebook 9 (Review) ← Consolidate
          ↓
   Tutorial T10 (Monday) → Build real CNN!
```

---

## 💻 How to Use These Notebooks

### Option 1: Local Jupyter
```bash
cd notebooks/
jupyter notebook
```

### Option 2: JupyterLab
```bash
cd notebooks/
jupyter lab
```

### Option 3: Google Colab
1. Upload notebook to Google Drive
2. Right-click → Open with → Google Colaboratory
3. Run all cells

### Option 4: VS Code
1. Install Python extension
2. Open .ipynb file
3. Select Python kernel
4. Run cells interactively

---

## 📊 Progress Tracker

As you complete each notebook, check it off:

- [x] **Notebook 0:** Setup & Prerequisites ✅
- [ ] **Notebook 1:** Convolution Concept
- [ ] **Notebook 2:** 1D Convolution
- [ ] **Notebook 3:** 2D Convolution  
- [ ] **Notebook 4:** Parameters
- [ ] **Notebook 5:** Hierarchical Learning
- [ ] **Notebook 6:** Pooling
- [ ] **Notebook 7:** Complete CNN
- [ ] **Notebook 8:** 3D Convolution
- [ ] **Notebook 9:** Review

**Current Progress:** 🔵⚪⚪⚪⚪⚪⚪⚪⚪⚪ (1/9)

---

## 🆘 Troubleshooting

### Can't import packages?
```bash
pip install numpy matplotlib scipy tensorflow
```

### Kernel dies?
- Reduce image sizes in examples
- Use smaller batch sizes
- Restart kernel and clear outputs

### Visualization not showing?
Add to first cell:
```python
%matplotlib inline
```

### Running in Colab?
All notebooks are Colab-compatible!

---

## 📚 Additional Resources

- **Lecture Notes:** `../do3-oct-27-Saturday/wip/v3_comprehensive_lecture_notes.md`
- **Week 10 Plan:** `../week10-plan.md`
- **Tutorial T10:** `../do4-oct-29-Monday/` (Coming Monday!)

---

## 👨‍🏫 Instructor Notes

### Delivery Options:
1. **Sequential:** One notebook per segment during lecture
2. **Parallel:** Students explore while you present
3. **Flipped:** Students complete before lecture
4. **Homework:** Assign as post-lecture practice

### Customization:
- Adjust difficulty in exercises
- Add/remove interactive elements
- Include/exclude TensorFlow training
- Translate to other languages

---

## ✅ Learning Outcomes

After completing all 9 notebooks, students will be able to:

✅ Explain convolution in plain language (no jargon)
✅ Calculate 1D and 2D convolutions manually
✅ Implement convolutions in NumPy/SciPy
✅ Understand and apply all 4 convolution parameters
✅ Calculate output dimensions for any configuration
✅ Design multi-layer CNN architectures
✅ Build complete CNNs in TensorFlow/Keras
✅ Visualize and interpret feature maps
✅ Compare CNN vs MLP efficiency
✅ Apply CNNs to real image classification tasks

---

## 🎓 Assessment Integration

### Unit Test 2 (Oct 31):
- Convolution calculations (Notebooks 2-3)
- Parameter effects (Notebook 4)
- Architecture design (Notebook 7)

### Tutorial T10 (Oct 29):
- Directly builds on Notebook 7
- Fashion-MNIST CNN implementation

### Homework:
- Exercises from each notebook
- Cumulative review problems

---

**Questions?** Ask during lecture or office hours!

**Happy Learning! 🚀**
