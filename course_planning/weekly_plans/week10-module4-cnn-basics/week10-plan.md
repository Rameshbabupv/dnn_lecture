# Week 10 Plan - Module 4: CNN Basics

## Week Overview

**Week Number:** 10/15
**Module:** 4 (Convolutional Neural Networks) - Week 1 of 3
**Dates:** October 27-29, 2025 (Rescheduled due to Diwali holidays)
**Assessment Context:** Preparing for Unit Test 2 (Oct 31) covering Modules 3-4

---

## Course Progress Context

### ✅ Completed (Weeks 1-9)
- **Module 1 (Weeks 1-3):** Neural networks fundamentals, TensorFlow, activation functions
- **Module 2 (Weeks 4-6):** Optimization, regularization, normalization
- **Module 3 (Weeks 7-9):** Image processing, segmentation, feature extraction
- **Week 9 Bridge (Oct 17):** CNN Introduction - WHY CNNs exist (conceptual)

### 🎯 Current Position (Week 10)
- **Module 4 begins:** Technical deep-dive into HOW CNNs work
- **Focus:** Mathematical foundations and practical implementation
- **Tutorial T10:** First hands-on CNN implementation in Keras

### ⏳ Coming Next (Weeks 11-12)
- **Week 11:** CNN layers, pooling, regularization
- **Week 12:** Transfer learning, pre-trained architectures
- **Oct 31:** Unit Test 2 (Modules 3-4)

---

## Day Order Schedule

**NOTE:** Schedule adjusted due to Diwali holidays (October 20-25, 2025)

### **Day Order 3 (do3) - October 27, 2025 (Saturday)**
- **Duration:** 2 hours (8:00 AM - 9:40 AM IST)
- **Format:** Lecture session
- **Focus:** CNN mathematical foundations and architecture components
- **Type:** Theory + demonstrations

### **Day Order 4 (do4) - October 29, 2025 (Monday)**
- **Duration:** 1 hour (4:00 PM - 4:50 PM IST)
- **Format:** Tutorial/Practical session
- **Focus:** **Tutorial T10** - Building CNN in Keras
- **Type:** Hands-on coding

---

## Week 10 Learning Objectives

By the end of Week 10, students should be able to:

1. **Explain** the biological motivation for CNNs (visual cortex hierarchy)
2. **Calculate** convolution operations manually (1D and 2D)
3. **Understand** convolution parameters (stride, padding, kernel size)
4. **Design** basic CNN architectures with appropriate layers
5. **Implement** CNN classification models using Keras
6. **Compare** CNN vs traditional MLP performance
7. **Visualize** learned filters and feature maps
8. **Calculate** output dimensions and parameter counts

---

## DO3 (Oct 22) - 2-Hour Lecture Plan

### **Theme:** "HOW do CNNs Work? - Technical Deep Dive"

### **Hour 1 (60 minutes): Convolution Mathematics**

#### Segment 1.1: Recap and Bridge (10 minutes)
- Quick recap of Oct 17 lecture (WHY CNNs?)
- Manual features → Learned features (Detective Maria callback)
- Today's focus: Mathematical mechanics

#### Segment 1.2: 1D Convolution Foundation (15 minutes)
- Signal processing origins
- 1D convolution step-by-step calculation
- Example: [1, 2, 3, 4, 5] * [1, 0, -1]
- Understanding "sliding window"
- Code snippet: NumPy 1D convolution

#### Segment 1.3: 2D Convolution for Images (20 minutes)
- Extending to 2D (images)
- Matrix convolution calculation
- 3×3 kernel on 5×5 image example
- Edge detection kernel demonstration
- Feature map concept
- Live calculation on board

#### Segment 1.4: Convolution Parameters (15 minutes)
- **Stride:** How much to slide (1, 2, etc.)
- **Padding:** Valid vs Same (maintaining dimensions)
- **Kernel size:** 3×3, 5×5, 7×7 trade-offs
- **Number of filters:** Multiple feature maps
- Output dimension formula: `(W - F + 2P) / S + 1`
- Parameter calculation examples

---

### **Hour 2 (60 minutes): CNN Architecture Components**

#### Segment 2.1: From Single Conv to Complete CNN (15 minutes)
- Single conv layer limitations
- Stacking convolution layers (hierarchy)
- Feature learning progression (edges → shapes → objects)
- Receptive field concept
- LeNet-5 walkthrough (classic example)

#### Segment 2.2: Pooling Mechanisms (15 minutes)
- **Why pooling?** Translation invariance, dimension reduction
- **Max pooling:** Keep strongest activations
- **Average pooling:** Smooth features
- **Global pooling:** Spatial dimension collapse
- Pooling calculation examples (2×2, stride 2)
- Pooling vs stride trade-offs

#### Segment 2.3: Complete CNN Pipeline (15 minutes)
- Input → [Conv → ReLU → Pool] × N → Flatten → FC → Softmax
- Where does classification happen? (FC layers)
- Role of each component
- Example: CIFAR-10 classifier architecture
- Parameter counting (conv vs FC layers)
- Why CNNs have fewer parameters than MLP

#### Segment 2.4: 3D Convolution Preview (10 minutes)
- 2D conv with multiple channels (RGB images)
- 3D convolution concept (video, medical imaging)
- Spatial vs spatiotemporal features
- Real-world applications

#### Segment 2.5: Wrap-up and Bridge (5 minutes)
- Key takeaways (4-5 bullet points)
- Connection to Tutorial T10 tomorrow
- Preview of Week 11 (famous architectures)
- Homework assignment

---

## DO4 (Oct 23) - 1-Hour Tutorial Plan

### **Tutorial T10: Building Programs to Perform Classification Using CNN In Keras**

### **Session Structure (50 minutes)**

#### Part 1: Setup and Data Loading (10 minutes)
- Load Fashion-MNIST or CIFAR-10 dataset
- Data exploration (shapes, classes)
- Preprocessing (normalization, reshaping)
- Train/validation/test split

#### Part 2: Building First CNN (15 minutes)
- Keras Sequential API
- Add Conv2D layers (16, 32 filters)
- Add MaxPooling2D layers
- Add Flatten and Dense layers
- Model summary and parameter count
- Understanding layer outputs

#### Part 3: Training and Evaluation (15 minutes)
- Compile with optimizer (Adam), loss (categorical_crossentropy)
- Fit with validation data
- Plot training history (accuracy, loss curves)
- Evaluate on test set
- Compare to MLP baseline (if time)

#### Part 4: Visualization (10 minutes)
- Extract and visualize first conv layer filters
- Visualize feature maps for sample image
- Observe hierarchical learning
- Discuss what CNN learned

---

## Learning Resources

### Required Reading (Before DO3)
1. Chollet Ch. 5: "Deep learning for computer vision" (pages 145-165)
2. Goodfellow Ch. 9: "Convolutional Networks" (sections 9.1-9.3)
3. Week 9 DO4 lecture notes review (CNN introduction)

### Supplementary Materials
4. Stanford CS231n notes on CNNs
5. 3Blue1Brown: "But what is a convolution?"
6. Keras documentation: Conv2D, MaxPooling2D layers

### Tutorial Resources
- Fashion-MNIST dataset documentation
- Keras Sequential model guide
- TensorFlow visualization tutorials

---

## Assessment Integration

### Unit Test 2 (Oct 31) - Expected Week 10 Questions

#### MCQ (1 mark each)
- Convolution parameter effects (stride, padding)
- Pooling operation identification
- CNN architecture component ordering
- Parameter counting in conv layers
- Receptive field concept

#### 5-Mark Questions
- Calculate convolution output dimensions given parameters
- Design CNN architecture for specific problem
- Explain biological motivation for CNNs
- Compare CNN vs MLP advantages/disadvantages
- Trace feature map dimensions through network

#### 10-Mark Questions
- Complete convolution calculation (show all steps)
- Design and justify CNN architecture for image classification
- Implement CNN in Keras (code + explanation)
- Analyze CNN layer outputs and learned features

---

## Homework Assignment

**Due: Before Week 11 lecture (Oct 29)**

### Task 1: Convolution Calculation Practice
Calculate output of 2D convolution:
- Input: 6×6 image
- Kernel: 3×3 edge detector
- Stride: 1, Padding: 0
- Show all intermediate steps

### Task 2: Architecture Design
Design CNN for MNIST digit classification:
- Specify layer types, filter counts, kernel sizes
- Calculate output dimensions at each layer
- Justify design choices
- Estimate total parameters

### Task 3: Code Exploration
Modify Tutorial T10 code:
- Add one more convolutional layer
- Experiment with different kernel sizes (3×3, 5×5)
- Compare training time and accuracy
- Document observations (2-3 paragraphs)

---

## Teaching Notes

### Key Pedagogical Strategies

#### 1. Build on Week 9 Foundation
- Reference Detective Maria analogy
- Bridge conceptual → mathematical
- Show continuity from manual features

#### 2. Calculation Before Code
- Work through convolution by hand
- Use simple numbers (3×3, 5×5)
- Board work for transparency
- Only then show code

#### 3. Visual-First Approach
- Draw every architecture component
- Animate sliding window (if possible)
- Show real feature maps
- Use color-coded diagrams

#### 4. Frequent Reality Checks
- "Why is this useful?" after each concept
- Real-world applications
- Compare to student intuition
- Connect to previous modules

### Common Student Struggles

#### Expected Questions:
1. **"How is convolution different from matrix multiplication?"**
   - Answer: Weight sharing, local connectivity, translation equivariance

2. **"Why do we need pooling if conv already reduces dimensions?"**
   - Answer: Translation invariance, computational efficiency, overfitting reduction

3. **"How many filters should I use?"**
   - Answer: Start small (16-32), double after pooling, depends on complexity

4. **"When do I use padding?"**
   - Answer: Valid = shrink, Same = maintain; depends on architecture depth

### Timing Contingencies

**If ahead of schedule:**
- Deep dive into receptive fields
- Show more architecture examples
- Discuss batch normalization preview

**If behind schedule:**
- Skip 3D convolution (move to Week 11)
- Reduce pooling examples
- Simplify architecture walkthrough

---

## Connection to Course Flow

### Backward Links (What Built This)
```
Module 1-2: Neural networks, backprop, optimization
    ↓
Module 3: Image processing, manual feature extraction
    ↓
Week 9: CNN motivation (WHY?) ← PREVIOUS
    ↓
Week 10: CNN mechanics (HOW?) ← CURRENT
```

### Forward Links (What Comes Next)
```
Week 10: Basic CNN mechanics ← TODAY
    ↓
Week 11: CNN layers, regularization, famous architectures
    ↓
Week 12: Transfer learning, pre-trained models
    ↓
Unit Test 2: Modules 3-4 assessment (Oct 31)
```

---

## Deliverables Checklist

### DO3 Materials (2-hour lecture)
- [ ] Comprehensive lecture notes (70-20-10 rule)
- [ ] Slide deck with visual examples
- [ ] Convolution calculation handout
- [ ] Architecture diagrams (LeNet, simple CNN)
- [ ] Output dimension cheat sheet

### DO4 Materials (1-hour tutorial)
- [ ] Tutorial T10 complete code (starter + solution)
- [ ] Dataset loading instructions
- [ ] Architecture design worksheet
- [ ] Visualization code snippets
- [ ] Troubleshooting guide

### Assessment Materials
- [ ] Homework assignment sheet
- [ ] Practice MCQ questions (10 questions)
- [ ] Sample 5-mark questions (3 questions)
- [ ] Sample 10-mark questions (2 questions)

### Student Resources
- [ ] Reading list with page numbers
- [ ] Quick reference guide (convolution formulas)
- [ ] Keras API cheat sheet
- [ ] Week 10 summary handout

---

## Success Metrics

### Students should leave Week 10 able to:
- ✅ Calculate convolution output dimensions manually
- ✅ Explain convolution with visual diagrams
- ✅ Write basic CNN in Keras (3-5 layers)
- ✅ Understand why CNNs work for images
- ✅ Debug common CNN architecture errors
- ✅ Compare different architecture choices

### Red Flags (Intervention Needed):
- ❌ Confusion about convolution vs fully-connected
- ❌ Unable to calculate output dimensions
- ❌ Keras syntax errors blocking progress
- ❌ No connection to Week 9 manual features

---

## Instructor Preparation

### Before DO3 Lecture:
1. Review Week 9 lecture notes (CNN introduction)
2. Prepare convolution calculation examples (3-5 examples)
3. Test all code snippets in fresh environment
4. Create visual aids (architecture diagrams)
5. Print handouts (convolution formulas, dimension calculator)

### Before DO4 Tutorial:
1. Test Tutorial T10 code end-to-end
2. Download datasets (Fashion-MNIST, CIFAR-10)
3. Prepare common error solutions
4. Set up visualization code
5. Create assessment rubric for T10

### Student Prerequisites:
- ✅ Completed Week 9 lectures
- ✅ TensorFlow/Keras environment working
- ✅ Basic NumPy operations
- ✅ Understanding of classification metrics

---

## Related Files

### Lecture Materials
- `do3-oct-22/wip/comprehensive_lecture_notes.md` (to be created)
- `do3-oct-22/wip/slide_deck.md` (to be created)
- `do3-oct-22/wip/convolution_examples.md` (to be created)

### Tutorial Materials
- `do4-oct-23/wip/tutorial_t10_starter.py` (to be created)
- `do4-oct-23/wip/tutorial_t10_solution.py` (to be created)
- `do4-oct-23/wip/comprehensive_tutorial_notes.md` (to be created)

### Assessment Materials
- `week10-homework-assignment.md` (to be created)
- `week10-practice-questions.md` (to be created)

---

## Status
- ✅ Week 10 directory structure created
- ✅ Week 10 plan documented
- ⏳ Lecture materials (pending)
- ⏳ Tutorial materials (pending)
- ⏳ Assessment materials (pending)

**Last Updated:** October 2025
**Next Review:** After Week 10 delivery
