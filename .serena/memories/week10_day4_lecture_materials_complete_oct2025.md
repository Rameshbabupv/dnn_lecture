# Week 10 Day 4 (DO4) Lecture Materials - Complete Status

**Course:** Deep Neural Network Architectures (21CSE558T)
**Module 4:** CNN Basics - Week 1 of 3
**Session:** DO4 - October 29, 2025 (Monday), 4:00-4:50 PM IST
**Last Updated:** October 26, 2025

---

## Session Format Change

**IMPORTANT:** Week 10 Day 4 is a **LECTURE-ONLY session**, NOT a hands-on tutorial.
- Originally planned as Tutorial T10 (hands-on coding)
- Changed to lecture format (instructor demonstrates, students watch)
- Duration: 50 minutes (1 hour session)

---

## Complete Materials Created (12 Files)

**Location:** `/Users/rameshbabu/data/projects/srm/lectures/Deep_Neural_Network_Architectures/course_planning/weekly_plans/week10-module4-cnn-basics/do4-oct-29-Monday/`

### Core Teaching Materials:

| # | File Name | Size | Purpose | Usage |
|---|-----------|------|---------|-------|
| 1 | **v1_comprehensive_lecture_notes.md** | 45K | Complete 50-min lecture guide with timeline, scripts, code demos | PRIMARY - Use during lecture |
| 2 | **tutorial_t10_solution.py** | 22K | Working code for live demonstrations (90% accuracy) | Reference during lecture |
| 3 | **tutorial_t10_fashion_mnist_cnn.ipynb** | 28K | Interactive notebook (29 cells) - alternative presentation format | Optional lecture format |
| 4 | **quick_reference_cheat_sheet.md** | 9.6K | 2-page code snippets, formulas, common errors | Share with students |

### Student Materials (Post-Lecture):

| # | File Name | Purpose | When to Share |
|---|-----------|---------|---------------|
| 5 | **tutorial_t10_starter.py** | Template for self-practice (TODO markers) | After lecture |
| 6 | **architecture_design_worksheet.md** | Pre-coding planning exercise | After lecture (optional) |
| 7 | **troubleshooting_guide.md** | 17 common errors + solutions | After lecture |
| 8 | **week10_homework_assignment.md** | 3 tasks, 100 points, due Nov 3 | After lecture |
| 9 | **week10_practice_questions.md** | Unit Test 2 prep (12 MCQ + 5 SAQ + 2 LAQ) | After lecture |

### Documentation:

| # | File Name | Purpose |
|---|-----------|---------|
| 10 | **README.md** | Complete materials overview and usage guide |
| 11 | **COMPLETION_SUMMARY.md** | Project status and deliverables checklist |
| 12 | **create_jupyter_notebook.py** | Script to regenerate Jupyter notebook |

**Total Files:** 12
**Total Size:** ~202K
**Status:** ✅ All complete, production ready

---

## Lecture Notes Structure (v1_comprehensive_lecture_notes.md)

### Timeline (50 minutes):
- **0:00-0:05** (5 min): Introduction, objectives, recap
- **0:05-0:15** (10 min): Data loading and preprocessing
- **0:15-0:30** (15 min): CNN architecture building (CORE)
- **0:30-0:40** (10 min): Model training and history
- **0:40-0:48** (8 min): Results and visualizations
- **0:48-0:50** (2 min): Wrap-up and next steps

### Content Coverage:
1. **Fashion-MNIST dataset** (10 classes, 28×28 images)
2. **Preprocessing** (normalize, reshape, one-hot encoding)
3. **CNN architecture** (2 conv blocks + dense layers)
4. **Output dimension calculations** (formula: (W-F+2P)/S+1)
5. **Parameter counting** (121,930 total parameters)
6. **Training** (10 epochs, Adam optimizer, categorical crossentropy)
7. **Evaluation** (~90% test accuracy expected)
8. **Visualizations** (filters, feature maps, predictions)

### Teaching Approach:
- **70%** Explanations (concepts, why things work)
- **20%** Code walkthroughs (how to implement)
- **10%** Visualizations (results interpretation)

### Key Features:
- ✅ Complete teaching scripts (word-for-word)
- ✅ Live code demonstrations (what to type)
- ✅ Explanations for each concept (WHY focus)
- ✅ Student engagement questions
- ✅ Common student questions + answers
- ✅ Instructor tips and backup plans
- ✅ Pre/post lecture checklists

---

## Materials Priority for Lecture

### HIGH PRIORITY (Use During Lecture):
1. **v1_comprehensive_lecture_notes.md** - Main teaching guide
2. **tutorial_t10_solution.py** - Live code demonstrations
3. **quick_reference_cheat_sheet.md** - Share with students

### MEDIUM PRIORITY (Reference):
4. **tutorial_t10_fashion_mnist_cnn.ipynb** - Alternative presentation
5. **architecture_design_worksheet.md** - Example design thinking
6. **README.md** - Overview and context

### LOW PRIORITY (Post-Lecture):
7. **tutorial_t10_starter.py** - For student self-practice
8. **week10_homework_assignment.md** - Assign after lecture
9. **week10_practice_questions.md** - Unit Test 2 prep
10. **troubleshooting_guide.md** - Student reference
11. **COMPLETION_SUMMARY.md** - Internal documentation
12. **create_jupyter_notebook.py** - Development tool

---

## Expected Lecture Outcomes

### Model Performance (when demonstrating code):
- Training accuracy: 95-97% (after 10 epochs)
- Validation accuracy: 90-92%
- Test accuracy: 90-92%
- Training time: 2-3 minutes (CPU), 30 seconds (GPU)
- Total parameters: 121,930

### Student Learning (after lecture):
Students should be able to:
- ✅ Explain CNN architecture components
- ✅ Calculate output dimensions using formula
- ✅ Understand why CNNs work for images
- ✅ Write basic CNN in Keras (with reference)
- ✅ Interpret training curves
- ✅ Recognize learned filters and feature maps

---

## Connection to Course Timeline

### Backward Links:
- **Week 9 DO4 (Oct 17):** CNN Introduction - WHY CNNs exist
- **Week 10 DO3 (Oct 27):** CNN Theory - 10 Jupyter notebooks (00-09)
- **Today (Oct 29):** CNN Implementation - HOW to code CNNs

### Forward Links:
- **October 31 (Friday):** Unit Test 2 (Modules 3-4)
  - Will include CNN questions from Week 10
  - Practice questions available
- **Week 11:** Famous CNN architectures (LeNet, AlexNet, VGG, ResNet)
- **Week 12:** Transfer learning and pre-trained models

---

## Homework Assignment Details

**Due:** November 3, 2025 (before Week 11)
**Total Points:** 100 (+5 bonus)

### Tasks:
1. **Task 1 (30 pts):** Manual convolution calculation
   - 6×6 image, 3×3 kernel
   - Show all steps
   - Interpret results

2. **Task 2 (40 pts):** CNN architecture design for MNIST
   - Complete architecture specification
   - Output dimension calculations
   - Design justifications
   - Training configuration

3. **Task 3 (30 pts):** Code experimentation
   - Experiment 1: Add third conv block
   - Experiment 2: Compare kernel sizes (3×3, 5×5, 7×7)
   - Experiment 3: Vary filter counts (16→32, 32→64, 64→128)
   - Document observations (2-3 paragraphs)

**Deliverables:** ZIP file with 4 documents
- task1_calculations.pdf
- task2_architecture_design.pdf
- task3_code.py
- task3_observations.pdf

---

## Unit Test 2 Preparation

**Date:** October 31, 2025 (2 days after lecture!)
**Coverage:** Modules 3-4 (Week 10 content ~30%)

### Week 10 Question Types Expected:
- **MCQ (1 mark):** Output dimensions, parameter counting, layer purposes
- **SAQ (5 marks):** Architecture design, dimension calculations, CNN vs MLP
- **LAQ (10 marks):** Complete convolution calculation, Keras implementation

### Practice Materials:
- week10_practice_questions.md contains:
  - 12 MCQ with answers
  - 5 SAQ with solutions
  - 2 LAQ with marking schemes

---

## Instructor Preparation Checklist

### Before Lecture (October 29):
- [ ] Read v1_comprehensive_lecture_notes.md completely
- [ ] Test tutorial_t10_solution.py (verify 90% accuracy)
- [ ] Pre-generate visualizations (backup if live code fails)
- [ ] Have Google Colab ready (backup environment)
- [ ] Print/share quick_reference_cheat_sheet.md
- [ ] Review "Common Student Questions" section
- [ ] Test projector/screen setup

### During Lecture:
- [ ] Follow 50-minute timeline
- [ ] Live code tutorial_t10_solution.py
- [ ] Explain WHY for each design choice
- [ ] Show visualizations (filters, feature maps)
- [ ] Engage students with questions
- [ ] Leave 2 minutes for Q&A

### After Lecture:
- [ ] Share tutorial_t10_solution.py
- [ ] Share tutorial_t10_fashion_mnist_cnn.ipynb
- [ ] Share quick_reference_cheat_sheet.md
- [ ] Upload week10_homework_assignment.md
- [ ] Share week10_practice_questions.md (Unit Test 2 prep)
- [ ] Collect student feedback

---

## Technical Requirements

### Software:
- Python 3.8+
- TensorFlow 2.15+ (with Keras)
- NumPy 1.24+
- Matplotlib 3.7+

### Hardware:
- CPU: 4GB RAM minimum (2-3 min training)
- GPU: 8GB VRAM recommended (30 sec training)
- Alternative: Google Colab (free GPU)

### Installation:
```bash
pip install tensorflow numpy matplotlib
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## Key Teaching Points (From Lecture Notes)

### Critical Concepts to Emphasize:

1. **Reshape is Critical:**
   - CNN expects 4D: (samples, height, width, channels)
   - Common error: forgetting to add channel dimension
   - Show the error if forgotten!

2. **Weight Sharing:**
   - Same 3×3 filter used across ENTIRE image
   - Parameter efficiency vs fully connected
   - Example: 3×3 filter = 9 parameters regardless of image size

3. **Hierarchical Learning:**
   - First layers: simple features (edges)
   - Deep layers: complex features (objects)
   - Learned automatically through backpropagation

4. **Output Dimension Formula:**
   - (W - F + 2P) / S + 1
   - Students WILL be tested on this
   - Practice calculating manually

5. **CNN vs MLP:**
   - CNN: ~122,000 parameters, 90% accuracy
   - MLP: Similar parameters, 85-88% accuracy
   - CNN learns spatial structure, MLP doesn't

### Common Student Misconceptions:

❌ "CNNs only work for images" → ✓ Also audio, text, time series
❌ "Filters are hand-designed" → ✓ Learned automatically
❌ "More filters = always better" → ✓ Risk of overfitting
❌ "Pooling learns features" → ✓ 0 parameters, just downsamples
❌ "Validation = training accuracy" → ✓ Small gap is normal

---

## Session Format Notes

**LECTURE-ONLY means:**
- Instructor demonstrates code live
- Students watch and take notes
- No hands-on coding during session
- Materials shared after for self-practice

**NOT a hands-on tutorial:**
- Students don't code during session
- tutorial_t10_starter.py used for post-lecture practice
- Homework assignment for hands-on experience

**Why lecture instead of tutorial:**
- Instructor decision for Week 10
- Full materials still created (can switch format later)
- Focus on understanding before practicing

---

## Success Metrics

### Lecture Success Indicators:
- Students can explain each CNN layer type
- Students can calculate output dimensions
- Students understand WHY CNNs work for images
- Students can write basic CNN (with reference)
- Questions focus on concepts, not syntax errors

### Red Flags:
- Confusion about input shapes (4D vs 3D)
- Not understanding convolution vs fully connected
- Thinking filters are manually designed
- Unable to follow output dimension calculations
- Lost during architecture building section

---

## Files Status Summary

| Category | Files | Status |
|----------|-------|--------|
| Lecture Materials | 4 files | ✅ Complete |
| Student Practice | 5 files | ✅ Complete |
| Documentation | 3 files | ✅ Complete |
| **Total** | **12 files** | ✅ **Production Ready** |

**All materials created:** October 26, 2025
**Ready for delivery:** October 29, 2025
**No modifications needed unless requested**

---

## Related Memories

- `week10_jupyter_notebooks_complete_oct2025` - DO3 lecture materials (10 notebooks)
- `week10_current_work_status_oct2025` - Overall Week 10 status
- `week10_cnn_basics_plan` - Week 10 planning document
- `week10_diwali_schedule_adjustment_oct2025` - Schedule changes
- `week10_do3_lecture_notes_v2_status` - DO3 lecture notes details

---

## Next Steps

### Immediate (Before Oct 29):
1. Instructor reviews v1_comprehensive_lecture_notes.md
2. Test tutorial_t10_solution.py execution
3. Prepare lecture environment
4. Review backup plans

### After Lecture (Oct 29):
1. Share all student materials
2. Assign homework (due Nov 3)
3. Share practice questions (Unit Test 2 prep)
4. Collect feedback for future improvements

### Future (Week 11+):
1. Famous CNN architectures lecture
2. Transfer learning materials
3. Use feedback to improve tutorial format

---

**Status:** ✅ All Week 10 Day 4 materials complete and production ready
**Format:** Lecture-only (50 minutes)
**Primary Material:** v1_comprehensive_lecture_notes.md
**Last Updated:** October 26, 2025
