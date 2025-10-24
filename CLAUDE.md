# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Course Overview

This repository contains course materials for **Deep Neural Network Architectures (21CSE558T)**, a 3-credit M.Tech course at SRM University. The course runs for 15 weeks with 45 total contact hours (2L + 1T + 0P format).

### Course Structure
- **5 Modules**: Introduction to Deep Learning → Optimization & Regularization → Image Processing & DNNs → CNNs & Transfer Learning → Object Detection
- **15 Practical Tasks**: Hands-on implementations using TensorFlow/Keras, OpenCV
- **Assessment**: 45% Unit Tests + 15% Lab/Practice + 40% Final Exam

## Key Course Components

### Learning Technologies Stack
- **Primary**: TensorFlow 2.x with Keras API
- **Image Processing**: OpenCV
- **Platform**: Google Colab (recommended for students)
- **Languages**: Python (primary), MATLAB (supplementary)

### Module Progression
1. **Module 1 (Weeks 1-3)**: Perceptron → MLP → TensorFlow basics → Activation functions
2. **Module 2 (Weeks 4-6)**: Gradient descent variants → Regularization → Normalization
3. **Module 3 (Weeks 7-9)**: Image processing → Feature extraction → Classification
4. **Module 4 (Weeks 10-12)**: CNN architectures → Transfer learning → Pre-trained models
5. **Module 5 (Weeks 13-15)**: Object detection → YOLO/SSD → R-CNN family

## File Structure and Content

### Primary Course Files
- **Course Plan.md**: Complete 15-week schedule with learning objectives, assessments, and timeline
- **syllabus.md**: Official university syllabus with course outcomes and module details
- **Course Planning n Delivery.md**: Teaching methodology and practical implementation guidance
- **Books.md**: Recommended textbooks and reference materials
- **Question to follow.md**: Course-related queries and considerations
- **Monthly plans.md**: Granular monthly planning documents

### Resource Directories
- **books/**: PDF textbooks including Chollet's "Deep Learning with Python", Goodfellow's "Deep Learning", and specialized texts on CNNs and MATLAB implementations
- **docs/chat_history/**: Comprehensive conversation tracking system for maintaining continuity across Claude sessions

## Chat History System (MANDATORY)

**CRITICAL REQUIREMENT**: This project uses the comprehensive chat history system in `docs/chat_history/`. 

**Every Claude AI session MUST:**
1. Read `docs/chat_history/CLAUDE.md` first
2. **ASK USER FOR CURRENT DATE AND TIME** - Never assume or use system date/time
3. Execute morning startup routine from `templates/morning-startup-routine.md`
4. Capture ALL significant Q&A using embedded Q&A format with ACTUAL user-provided timestamps
5. End session with `templates/end-of-day-ceremony.md` checklist

## ⚠️ CRITICAL DATE/TIME REQUIREMENT

**MANDATORY**: Always ask the user for current date and time before creating any timestamped files or entries.

**NEVER:**
- Use system date/time information
- Assume current date from environment
- Guess timestamps based on conversation flow

**ALWAYS:**
- Ask user: "What is the current date and time in Eastern Time?"
- Use the EXACT date and time provided by the user
- Apply user's timezone information consistently
- Create files with user-confirmed date format (YYYY-MM-DD)

**System Location**: All chat history files in `docs/chat_history/sessions/` and `docs/chat_history/daily_summaries/`

## Common Development Tasks

### Course Content Development
- **Lecture Material**: Create structured content following the progressive module approach
- **Tutorial Tasks**: Develop hands-on programming exercises using TensorFlow/Keras
- **Assessment Design**: Build unit tests covering theoretical concepts and practical implementation

### Student Resource Creation
- **Code Examples**: Implement reference solutions for the 15 tutorial tasks (T1-T15)
- **Visual Aids**: Create diagrams for neural network architectures and concepts
- **Assignment Templates**: Design structured programming assignments with evaluation criteria

### Teaching Support Materials
- **Presentation Templates**: Develop slide decks for each module
- **Lab Notebooks**: Create Google Colab notebooks for hands-on sessions
- **Evaluation Rubrics**: Design assessment criteria for practical work

## Course-Specific Guidelines

### Academic Standards
- Follow progressive learning: Each module builds upon previous concepts
- Maintain theory-practice balance: 40-60% split recommended
- Ensure practical tasks align with learning objectives (CO-1 through CO-5)

### Implementation Approach
- **Beginner-Friendly**: Start with basic concepts (perceptron) before advanced topics (object detection)
- **Hands-On Focus**: Every theoretical concept should have corresponding practical implementation
- **Industry Relevance**: Use current frameworks and pre-trained models (ImageNet, YOLO, etc.)

### Assessment Alignment
- **Unit Tests**: Cover modules 1-2 (Sep 19) and modules 3-4 (Oct 31)
- **Practical Evaluation**: Continuous assessment of tutorial implementations
- **Final Examination**: Comprehensive coverage with integration emphasis

## Key Learning Outcomes Timeline

- **Week 6**: Simple deep neural networks (CO-1)
- **Week 9**: Multi-layer networks with activations (CO-2)  
- **Week 12**: Deep learning for image processing (CO-3)
- **Week 15**: CNNs and transfer learning (CO-4, CO-5)

## Project Context

This is an **academic course repository** focused on teaching deep learning concepts progressively. When working with this codebase:
- Prioritize educational clarity over production optimization
- Include detailed explanations suitable for M.Tech students
- Maintain alignment with university assessment requirements
- Consider diverse student backgrounds in programming experience

## Quick Reference

**Course Code**: 21CSE558T  
**Credits**: 3 (2L + 1T + 0P)  
**Duration**: 15 weeks (Aug 11 - Nov 21, 2025)  
**Target Audience**: M.Tech students  
**Prerequisites**: Basic Python, linear algebra, calculus

## Technical Environment & Commands

### Python Environment Setup
The project uses multiple Python virtual environments:
- **labs/srnenv/**: Primary lab environment with TensorFlow 2.20.0, OpenCV, NumPy
- **labs/exercise-1/mnist-env/**: MNIST-specific environment  
- **course_planning/.../tensorflow_env/**: Tutorial-specific environments

**Activating environments:**
```bash
# Primary lab environment
source labs/srnenv/bin/activate

# MNIST exercise environment  
source labs/exercise-1/mnist-env/bin/activate
```

### Running Lab Applications
```bash
# Desktop MNIST application (Tkinter-based)
cd labs && python simple_mnist_ui.py

# Web interface (Gradio-based - has compatibility issues)
cd labs && python gradioUI.py

# Course handout generation
python create_course_handout_excel.py
```

### Dependencies & Requirements
- **TensorFlow**: 2.x with Keras API (primary deep learning framework)
- **OpenCV**: 4.x for computer vision tasks  
- **NumPy**: For numerical computations
- **Matplotlib**: For visualization
- **Gradio**: For web interfaces (compatibility issues with Python 3.13)
- **Tkinter**: For desktop GUI applications

Install dependencies from requirements files:
```bash
pip install -r course_planning/weekly_plans/week-02-aug-18-22/day4/requirements.txt
```

## Architecture & Code Structure

### Repository Organization
```
├── Course planning & documentation (root level .md files)
├── books/                     # PDF textbooks and references
├── chat_history/              # Comprehensive conversation tracking system  
├── course_planning/           # Structured weekly/monthly plans
│   ├── templates/             # Lecture and tutorial templates
│   └── weekly_plans/          # Week-by-week course delivery
├── labs/                      # Hands-on programming exercises
│   ├── *.py                   # Lab applications (MNIST, Gradio UI)
│   ├── srnenv/                # Primary virtual environment
│   └── exercise-1/            # Student exercise implementations
├── docs/                      # Additional documentation
└── sample-docs/               # Reference documents
```

### Key Technical Components

**Lab Applications Architecture:**
- **simple_mnist_ui.py**: Tkinter-based MNIST digit recognition with drawing canvas
- **gradioUI.py**: Web-based interface (has Python 3.13 compatibility issues)  
- **Neural Network**: MLP with 128→64→10 architecture, dropout regularization
- **Preprocessing Pipeline**: Image normalization, bounding box detection, 28×28 resizing

**Course Generation System:**
- **create_course_handout_excel.py**: Automated Excel workbook generation
- **Multi-sheet structure**: Course overview, learning outcomes, assessments, schedules
- **Data-driven approach**: Uses structured dictionaries for content management

### Development Patterns

**Neural Network Implementation Pattern:**
```python
# Standard model architecture used throughout course
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'), 
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

**Image Preprocessing Pattern:**
- Convert to grayscale → Invert colors → Find bounding box → Resize to 28×28
- Normalization to [0,1] range for neural network input

### Tutorial Task Structure (T1-T15)
Each tutorial follows progressive complexity:
- **T1-T3**: TensorFlow environment, tensors, basic operations
- **T4-T6**: Neural networks, Keras implementation, gradient descent  
- **T7-T9**: Image processing, segmentation, feature extraction
- **T10-T12**: CNN classification, data augmentation, LSTM
- **T13-T15**: Pre-trained models, transfer learning, object detection

## Troubleshooting Common Issues

### Lab Environment Issues
- **Tkinter import errors**: System-level tkinter installation required
- **Gradio compatibility**: Python 3.13 has audioop/pyaudioop module issues
- **TensorFlow warnings**: Protobuf version warnings are non-critical
- **Model not found**: Applications auto-train new models when needed

### Development Workflow
1. Plan using weekly templates in `course_planning/templates/`
2. Implement exercises following T1-T15 progression
3. Test technical implementations in appropriate virtual environments
4. Document using chat history system for continuity