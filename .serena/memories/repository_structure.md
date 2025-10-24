# Repository Structure

## Root Directory
```
Deep_Neural_Network_Architectures/
├── Course Documentation (*.md files)
│   ├── CLAUDE.md - AI assistant instructions
│   ├── Course_Plan.md - 15-week schedule
│   ├── syllabus.md - Official syllabus
│   ├── Books.md - Reading materials
│   └── Assessment_Schedule.md - Test dates
│
├── books/ - PDF textbooks
│   ├── Deep Learning books (Chollet, Goodfellow)
│   └── CNN and MATLAB references
│
├── labs/ - Hands-on exercises
│   ├── simple_mnist_ui.py - Tkinter MNIST app
│   ├── gradioUI.py - Web interface
│   ├── exercise-1/ - Student exercises
│   └── srnenv/ - Primary virtual environment
│
├── course_planning/ - Teaching materials
│   ├── weekly_plans/ - Week-by-week content
│   ├── templates/ - Lecture/tutorial templates
│   ├── monthly_plans/ - Monthly schedules
│   └── assessment_schedules/ - Exam planning
│
├── docs/ - Additional documentation
│   └── chat_history/ - AI conversation tracking
│
└── create_course_handout_excel.py - Handout generator
```

## Key Directories

### /labs
- Contains all practical implementations
- Virtual environments for different exercises
- MNIST demos with GUI interfaces
- Student exercise templates

### /course_planning
- Structured teaching materials
- Weekly and monthly planning documents
- Templates for lectures and tutorials
- Assessment schedules and rubrics

### /books
- Reference textbooks in PDF format
- Deep learning theory and practice books
- CNN and computer vision resources

### /docs/chat_history
- Comprehensive conversation tracking
- Daily summaries and session logs
- Templates for startup/shutdown routines
- CRITICAL: Must track all AI interactions

## File Naming Conventions
- Tutorial tasks: T1-T15 format
- Weekly plans: week-XX-month-DD-DD/
- Sessions: YYYY-MM-DD format
- Python files: lowercase_with_underscores.py
- Markdown docs: Title_Case_or_kebab-case.md