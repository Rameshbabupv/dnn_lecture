# GitHub Classroom Setup Guide
## Deep Neural Network Architectures Course (21CSE558T)

---

## Overview
This guide will transform your current manual GitHub workflow into an automated GitHub Classroom system for better student management and submission tracking.

## Current vs Target Workflow

### **Current State:**
```
You (Instructor) → students_github repo → Students manually clone/sync → Manual submission checking
```

### **Target State:**
```
You → GitHub Classroom → Auto-generated student repos → Centralized dashboard → Automated tracking
```

---

## Step-by-Step Setup Process

### **Step 1: Create GitHub Classroom Organization**
1. **Visit**: https://classroom.github.com
2. **Sign in** with your GitHub account
3. **Create New Organization** (or use existing):
   - **Suggested Name**: `srm-deep-learning-2025`
   - **Description**: "SRM University - Deep Neural Network Architectures Course"
   - **Visibility**: Public (easier for students to find)
4. **Add billing info** (required even for free usage)

### **Step 2: Create Your Classroom**
1. **Click "New Classroom"**
2. **Select** your organization
3. **Classroom Name**: "Deep Neural Network Architectures 2025"
4. **Add student identifiers** (optional): Email addresses or student IDs

### **Step 3: Prepare Template Repository**
1. **Go to your existing** `students_github` repository
2. **Settings → Template repository** → Check "Template repository"
3. **Organize structure** (recommended):
   ```
   students_github/
   ├── course_materials/     # Your lecture content
   ├── assignments/          # Assignment templates
   ├── resources/           # Reference materials
   ├── submissions/         # Folder for student work
   └── README.md           # Course overview
   ```

### **Step 4: Create First Assignment**
1. **In GitHub Classroom** → "New Assignment"
2. **Assignment Title**: "Module 1 - Introduction to Deep Learning"
3. **Repository Name**: `module1-intro-deep-learning`
4. **Template Repository**: Select your `students_github` repo
5. **Deadline**: Set appropriate date
6. **Settings**:
   - ✅ Students get admin access to their repos
   - ✅ Enable assignment deadline
   - ✅ Private repositories (students can't see each other's work)

### **Step 5: Generate Assignment Link**
1. **Copy the assignment invitation link**
2. **Share with students** via email/LMS
3. **Students click link** → Automatic repo creation

---

## Student Workflow (After Setup)

### **For Students:**
1. **Click assignment link** → "Accept Assignment"
2. **GitHub creates** `module1-intro-deep-learning-[username]`
3. **Clone their repo**:
   ```bash
   git clone https://github.com/srm-deep-learning-2025/module1-intro-deep-learning-studentname.git
   ```
4. **Complete assignment** in `submissions/` folder
5. **Commit and push** to submit (no manual submission needed)

### **For You (Instructor):**
1. **Monitor progress** via Classroom dashboard
2. **See all submissions** in one place with timestamps
3. **Download all repos** for grading (bulk download available)
4. **Provide feedback** directly in student repos

---

## Benefits You'll Gain

### **Automated Management**
- ✅ **No more manual repo checking** - centralized dashboard
- ✅ **Automatic deadline tracking** - see who submitted on time
- ✅ **Bulk operations** - download all submissions at once

### **Better Student Experience**
- ✅ **Individual workspaces** - each student has private repo
- ✅ **No Git conflicts** - no need to sync with upstream
- ✅ **Clear instructions** - standardized workflow

### **Enhanced Teaching**
- ✅ **Progress tracking** - see student activity over time
- ✅ **Easy feedback** - comment directly on code
- ✅ **Reusable templates** - create once, use for all assignments

---

## Sample Assignment Structure

### **Module 1 Assignment Example:**
```
module1-intro-deep-learning-[student]/
├── course_materials/
│   ├── lecture_slides.pdf
│   ├── tutorial_notebook.ipynb
│   └── reading_materials.md
├── assignments/
│   ├── task1_perceptron.py
│   ├── task2_mlp.py
│   └── requirements.txt
├── submissions/
│   ├── task1_solution.py      # Student works here
│   ├── task2_solution.py      # Student works here
│   └── reflection.md          # Student works here
└── README.md                  # Assignment instructions
```

---

## Timeline and Next Steps

### **Setup Time**: 30-45 minutes total
### **Immediate Benefits**: Starting from first assignment

### **Action Items After Setup:**
1. **Create student instruction document**
2. **Test workflow** with a sample assignment
3. **Train students** on the new process (15-minute orientation)
4. **Migrate existing assignments** to classroom format

---

## Troubleshooting Common Issues

### **Student Issues:**
- **"Can't accept assignment"** → Check GitHub account permissions
- **"Repo not created"** → Students need to refresh page after accepting

### **Instructor Issues:**
- **"Template not working"** → Ensure repo is marked as template
- **"Students can't see content"** → Check repository visibility settings

---

## Resources for Reference
- **GitHub Classroom Docs**: https://docs.github.com/en/education/manage-coursework-with-github-classroom
- **Video Tutorial**: GitHub's official classroom setup guide
- **Student Onboarding**: Template instructions included in setup

---

**Ready to Start?** Begin with Step 1 when you return to this setup process.