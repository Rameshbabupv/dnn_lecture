# Sample Images Guide for Tutorial
## Week 8, Day 4 - ROI Extraction & Morphological Operations

**Course**: Deep Neural Network Architectures (21CSE558T)
**Date**: October 9, 2025
**Purpose**: Guide for preparing/obtaining sample images for the tutorial

---

## 📁 Directory Structure

Create the following folder structure before the tutorial:

```
course_planning/weekly_plans/week8-module3-segmentation/do4-oct-09/
├── images/                    # Input images folder
│   ├── group_photo.jpg       # For ROI extraction demo
│   ├── coins.jpg             # For contour-based extraction
│   ├── noisy_document.jpg    # For morphology opening demo
│   ├── broken_text.jpg       # For morphology closing demo
│   ├── scanned_doc.jpg       # For integration pipeline
│   └── shapes.jpg            # For gradient demo
└── output/                    # Results will be saved here
    └── (empty - created by code)
```

---

## 🖼️ Required Sample Images

### 1. **Group Photo** (`group_photo.jpg`)

**Purpose**: Basic ROI extraction using rectangular coordinates

**Specifications**:
- Resolution: 800×600 or similar
- Content: Group of people, clearly visible faces
- Format: JPG or PNG
- Quality: Good lighting, clear faces

**Where to Get**:
- **Option 1**: Take a photo of your class/group
- **Option 2**: Download from free stock sites:
  - Unsplash: https://unsplash.com/s/photos/group-photo
  - Pexels: https://www.pexels.com/search/group%20photo/
  - Search: "group photo", "team photo", "family photo"

**How to Use**:
```python
# Students will extract individual faces
img = cv2.imread('images/group_photo.jpg')
roi = img[100:300, 150:350]  # Extract one face
```

**Alternative**: If no suitable photo, create synthetic:
```python
# Synthetic group photo (fallback)
img = np.zeros((600, 800, 3), dtype=np.uint8)
cv2.rectangle(img, (100, 100), (300, 400), (255, 200, 150), -1)  # Face 1
cv2.rectangle(img, (400, 150), (600, 450), (255, 180, 130), -1)  # Face 2
cv2.imwrite('images/group_photo.jpg', img)
```

---

### 2. **Coins Image** (`coins.jpg`)

**Purpose**: Contour-based ROI extraction with multiple objects

**Specifications**:
- Resolution: 600×400 or similar
- Content: Multiple coins on plain background
- Background: Contrasting color (white/black recommended)
- Format: JPG or PNG

**Where to Get**:
- **Option 1**: Take photo of real coins
  - Place 5-8 coins on white paper
  - Good lighting from above
  - Use phone camera
  - Ensure high contrast

- **Option 2**: Download:
  - Search: "coins on white background", "coin detection dataset"
  - COCO dataset: https://cocodataset.org/ (filter by "coin")

**How to Create Your Own**:
```python
# Steps to photograph coins:
# 1. Place white A4 paper on table
# 2. Arrange 5-8 coins (not touching)
# 3. Use phone camera from directly above
# 4. Ensure even lighting (no shadows)
# 5. Save as coins.jpg
```

**Synthetic Alternative**:
```python
# Create synthetic coins image
coins = np.zeros((400, 600), dtype=np.uint8)
cv2.circle(coins, (100, 100), 40, 255, -1)
cv2.circle(coins, (250, 150), 50, 255, -1)
cv2.circle(coins, (450, 120), 45, 255, -1)
cv2.circle(coins, (200, 300), 35, 255, -1)
cv2.imwrite('images/coins.jpg', coins)
```

---

### 3. **Noisy Document** (`noisy_document.jpg`)

**Purpose**: Morphological opening to remove noise

**Specifications**:
- Resolution: 800×600 or similar
- Content: Text with salt-and-pepper noise
- Background: Light colored
- Text: Dark colored

**Where to Get**:
- **Option 1**: Create from clean document
  ```python
  # Add noise to clean document
  import cv2
  import numpy as np

  # Load clean document
  clean = cv2.imread('clean_doc.jpg', 0)

  # Add salt-and-pepper noise
  noise = np.random.randint(0, 2, clean.shape) * 255
  noise = (noise * 0.05).astype(np.uint8)  # 5% noise
  noisy = cv2.bitwise_or(clean, noise)

  cv2.imwrite('images/noisy_document.jpg', noisy)
  ```

- **Option 2**: Download noisy document images
  - Search: "noisy document image processing"
  - Dataset: Document Image Denoising datasets

- **Option 3**: Scan old document
  - Use low-quality scanner settings
  - Old paper naturally has noise

**Synthetic Creation**:
```python
# Create noisy document from scratch
doc = np.zeros((600, 800), dtype=np.uint8)

# Add text blocks
cv2.rectangle(doc, (50, 100), (350, 150), 255, -1)
cv2.rectangle(doc, (50, 200), (400, 250), 255, -1)
cv2.rectangle(doc, (50, 300), (300, 350), 255, -1)

# Add noise
noise = np.random.randint(0, 256, doc.shape).astype(np.uint8)
noise = (noise > 250).astype(np.uint8) * 255  # 2% noise
noisy_doc = cv2.bitwise_or(doc, noise)

cv2.imwrite('images/noisy_document.jpg', noisy_doc)
```

---

### 4. **Broken Text** (`broken_text.jpg`)

**Purpose**: Morphological closing to fill holes

**Specifications**:
- Resolution: 400×300 or similar
- Content: Text with gaps/holes inside letters
- Format: Binary (black text on white) or grayscale

**How to Create**:

**Method 1: From printed text**
```python
# 1. Print text in large font (Arial Black, 72pt)
# 2. Photocopy it with low toner/ink
# 3. Result: faded text with holes
# 4. Scan and save as broken_text.jpg
```

**Method 2: Digital creation**
```python
# Create text with holes
text_img = np.zeros((300, 400), dtype=np.uint8)

# Draw text blocks
cv2.rectangle(text_img, (50, 100), (120, 200), 255, -1)
cv2.rectangle(text_img, (140, 100), (210, 200), 255, -1)

# Add holes (black circles)
cv2.circle(text_img, (85, 150), 10, 0, -1)
cv2.circle(text_img, (175, 150), 12, 0, -1)

cv2.imwrite('images/broken_text.jpg', text_img)
```

**Method 3: Erosion of normal text**
```python
# Create broken text from normal text
normal = cv2.imread('normal_text.jpg', 0)
kernel = np.ones((3,3), np.uint8)
eroded = cv2.erode(normal, kernel, iterations=2)
cv2.imwrite('images/broken_text.jpg', eroded)
```

---

### 5. **Scanned Document** (`scanned_doc.jpg`)

**Purpose**: Complete pipeline (segmentation → cleaning → ROI extraction)

**Specifications**:
- Resolution: 1000×800 or A4 size
- Content: Document with multiple text blocks
- Quality: Moderate (some noise acceptable)
- Background: Should be separable from text

**Where to Get**:
- **Option 1**: Scan actual document
  - Use office scanner
  - Medium quality settings (150-200 DPI)
  - Ensure some text regions are separate

- **Option 2**: Download sample documents
  - ICDAR dataset: https://rrc.cvc.uab.es/
  - DocBank dataset
  - Search: "scanned document dataset"

- **Option 3**: Create composite document
  ```python
  # Create multi-block document
  doc = np.ones((800, 1000), dtype=np.uint8) * 240  # Light gray bg

  # Multiple text blocks
  cv2.rectangle(doc, (50, 50), (400, 150), 0, -1)
  cv2.rectangle(doc, (450, 50), (950, 150), 0, -1)
  cv2.rectangle(doc, (50, 200), (950, 300), 0, -1)
  cv2.rectangle(doc, (50, 350), (600, 450), 0, -1)

  # Add realistic noise
  noise = np.random.randint(0, 50, doc.shape).astype(np.uint8)
  scanned = np.clip(doc + noise - 25, 0, 255).astype(np.uint8)

  cv2.imwrite('images/scanned_doc.jpg', scanned)
  ```

---

### 6. **Shapes Image** (`shapes.jpg`)

**Purpose**: Morphological gradient demonstration

**Specifications**:
- Resolution: 500×500 or similar
- Content: Various geometric shapes on plain background
- Format: Binary (white shapes on black background)

**How to Create**:
```python
# Create shapes image
shapes = np.zeros((500, 500), dtype=np.uint8)

# Draw various shapes
cv2.rectangle(shapes, (50, 50), (150, 150), 255, -1)      # Square
cv2.circle(shapes, (350, 100), 60, 255, -1)               # Circle
cv2.ellipse(shapes, (250, 350), (80, 50), 0, 0, 360, 255, -1)  # Ellipse

# Triangle
pts = np.array([[100, 400], [150, 300], [200, 400]], np.int32)
cv2.fillPoly(shapes, [pts], 255)

cv2.imwrite('images/shapes.jpg', shapes)
```

---

## 🔧 Image Preparation Script

### Automated Setup Script

Create `prepare_images.py` to generate all synthetic images:

```python
import cv2
import numpy as np
import os

# Create directories
os.makedirs('images', exist_ok=True)
os.makedirs('output', exist_ok=True)

def create_group_photo():
    """Create synthetic group photo"""
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    # Background
    img[:, :] = (200, 220, 240)
    # Face 1
    cv2.rectangle(img, (100, 150), (250, 400), (255, 220, 180), -1)
    cv2.circle(img, (175, 250), 30, (0, 0, 0), -1)  # Eyes
    cv2.circle(img, (175, 350), 20, (150, 50, 50), -1)  # Mouth
    # Face 2
    cv2.rectangle(img, (350, 100), (500, 350), (255, 200, 160), -1)
    cv2.circle(img, (425, 200), 30, (0, 0, 0), -1)  # Eyes
    cv2.circle(img, (425, 300), 20, (150, 50, 50), -1)  # Mouth
    # Face 3
    cv2.rectangle(img, (550, 200), (700, 450), (255, 210, 170), -1)
    cv2.circle(img, (625, 300), 30, (0, 0, 0), -1)  # Eyes
    cv2.circle(img, (625, 400), 20, (150, 50, 50), -1)  # Mouth

    cv2.imwrite('images/group_photo.jpg', img)
    print("✓ Created: group_photo.jpg")

def create_coins():
    """Create synthetic coins image"""
    coins = np.zeros((400, 600), dtype=np.uint8)
    cv2.circle(coins, (100, 100), 40, 255, -1)
    cv2.circle(coins, (250, 150), 50, 255, -1)
    cv2.circle(coins, (450, 120), 45, 255, -1)
    cv2.circle(coins, (200, 300), 35, 255, -1)
    cv2.circle(coins, (400, 300), 55, 255, -1)
    # Add small noise dot
    cv2.circle(coins, (50, 350), 5, 255, -1)

    cv2.imwrite('images/coins.jpg', coins)
    print("✓ Created: coins.jpg")

def create_noisy_document():
    """Create noisy document"""
    doc = np.zeros((600, 800), dtype=np.uint8)
    # Text blocks
    cv2.rectangle(doc, (50, 100), (350, 150), 255, -1)
    cv2.rectangle(doc, (50, 200), (400, 250), 255, -1)
    cv2.rectangle(doc, (50, 300), (300, 350), 255, -1)
    cv2.rectangle(doc, (50, 400), (450, 450), 255, -1)

    # Add salt-and-pepper noise
    noise = np.random.randint(0, 256, doc.shape).astype(np.uint8)
    noise = (noise > 245).astype(np.uint8) * 255  # 4% noise
    noisy = cv2.bitwise_or(doc, noise)

    cv2.imwrite('images/noisy_document.jpg', noisy)
    print("✓ Created: noisy_document.jpg")

def create_broken_text():
    """Create text with holes"""
    text = np.zeros((300, 400), dtype=np.uint8)
    # Text blocks
    cv2.rectangle(text, (50, 100), (120, 200), 255, -1)
    cv2.rectangle(text, (140, 100), (210, 200), 255, -1)
    cv2.rectangle(text, (230, 100), (300, 200), 255, -1)

    # Add holes
    cv2.circle(text, (85, 150), 12, 0, -1)
    cv2.circle(text, (175, 150), 10, 0, -1)
    cv2.circle(text, (265, 150), 14, 0, -1)

    cv2.imwrite('images/broken_text.jpg', text)
    print("✓ Created: broken_text.jpg")

def create_scanned_document():
    """Create scanned document with multiple regions"""
    doc = np.ones((800, 1000), dtype=np.uint8) * 240

    # Text blocks (inverted for processing)
    cv2.rectangle(doc, (50, 50), (400, 150), 20, -1)
    cv2.rectangle(doc, (450, 50), (950, 150), 20, -1)
    cv2.rectangle(doc, (50, 200), (950, 300), 20, -1)
    cv2.rectangle(doc, (50, 350), (600, 450), 20, -1)

    # Add realistic noise
    noise = np.random.randint(-20, 20, doc.shape).astype(np.int16)
    scanned = np.clip(doc + noise, 0, 255).astype(np.uint8)

    cv2.imwrite('images/scanned_doc.jpg', scanned)
    print("✓ Created: scanned_doc.jpg")

def create_shapes():
    """Create shapes for gradient demo"""
    shapes = np.zeros((500, 500), dtype=np.uint8)

    # Various shapes
    cv2.rectangle(shapes, (50, 50), (150, 150), 255, -1)
    cv2.circle(shapes, (350, 100), 60, 255, -1)
    cv2.ellipse(shapes, (250, 350), (80, 50), 0, 0, 360, 255, -1)

    # Triangle
    pts = np.array([[100, 400], [150, 300], [200, 400]], np.int32)
    cv2.fillPoly(shapes, [pts], 255)

    cv2.imwrite('images/shapes.jpg', shapes)
    print("✓ Created: shapes.jpg")

# Generate all images
if __name__ == "__main__":
    print("Generating sample images...")
    print("-" * 40)

    create_group_photo()
    create_coins()
    create_noisy_document()
    create_broken_text()
    create_scanned_document()
    create_shapes()

    print("-" * 40)
    print("✅ All sample images created successfully!")
    print("\nImages saved in: ./images/")
    print("Output will be saved in: ./output/")
    print("\nYou can now run the tutorial notebook!")
```

### Run the Script

```bash
# Navigate to tutorial directory
cd course_planning/weekly_plans/week8-module3-segmentation/do4-oct-09/wip/

# Run preparation script
python prepare_images.py

# Verify images created
ls -la images/
```

---

## 📥 Alternative: Download Real Images

### Recommended Sources

1. **Kaggle Datasets**
   - Document Image Processing: https://www.kaggle.com/datasets
   - Coin Detection: https://www.kaggle.com/datasets

2. **Computer Vision Datasets**
   - ICDAR (Document Analysis): https://rrc.cvc.uab.es/
   - COCO Dataset: https://cocodataset.org/

3. **Free Stock Photos**
   - Unsplash: https://unsplash.com/
   - Pexels: https://www.pexels.com/
   - Pixabay: https://pixabay.com/

### Download Commands

```bash
# Example: Download from Unsplash
wget "https://unsplash.com/photos/random/?group" -O images/group_photo.jpg

# Example: Download coins image
wget "https://example.com/coins.jpg" -O images/coins.jpg
```

---

## ✅ Image Quality Checklist

Before the tutorial, verify each image:

### Group Photo
- [ ] Clear faces visible
- [ ] Good contrast
- [ ] Resolution ≥ 600×800
- [ ] JPG format

### Coins
- [ ] 5-8 distinct objects
- [ ] Plain background
- [ ] Good separation between coins
- [ ] High contrast

### Noisy Document
- [ ] Visible noise (salt-and-pepper)
- [ ] Text still readable
- [ ] Binary or grayscale
- [ ] ≥ 600×800 resolution

### Broken Text
- [ ] Clear holes visible
- [ ] Text recognizable
- [ ] Binary format
- [ ] Moderate size (300-500px)

### Scanned Document
- [ ] Multiple text regions
- [ ] Some noise present
- [ ] Regions separable
- [ ] ≥ 800×1000 resolution

### Shapes
- [ ] Multiple distinct shapes
- [ ] Clear boundaries
- [ ] Binary format
- [ ] 500×500 or similar

---

## 🎯 Quick Start for Students

### Student Instructions

**Before the tutorial:**

1. **Download the setup script:**
   ```bash
   # Get prepare_images.py from course materials
   wget [course-url]/prepare_images.py
   ```

2. **Run the script:**
   ```bash
   python prepare_images.py
   ```

3. **Verify images:**
   ```bash
   ls images/
   # Should show: group_photo.jpg, coins.jpg, noisy_document.jpg, etc.
   ```

4. **Open Jupyter notebook:**
   ```bash
   jupyter notebook tutorial_roi_morphology.ipynb
   ```

5. **You're ready!** All images are prepared.

---

## 📊 Expected Results

### After Running Tutorial

Your `output/` folder should contain:

```
output/
├── roi_extracted.jpg          # From Exercise 1.1
├── coin_0.jpg                 # From Exercise 1.2
├── coin_1.jpg
├── coin_2.jpg
├── coin_3.jpg
├── coin_4.jpg
├── cleaned_document.jpg       # From Exercise 2.1
├── repaired_text.jpg          # From Exercise 2.2
├── morphological_edges.jpg    # From Exercise 2.3
├── text_region_0.jpg          # From Integration
├── text_region_1.jpg
├── text_region_2.jpg
└── text_region_3.jpg
```

**Total expected output files: 15-20 images**

---

## 🔍 Troubleshooting Images

### Issue: Images too large (slow processing)

```python
# Resize images before processing
img = cv2.imread('large_image.jpg')
resized = cv2.resize(img, (800, 600))
cv2.imwrite('images/resized_image.jpg', resized)
```

### Issue: Images have wrong format

```python
# Convert to required format
img = cv2.imread('wrong_format.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('images/correct_format.jpg', gray)
```

### Issue: Poor contrast

```python
# Increase contrast
img = cv2.imread('low_contrast.jpg', 0)
enhanced = cv2.equalizeHist(img)
cv2.imwrite('images/enhanced.jpg', enhanced)
```

---

## 💾 Backup Strategy

### For Instructor

Keep a **backup set** of all images:

```bash
# Create backup
mkdir backup_images
cp images/* backup_images/

# If student images fail, restore backup
cp backup_images/* images/
```

### Cloud Storage

Upload to Google Drive/Dropbox:
```bash
# Share link with students
https://drive.google.com/tutorial-images.zip
```

---

## 📝 Summary

### What You Need

**Minimum (using synthetic images):**
- ✅ Run `prepare_images.py` script
- ✅ All images auto-generated
- ✅ Ready in 30 seconds

**Recommended (using real images):**
- ✅ Download from suggested sources
- ✅ Better learning experience
- ✅ Real-world examples

**Either way works!** The tutorial is designed to work with both synthetic and real images.

---

**🎓 You're All Set!**

*All materials prepared for Week 8, Day 4 tutorial session*

---

**Course**: 21CSE558T - Deep Neural Network Architectures
**Module 3**: Image Processing & Deep Neural Networks
**Week 8, Day 4** - October 9, 2025
