 ple Quick Reference Guide: ROI & Morphology
## Week 8, Day 4 - Tutorial Cheat Sheet

**Course**: 21CSE558T | **Date**: Oct 9, 2025 | **Duration**: 1 Hour

---

## 📦 Essential Imports

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

---

## 🎯 Part 1: ROI Extraction

### Method 1: Rectangular ROI (Array Slicing)

**Formula:**
```python
roi = img[y:y+h, x:x+w]
# Note: [rows, columns] = [y, x]
```

**Complete Example:**
```python
# Load image
img = cv2.imread('image.jpg')

# Define ROI coordinates
x, y, w, h = 100, 50, 200, 150

# Extract ROI
roi = img[y:y+h, x:x+w]

# Save
cv2.imwrite('roi.jpg', roi)
```

**⚠️ Common Mistake:**
- ❌ `img[x:x+w, y:y+h]` - WRONG ORDER
- ✅ `img[y:y+h, x:x+w]` - CORRECT (rows first!)

---

### Method 2: Contour-Based ROI

**Pipeline:**
```python
1. Grayscale → 2. Threshold → 3. Find Contours → 4. Extract ROI
```

**Complete Code:**
```python
# 1. Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Threshold
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 3. Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

# 4. Extract each ROI
for i, contour in enumerate(contours):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(contour)

    # Extract ROI
    roi = img[y:y+h, x:x+w]

    # Save
    cv2.imwrite(f'object_{i}.jpg', roi)
```

**🔍 Filter by Area (Remove Noise):**
```python
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 500:  # Minimum area threshold
        # Process this contour
        x, y, w, h = cv2.boundingRect(contour)
        roi = img[y:y+h, x:x+w]
```

---

## 🧹 Part 2: Morphological Operations

### Kernel (Structuring Element)

```python
# Create kernel
kernel = np.ones((5, 5), np.uint8)

# Common sizes: (3,3), (5,5), (7,7), (9,9)
```

---

### Operation 1: Erosion (Shrink)

**Purpose:** Remove small objects, separate touching objects

```python
eroded = cv2.erode(binary_img, kernel, iterations=1)
```

**Effect:**
- ✓ Removes noise
- ✗ Shrinks all objects

---

### Operation 2: Dilation (Expand)

**Purpose:** Fill holes, connect nearby objects

```python
dilated = cv2.dilate(binary_img, kernel, iterations=1)
```

**Effect:**
- ✓ Fills small holes
- ✗ Expands all objects (including noise)

---

### Operation 3: Opening (Erode → Dilate)

**Purpose:** Remove noise while preserving object size

```python
opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
```

**Use When:**
- Salt-and-pepper noise
- Small unwanted objects
- Need to preserve main object size

**Effect:**
- ✓ Removes small noise
- ✓ Preserves large objects
- ✓ Restores original size

---

### Operation 4: Closing (Dilate → Erode)

**Purpose:** Fill holes while preserving boundaries

```python
closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
```

**Use When:**
- Holes inside objects
- Broken text/lines
- Disconnected parts that should connect

**Effect:**
- ✓ Fills small holes
- ✓ Connects nearby components
- ✓ Restores original boundaries

---

### Operation 5: Morphological Gradient (Edges)

**Purpose:** Extract object boundaries

```python
gradient = cv2.morphologyEx(binary_img, cv2.MORPH_GRADIENT, kernel)
```

**Formula:** `Gradient = Dilation - Erosion`

**Use When:**
- Need edges from binary image
- Alternative to Canny for binary images

---

## 🔄 Quick Decision Guide

### When to Use Which Operation?

| Problem | Solution | Code |
|---------|----------|------|
| 🔴 Noisy dots on image | Opening | `cv2.MORPH_OPEN` |
| 🔴 Holes in objects | Closing | `cv2.MORPH_CLOSE` |
| 🔴 Objects too thick | Erosion | `cv2.erode()` |
| 🔴 Objects too thin | Dilation | `cv2.dilate()` |
| 🔴 Need edges only | Gradient | `cv2.MORPH_GRADIENT` |

---

## 🚀 Complete Pipeline Template

### Document Processing Pipeline

```python
import cv2
import numpy as np

def process_document(image_path):
    # 1. Load and convert
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # 3. Remove noise (Opening)
    kernel1 = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel1)

    # 4. Fill holes (Closing)
    kernel2 = np.ones((7, 7), np.uint8)
    filled = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel2)

    # 5. Find text regions
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # 6. Extract ROIs
    regions = []
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            roi = img[y:y+h, x:x+w]
            cv2.imwrite(f'region_{i}.jpg', roi)
            regions.append(roi)

    return regions

# Run
regions = process_document('document.jpg')
print(f"Extracted {len(regions)} regions")
```

---

## 🎨 Visualization Helper

### Display in Jupyter Notebook

```python
import matplotlib.pyplot as plt

def show_images(images, titles, cols=3):
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

    for i, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 3:
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Usage
show_images([img1, img2, img3], ['Original', 'Processed', 'Result'])
```

---

## ⚙️ Common Parameters

### Threshold Types
```python
cv2.THRESH_BINARY        # val > thresh → 255, else 0
cv2.THRESH_BINARY_INV    # val > thresh → 0, else 255
cv2.THRESH_OTSU          # Automatic threshold
```

### Contour Retrieval Modes
```python
cv2.RETR_EXTERNAL        # Only outermost contours
cv2.RETR_LIST            # All contours, no hierarchy
cv2.RETR_TREE            # Full hierarchy
```

### Morphology Operations
```python
cv2.MORPH_ERODE          # Erosion
cv2.MORPH_DILATE         # Dilation
cv2.MORPH_OPEN           # Opening (erode → dilate)
cv2.MORPH_CLOSE          # Closing (dilate → erode)
cv2.MORPH_GRADIENT       # Gradient (dilate - erode)
cv2.MORPH_TOPHAT         # Original - Opening
cv2.MORPH_BLACKHAT       # Closing - Original
```

---

## 🐛 Troubleshooting

### Issue: ROI is blank/black
```python
# Check coordinates are within bounds
print(f"Image shape: {img.shape}")
print(f"ROI: x={x}, y={y}, w={w}, h={h}")
print(f"Max x+w: {x+w}, should be < {img.shape[1]}")
print(f"Max y+h: {y+h}, should be < {img.shape[0]}")
```

### Issue: No contours found
```python
# Visualize binary image first
cv2.imshow('Binary', binary)
cv2.waitKey(0)

# Check threshold value
print(f"Unique values: {np.unique(binary)}")
```

### Issue: Too many contours (noise)
```python
# Filter by area
for contour in contours:
    area = cv2.contourArea(contour)
    if area > MIN_AREA:  # Adjust MIN_AREA
        # Process
```

### Issue: Morphology too aggressive
```python
# Try smaller kernel
kernel = np.ones((3, 3), np.uint8)  # Instead of (7, 7)

# Or reduce iterations
result = cv2.erode(img, kernel, iterations=1)  # Instead of 2
```

---

## 📊 Kernel Size Effects

| Kernel Size | Effect | Best For |
|-------------|--------|----------|
| 3×3 | Mild | Fine details, small noise |
| 5×5 | Moderate | General purpose |
| 7×7 | Strong | Large noise, thick objects |
| 9×9+ | Aggressive | Very noisy images |

**Rule of Thumb:** Start small (3×3), increase if needed

---

## 🎯 Quick Tips

### ✅ DO:
- Always check image shape before slicing
- Filter contours by area to remove noise
- Save intermediate results for debugging
- Use opening for noise, closing for holes
- Start with small kernels (3×3, 5×5)

### ❌ DON'T:
- Mix up row/column order in slicing
- Apply morphology on grayscale (use binary)
- Use huge kernels without testing
- Extract ROI from binary (use original image)
- Forget to check if contours list is empty

---

## 🔗 Useful OpenCV Functions

### Image Operations
```python
cv2.imread(path)                    # Load image
cv2.imwrite(path, img)              # Save image
cv2.cvtColor(img, code)             # Color conversion
cv2.threshold(img, thresh, max, type) # Thresholding
```

### Contour Operations
```python
cv2.findContours(img, mode, method) # Find contours
cv2.drawContours(img, contours, -1, color, thick) # Draw
cv2.boundingRect(contour)           # Get bounding box
cv2.contourArea(contour)            # Calculate area
```

### Morphological Operations
```python
cv2.erode(img, kernel, iterations)  # Erosion
cv2.dilate(img, kernel, iterations) # Dilation
cv2.morphologyEx(img, op, kernel)   # Compound ops
```

---

## 📝 Exercise Checklist

### Part 1: ROI Extraction
- [ ] Extract rectangular ROI from coordinates
- [ ] Extract multiple objects using contours
- [ ] Filter contours by area (remove noise)
- [ ] Save individual ROIs as separate files

### Part 2: Morphology
- [ ] Apply opening to remove noise
- [ ] Apply closing to fill holes
- [ ] Compare different kernel sizes
- [ ] Extract edges using gradient

### Part 3: Integration
- [ ] Build complete pipeline
- [ ] Process noisy document
- [ ] Extract clean text regions
- [ ] Submit code and outputs

---

## 🚨 Quick Formulas

### ROI Extraction
```python
roi = img[y:y+h, x:x+w]  # Remember: rows first!
```

### Morphology
```python
# Opening = Erode then Dilate (remove noise)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# Closing = Dilate then Erode (fill holes)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# Gradient = Dilate - Erode (edges)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
```

### Pipeline Order
```
Image → Grayscale → Threshold → Opening → Closing → Contours → ROI
```

---

## 💡 Remember

### ROI Extraction:
- **Rectangular**: When you know exact coordinates
- **Contour-based**: When objects need automatic detection

### Morphology:
- **Opening**: Removes (noise, small objects)
- **Closing**: Fills (holes, gaps, breaks)

### Always:
1. Visualize intermediate steps
2. Filter by area before extracting ROI
3. Save all outputs for verification
4. Test with small kernels first

---

## 📚 Resources

### Documentation
- OpenCV Docs: https://docs.opencv.org/4.x/
- ROI Tutorial: `/tutorial_roi_morphology.ipynb`
- Lecture Notes: `/comprehensive_lecture_notes.md`

### Key Textbook Pages
- Manaswi Ch. 5: Pages 125-145 (ROI & Morphology)
- Week 8 Day 3: Segmentation concepts

---

**🎓 Good Luck with the Tutorial!**

*Keep this cheat sheet handy during the 1-hour session*

---

**Course**: 21CSE558T - Deep Neural Network Architectures
**Module 3**: Image Processing & Deep Neural Networks
**Week 8, Day 4** - October 9, 2025
