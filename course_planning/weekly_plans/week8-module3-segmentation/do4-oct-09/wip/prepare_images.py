#!/usr/bin/env python3
"""
Image Preparation Script for Week 8, Day 4 Tutorial
Course: Deep Neural Network Architectures (21CSE558T)
Module 3: Image Processing & DNNs

This script creates all necessary sample images for the ROI & Morphology tutorial.
Run this before starting the tutorial to prepare all images.

Usage:
    python prepare_images.py
"""

import cv2
import numpy as np
import os
import sys

def create_directories():
    """Create necessary directories"""
    os.makedirs('images', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    print("✓ Created directories: images/ and output/")

def create_group_photo():
    """Create synthetic group photo for ROI extraction"""
    print("  Creating group_photo.jpg...", end=' ')

    img = np.zeros((600, 800, 3), dtype=np.uint8)

    # Background gradient (with clipping to prevent overflow)
    for i in range(600):
        b = min(200 + i//10, 255)
        g = max(220 - i//20, 0)
        r = max(240 - i//30, 0)
        img[i, :] = (b, g, r)

    # Face 1 (left)
    cv2.rectangle(img, (100, 150), (250, 400), (255, 220, 180), -1)
    cv2.ellipse(img, (175, 275), (60, 80), 0, 0, 360, (255, 210, 170), -1)
    cv2.circle(img, (150, 250), 15, (50, 50, 50), -1)  # Left eye
    cv2.circle(img, (200, 250), 15, (50, 50, 50), -1)  # Right eye
    cv2.ellipse(img, (175, 320), (25, 15), 0, 0, 180, (150, 50, 50), 3)  # Smile

    # Face 2 (center)
    cv2.rectangle(img, (350, 100), (500, 350), (255, 200, 160), -1)
    cv2.ellipse(img, (425, 225), (60, 80), 0, 0, 360, (255, 190, 150), -1)
    cv2.circle(img, (400, 200), 15, (30, 30, 30), -1)  # Left eye
    cv2.circle(img, (450, 200), 15, (30, 30, 30), -1)  # Right eye
    cv2.ellipse(img, (425, 270), (20, 12), 0, 0, 180, (120, 40, 40), 3)  # Smile

    # Face 3 (right)
    cv2.rectangle(img, (550, 200), (700, 450), (255, 210, 170), -1)
    cv2.ellipse(img, (625, 325), (60, 80), 0, 0, 360, (255, 200, 160), -1)
    cv2.circle(img, (600, 300), 15, (40, 40, 40), -1)  # Left eye
    cv2.circle(img, (650, 300), 15, (40, 40, 40), -1)  # Right eye
    cv2.ellipse(img, (625, 370), (22, 13), 0, 0, 180, (130, 45, 45), 3)  # Smile

    cv2.imwrite('images/group_photo.jpg', img)
    print("✓")

def create_coins():
    """Create synthetic coins image for contour-based extraction"""
    print("  Creating coins.jpg...", end=' ')

    coins = np.zeros((400, 600), dtype=np.uint8)

    # Main coins (different sizes)
    cv2.circle(coins, (100, 100), 40, 255, -1)
    cv2.circle(coins, (250, 150), 50, 255, -1)
    cv2.circle(coins, (450, 120), 45, 255, -1)
    cv2.circle(coins, (200, 300), 35, 255, -1)
    cv2.circle(coins, (400, 300), 55, 255, -1)

    # Small noise dot (to demonstrate filtering)
    cv2.circle(coins, (50, 350), 5, 255, -1)

    cv2.imwrite('images/coins.jpg', coins)
    print("✓")

def create_noisy_document():
    """Create noisy document for morphological opening demo"""
    print("  Creating noisy_document.jpg...", end=' ')

    doc = np.zeros((600, 800), dtype=np.uint8)

    # Text blocks
    cv2.rectangle(doc, (50, 100), (350, 150), 255, -1)
    cv2.rectangle(doc, (50, 200), (400, 250), 255, -1)
    cv2.rectangle(doc, (50, 300), (300, 350), 255, -1)
    cv2.rectangle(doc, (50, 400), (450, 450), 255, -1)

    # Add salt-and-pepper noise
    np.random.seed(42)  # For reproducibility
    noise = np.random.randint(0, 256, doc.shape).astype(np.uint8)
    noise = (noise > 245).astype(np.uint8) * 255  # 4% noise
    noisy = cv2.bitwise_or(doc, noise)

    cv2.imwrite('images/noisy_document.jpg', noisy)
    print("✓")

def create_broken_text():
    """Create text with holes for morphological closing demo"""
    print("  Creating broken_text.jpg...", end=' ')

    text = np.zeros((300, 400), dtype=np.uint8)

    # Text blocks (simulate letters)
    cv2.rectangle(text, (50, 100), (120, 200), 255, -1)
    cv2.rectangle(text, (140, 100), (210, 200), 255, -1)
    cv2.rectangle(text, (230, 100), (300, 200), 255, -1)

    # Add holes (simulate faded/broken text)
    cv2.circle(text, (85, 130), 12, 0, -1)
    cv2.circle(text, (85, 170), 10, 0, -1)
    cv2.circle(text, (175, 140), 10, 0, -1)
    cv2.circle(text, (175, 160), 8, 0, -1)
    cv2.circle(text, (265, 135), 14, 0, -1)
    cv2.circle(text, (265, 165), 11, 0, -1)

    cv2.imwrite('images/broken_text.jpg', text)
    print("✓")

def create_scanned_document():
    """Create scanned document for integration pipeline"""
    print("  Creating scanned_doc.jpg...", end=' ')

    doc = np.ones((800, 1000), dtype=np.uint8) * 240  # Light gray background

    # Text blocks (dark regions)
    cv2.rectangle(doc, (50, 50), (400, 150), 20, -1)
    cv2.rectangle(doc, (450, 50), (950, 150), 20, -1)
    cv2.rectangle(doc, (50, 200), (950, 300), 20, -1)
    cv2.rectangle(doc, (50, 350), (600, 450), 20, -1)

    # Add realistic scanning noise
    np.random.seed(42)
    noise = np.random.randint(-20, 20, doc.shape).astype(np.int16)
    scanned = np.clip(doc + noise, 0, 255).astype(np.uint8)

    # Add some small holes (simulate old document)
    for _ in range(5):
        x, y = np.random.randint(100, 900), np.random.randint(100, 700)
        cv2.circle(scanned, (x, y), 5, 240, -1)

    cv2.imwrite('images/scanned_doc.jpg', scanned)
    print("✓")

def create_shapes():
    """Create shapes image for morphological gradient demo"""
    print("  Creating shapes.jpg...", end=' ')

    shapes = np.zeros((500, 500), dtype=np.uint8)

    # Various geometric shapes
    cv2.rectangle(shapes, (50, 50), (150, 150), 255, -1)  # Square
    cv2.circle(shapes, (350, 100), 60, 255, -1)  # Circle
    cv2.ellipse(shapes, (250, 350), (80, 50), 0, 0, 360, 255, -1)  # Ellipse

    # Triangle
    pts = np.array([[100, 400], [150, 300], [200, 400]], np.int32)
    cv2.fillPoly(shapes, [pts], 255)

    # Pentagon
    pts2 = np.array([[400, 400], [450, 350], [430, 290], [370, 290], [350, 350]], np.int32)
    cv2.fillPoly(shapes, [pts2], 255)

    cv2.imwrite('images/shapes.jpg', shapes)
    print("✓")

def verify_images():
    """Verify all images were created successfully"""
    print("\nVerifying images...")

    required_images = [
        'group_photo.jpg',
        'coins.jpg',
        'noisy_document.jpg',
        'broken_text.jpg',
        'scanned_doc.jpg',
        'shapes.jpg'
    ]

    all_exist = True
    for img_name in required_images:
        img_path = os.path.join('images', img_name)
        if os.path.exists(img_path):
            size = os.path.getsize(img_path)
            print(f"  ✓ {img_name:25s} ({size:,} bytes)")
        else:
            print(f"  ✗ {img_name:25s} (MISSING)")
            all_exist = False

    return all_exist

def print_summary():
    """Print summary and next steps"""
    print("\n" + "="*60)
    print("✅ All sample images created successfully!")
    print("="*60)
    print("\n📁 Directory Structure:")
    print("  ├── images/               (6 input images)")
    print("  │   ├── group_photo.jpg")
    print("  │   ├── coins.jpg")
    print("  │   ├── noisy_document.jpg")
    print("  │   ├── broken_text.jpg")
    print("  │   ├── scanned_doc.jpg")
    print("  │   └── shapes.jpg")
    print("  └── output/               (results will be saved here)")
    print("\n🚀 Next Steps:")
    print("  1. Open Jupyter notebook:")
    print("     jupyter notebook tutorial_roi_morphology.ipynb")
    print("\n  2. Or run Python scripts directly:")
    print("     python exercise_1_1.py")
    print("\n  3. Check output/ folder for results")
    print("\n📚 Additional Resources:")
    print("  - Lecture notes: comprehensive_lecture_notes.md")
    print("  - Cheat sheet: quick_reference_cheat_sheet.md")
    print("  - Image guide: sample_images_guide.md")
    print("\n" + "="*60)

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("  Image Preparation Script for Week 8, Day 4 Tutorial")
    print("  Course: Deep Neural Network Architectures (21CSE558T)")
    print("="*60)
    print("\nGenerating sample images...")
    print("-" * 60)

    try:
        # Create directories
        create_directories()
        print()

        # Generate all images
        create_group_photo()
        create_coins()
        create_noisy_document()
        create_broken_text()
        create_scanned_document()
        create_shapes()

        # Verify creation
        if verify_images():
            print_summary()
            return 0
        else:
            print("\n❌ Error: Some images were not created successfully.")
            print("Please check the error messages above.")
            return 1

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
