# Character Naming and Scientific Terminology Rules

## Critical Rule for Lecture Note Creation

### CHARACTER NAMING CONVENTION

**MANDATORY PREFIX RULE:**
- All fictional teaching characters MUST be prefixed with "**Character:**"
- Never use plain names like "Giovanni" or "Sarah" alone
- Always write "**Character: Dr. Priya**" or "**Character: Arjun**"

**REASON:**
- Prevents student confusion between fictional teaching characters and actual scientists/researchers
- Clear distinction between pedagogical analogies and historical facts
- Students studying for exams won't confuse story characters with real people

### CHARACTER NAME GUIDELINES

**Use Indian Names Only:**
- Characters should have Indian names for cultural relevance and student relatability
- Examples: Priya, Arjun, Meera, Rajesh, Kavya, Aditya, Sneha, Rohan, etc.
- Avoid Western names (Giovanni, Sarah, Sophia, Maria, etc.)

**First Introduction Format:**
```markdown
**Meet Character: Dr. Priya - Cardiologist Extraordinaire**

Character: Dr. Priya monitors heart rhythms...
```

**Subsequent References:**
- Use full form: "**Character: Dr. Priya**" (preferred)
- Or shorter: "**Character: Priya**" (acceptable)
- NEVER just "Dr. Priya" or "Priya" alone

**In Code Comments:**
```python
# Character: Dr. Priya's setup
signal = np.array([1, 1, 2, 1, 3, 2, 1, 2])
```

---

## SCIENTIFIC TERMINOLOGY RULES

### REAL SCIENTISTS - Always Highlight

**Format:** Use full names with context on first mention

**Examples:**
```markdown
✅ CORRECT:
- **David Hubel and Torsten Wiesel (1959)** - Nobel Prize winners for visual cortex research
- **Yann LeCun** - Pioneer of Convolutional Neural Networks
- **Geoffrey Hinton** - "Godfather of Deep Learning"
- **Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton** - Created AlexNet (2012)

❌ WRONG:
- Hubel discovered... (too casual)
- LeCun invented CNNs (no context)
```

**Purpose:**
- Students recognize these are REAL historical figures
- Proper attribution for scientific contributions
- Exam preparation - students need to know key researchers

---

### THEORY/ALGORITHM NAMES - Always Point Out

**Format:** Bold or clearly marked as technical terms

**Examples:**
```markdown
✅ CORRECT:
- **AlexNet** architecture (2012 ImageNet breakthrough)
- **Backpropagation** algorithm
- **ReLU** (Rectified Linear Unit) activation function
- **ImageNet** dataset challenge
- **Batch Normalization** technique
- **LeNet-5** (Yann LeCun, 1998)

❌ WRONG:
- alexnet architecture (lowercase, no context)
- The relu function (not marked as technical term)
```

**Purpose:**
- Students recognize these are established technical terms
- Proper capitalization for architecture/algorithm names
- Clear separation from general concepts

---

### ARCHITECTURE NAMES - Specific Formatting

**Well-Known Architectures:**
```markdown
- **LeNet-5** (1998) - Yann LeCun
- **AlexNet** (2012) - Krizhevsky et al.
- **VGGNet** (2014) - Visual Geometry Group, Oxford
- **ResNet** (2015) - Microsoft Research
- **InceptionNet / GoogLeNet** (2014) - Google
- **MobileNet** (2017) - Google
- **EfficientNet** (2019) - Google
```

**Always Include:**
1. Architecture name in bold
2. Year introduced
3. Research team/institution (optional but recommended)

---

## COMPLETE EXAMPLE - CORRECT FORMAT

### ❌ WRONG (Old Style):
```markdown
## The Photographer Example

Meet Giovanni, a photographer who needs to detect edges.

Giovanni uses a 3×3 filter to scan his images.

Hubel discovered visual cortex cells in 1959.
Later, LeCun created AlexNet using CNNs.
```

### ✅ CORRECT (New Style):
```markdown
## The Photographer Example

**Meet Character: Arjun - Master Photographer**

**Character: Arjun** needs to detect edges in photographs.

**Character: Arjun** uses a 3×3 filter to scan his images.

**Scientific Background:**
**David Hubel and Torsten Wiesel (1959)** discovered specialized neurons in the visual cortex that respond to specific orientations - this biological insight inspired CNNs.

Later, **Yann LeCun** pioneered Convolutional Neural Networks with **LeNet-5 (1998)**. The breakthrough came with **AlexNet (2012)** by **Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton**, which won the ImageNet challenge.
```

---

## QUICK REFERENCE CHECKLIST

### When Writing Lecture Notes:

**Characters (Fictional):**
- [ ] Prefixed with "**Character:**"
- [ ] Uses Indian name (Priya, Arjun, Meera, etc.)
- [ ] Clearly a teaching analogy
- [ ] Consistent throughout document

**Scientists (Real):**
- [ ] Full name on first mention
- [ ] Year/context provided
- [ ] Bolded or highlighted
- [ ] Proper attribution for discoveries

**Theories/Algorithms:**
- [ ] Proper capitalization (**ReLU**, not "relu")
- [ ] Bolded as technical terms
- [ ] Explained on first use
- [ ] Context provided when relevant

**Architectures:**
- [ ] Standard name format (**AlexNet**, **VGGNet**)
- [ ] Year included
- [ ] Research team credited (optional)

---

## RATIONALE

### Why This Matters for Students:

1. **Exam Preparation:**
   - Students need to cite real scientists in essays
   - Confusing "Character: Arjun" with "Yann LeCun" would be catastrophic
   - Clear distinction helps study notes

2. **Professional Communication:**
   - Learning proper attribution early
   - Understanding difference between pedagogical tools and scientific facts
   - Preparing for research paper reading

3. **Cultural Relevance:**
   - Indian names increase engagement
   - Students relate better to familiar names
   - Reduces cognitive load (not learning foreign names while learning CNNs)

4. **Clarity:**
   - "Character:" prefix is unmistakable
   - No ambiguity in any context
   - Respects actual scientists by not mixing with fiction

---

## EXAMPLES FROM WEEK 10 LECTURE NOTES

### Character Usage:
- **Character: Dr. Priya** - Cardiologist explaining 1D convolution
- **Character: Arjun** - Photographer explaining 2D convolution  
- **Character: Meera** - Factory Manager explaining CNN pipeline
- **Character: Dr. Rajesh** - Radiologist explaining 3D convolution
- **Character: Detective Kavya** - Postal detective from Week 9

### Scientist References:
- **David Hubel and Torsten Wiesel (1959)** - Visual cortex discoveries
- **Yann LeCun** - CNN pioneer
- **Geoffrey Hinton** - Deep learning pioneer
- **Alex Krizhevsky** - AlexNet creator

### Architecture/Algorithm Names:
- **LeNet-5 (1998)**
- **AlexNet (2012)**
- **VGGNet (2014)**
- **ResNet (2015)**
- **ReLU** activation
- **Backpropagation**
- **Batch Normalization**

---

## ENFORCEMENT

**This rule applies to:**
- All comprehensive lecture notes
- Tutorial materials
- Homework assignments
- Assessment questions
- Slide presentations
- Code examples and comments

**No exceptions** - maintain consistency across all teaching materials.

---

**Created:** October 2025
**Course:** 21CSE558T - Deep Neural Network Architectures
**Purpose:** Ensure clear distinction between pedagogical characters and scientific terminology
**Status:** MANDATORY RULE for all future content creation
