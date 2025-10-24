# Unit Test 2 Schedule - Modules 3 & 4

**Date:** October 31, 2025 (Week 12)  
**Duration:** 90 minutes  
**Coverage:** Module 3 (Image Processing & Deep Networks) + Module 4 (CNNs & Transfer Learning)  
**Weight:** 22.5% of total course grade (45% of CLA-1)

---

## 📋 Test Overview

### Scope and Coverage
**Module 3: Image Processing & Deep Networks (45% of test)**
- Digital image representation and manipulation
- Feature extraction and preprocessing techniques
- Deep neural networks for image classification
- Image enhancement and noise reduction
- OpenCV integration with deep learning

**Module 4: CNNs & Transfer Learning (55% of test)**
- Convolution operation and pooling layers
- CNN architectures (LeNet, AlexNet, VGG, ResNet)
- Transfer learning principles and implementation
- Fine-tuning strategies and best practices
- Modern CNN innovations and applications

---

## 📊 Question Distribution

### Theory Section (50 points - 55 minutes)
**Question Types:**
- **Multiple Choice (20 points):** 20 questions × 1 point each
- **Short Answer (15 points):** 5 questions × 3 points each  
- **Analytical Problems (15 points):** 3 questions × 5 points each

**Content Distribution:**
- Module 3 concepts: 22 points
- Module 4 concepts: 23 points
- Integration/application: 5 points

### Practical Section (50 points - 35 minutes)
**Coding Problems:**
- **CNN Implementation (25 points):** Build and train a CNN for image classification
- **Transfer Learning Application (15 points):** Fine-tune a pre-trained model
- **Image Processing Pipeline (10 points):** Implement preprocessing and augmentation

---

## 📚 Study Guide and Topics

### Module 3 - Key Topics for Review

#### Digital Image Processing
- [ ] Image representation: pixels, channels, color spaces
- [ ] Image formats and their implications for ML
- [ ] Histogram analysis and equalization
- [ ] Spatial and frequency domain operations

#### Feature Extraction
- [ ] Traditional methods: edges, corners, textures
- [ ] Haar features and cascade classifiers
- [ ] SIFT/SURF concepts (high-level understanding)
- [ ] Deep learning automatic feature extraction

#### OpenCV Integration
- [ ] Image loading, manipulation, and saving
- [ ] Basic image processing operations
- [ ] Integration with TensorFlow/Keras pipelines
- [ ] Real-time image processing considerations

#### Deep Networks for Images
- [ ] Challenges of high-dimensional image data
- [ ] Flattening vs. preserving spatial structure
- [ ] Data augmentation techniques and rationale
- [ ] Preprocessing pipelines for different image types

#### Image Enhancement Applications
- [ ] Noise reduction using autoencoders
- [ ] Super-resolution concepts
- [ ] Style transfer (conceptual understanding)
- [ ] Generative models for image synthesis

### Module 4 - Key Topics for Review

#### Convolution Fundamentals
- [ ] Convolution operation: mathematical definition
- [ ] Kernels/filters: concept and examples
- [ ] Padding strategies: valid, same, full
- [ ] Stride and its effect on output dimensions
- [ ] Parameter sharing and translation invariance

#### CNN Architecture Components
- [ ] Convolution layers: parameters and computations
- [ ] Pooling layers: max, average, global pooling
- [ ] Fully connected layers in CNN context
- [ ] Activation functions specific to CNNs
- [ ] Batch normalization in convolutional networks

#### Classic CNN Architectures
- [ ] LeNet: pioneering CNN design
- [ ] AlexNet: breakthrough and innovations
- [ ] VGG: depth and uniform architecture
- [ ] ResNet: residual connections and skip connections
- [ ] Inception: multi-scale feature extraction

#### Transfer Learning
- [ ] Pre-trained models and ImageNet
- [ ] Feature extraction vs. fine-tuning
- [ ] Layer freezing strategies
- [ ] Domain adaptation considerations
- [ ] When transfer learning is effective

#### Modern CNN Innovations
- [ ] Depthwise separable convolutions
- [ ] Squeeze-and-excitation blocks
- [ ] Attention mechanisms in CNNs
- [ ] EfficientNet scaling principles
- [ ] Vision Transformers (conceptual)

---

## 💻 Practical Preparation

### Coding Skills Assessment
Students should be prepared to:
- [ ] Implement CNN architectures using Keras layers
- [ ] Configure convolution and pooling parameters appropriately
- [ ] Load and fine-tune pre-trained models
- [ ] Implement image preprocessing and augmentation
- [ ] Debug CNN training issues and interpret results

### Sample Code Scenarios

**CNN Architecture:**
```python
# Be prepared to complete or analyze similar architectures
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=?, kernel_size=?, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=?),
    tf.keras.layers.Conv2D(filters=?, kernel_size=?, activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(units=?, activation='softmax')
])
```

**Transfer Learning Setup:**
```python
# Understand pre-trained model usage
base_model = tf.keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = ?  # Understand when to freeze/unfreeze

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(?, activation='softmax')
])
```

**Image Preprocessing:**
```python
# Data augmentation and preprocessing
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=?,
    width_shift_range=?,
    height_shift_range=?,
    horizontal_flip=?,
    rescale=?
)
```

---

## 📈 Assessment Criteria

### Theory Section Grading
**Multiple Choice (20 points):**
- Understanding of CNN components and operations
- Knowledge of architecture design principles
- Transfer learning concepts and applications

**Short Answer (15 points):**
- Clear explanations of convolution operations
- Accurate descriptions of transfer learning strategies
- Proper reasoning for architecture choices

**Analytical Problems (15 points):**
- Calculation of CNN output dimensions
- Analysis of parameter counts and computational complexity
- Design decisions for specific image processing tasks

### Practical Section Grading
**CNN Implementation (25 points):**
- Correct layer configuration and parameters
- Appropriate architecture for given problem
- Proper compilation and training setup

**Transfer Learning (15 points):**
- Correct pre-trained model selection and setup
- Appropriate fine-tuning strategy
- Understanding of layer freezing/unfreezing

**Image Processing (10 points):**
- Effective preprocessing pipeline
- Appropriate data augmentation choices
- Integration with model training

---

## 🎯 Problem-Solving Focus Areas

### CNN Architecture Design
**Key Considerations:**
- Input image dimensions and channel count
- Appropriate filter sizes and numbers
- Pooling strategy for dimension reduction
- Final layer configuration for classification

**Common Pitfalls:**
- Dimension mismatches between layers
- Over-parameterization leading to overfitting
- Inadequate regularization in deep networks
- Poor choice of activation functions

### Transfer Learning Strategy
**Decision Framework:**
- Dataset size and similarity to ImageNet
- Computational resources and time constraints
- Target domain characteristics
- Performance requirements

**Implementation Details:**
- Which layers to freeze/unfreeze
- Learning rate adjustment for different layers
- Data preprocessing compatibility
- Fine-tuning schedule and monitoring

---

## 🕐 Test Day Schedule

### Before the Test
- **1 Week Prior:** Comprehensive review sessions with CNN architectures
- **3 Days Prior:** Transfer learning workshop and practice problems
- **1 Day Prior:** Final Q&A and technical setup verification

### Test Day Timeline
- **Arrival:** 15 minutes before start time
- **Setup:** 10 minutes for environment verification
- **Theory Section:** 55 minutes (50 points)
- **Break:** 5 minutes (optional)
- **Practical Section:** 35 minutes (50 points)
- **Buffer:** 5 minutes for submission and technical issues

### Technical Requirements
- **GPU Access:** CUDA-compatible environment or cloud resources
- **Memory:** Sufficient RAM for loading pre-trained models
- **Storage:** Space for dataset and model checkpoints
- **Libraries:** TensorFlow 2.x, OpenCV, matplotlib

---

## 🔍 Advanced Preparation Topics

### CNN Mathematics
- [ ] **Convolution Output Size:** Formula and calculation
- [ ] **Parameter Counting:** Weights and biases in each layer
- [ ] **Receptive Field:** Understanding and calculation
- [ ] **Computational Complexity:** FLOPs and memory requirements

### Architecture Analysis
- [ ] **Design Rationale:** Why specific architectures work
- [ ] **Trade-offs:** Accuracy vs. speed vs. memory
- [ ] **Modern Trends:** Efficiency and mobile deployment
- [ ] **Ablation Studies:** Understanding component contributions

### Transfer Learning Depth
- [ ] **Domain Similarity:** Measuring and implications
- [ ] **Feature Hierarchy:** What different layers capture
- [ ] **Catastrophic Forgetting:** Prevention strategies
- [ ] **Multi-task Learning:** Shared representations

---

## 🛠️ Hands-on Practice Recommendations

### Essential Exercises
1. **CNN from Scratch:** Implement LeNet on CIFAR-10
2. **Architecture Comparison:** VGG vs. ResNet performance analysis
3. **Transfer Learning:** Fine-tune VGG16 on custom dataset
4. **Data Augmentation:** Impact analysis on model performance
5. **Visualization:** Feature maps and learned filters

### Advanced Challenges
1. **Custom Architecture:** Design CNN for specific image task
2. **Multi-class Classification:** Handle imbalanced datasets
3. **Object Localization:** Extend classification to localization
4. **Cross-domain Transfer:** Transfer between different image types
5. **Model Optimization:** Reduce parameters while maintaining accuracy

---

## 📋 Administrative and Support Information

### Study Resources
- **Lab Access:** Extended hours during test week
- **Computing Resources:** GPU clusters for intensive training
- **Reference Materials:** Allowed documentation and papers
- **Practice Datasets:** CIFAR-10, Fashion-MNIST, custom datasets

### Support Services
- **Review Sessions:** Architecture walkthrough and debugging
- **Individual Consultation:** One-on-one problem-solving help
- **Technical Support:** Environment setup and troubleshooting
- **Peer Study Groups:** Collaborative learning opportunities

### Assessment Feedback
- **Immediate:** Common mistakes and solution approaches
- **Detailed:** Individual performance analysis within 48 hours
- **Follow-up:** Preparation guidance for final examination
- **Remediation:** Additional practice for challenging concepts

---

## 🚀 Preparation for Final Exam

### Integration with Final Assessment
This unit test serves as preparation for the comprehensive final exam covering all modules. Strong performance here indicates readiness for:
- Object detection algorithms (Module 5)
- Integration of multiple deep learning techniques
- Real-world application development
- Research and innovation in computer vision

### Skill Building Continuation
- Advanced CNN architectures and innovations
- Object detection and segmentation
- Deployment and optimization
- Current research trends and future directions