# Deep Neural Network Architecture Diagrams

**Course:** 21CSE558T - Deep Neural Network Architectures
**For:** M.Tech Students, SRM University
**Week 5, Day 4 - Module 2: Optimization & Regularization**

---

## 1. Basic Deep Neural Network (Multi-Layer Perceptron)

```
     INPUT LAYER    HIDDEN LAYER 1   HIDDEN LAYER 2   HIDDEN LAYER 3   OUTPUT LAYER
         (4)            (6)              (8)              (6)             (3)

        x₁ ●────────●───●────────────●───●────────────●───●────────────● y₁
           │    ╱   │   │    ╱       │   │    ╱       │   │    ╱       │
        x₂ ●───╱────●───●───╱────────●───●───╱────────●───●───╱────────● y₂
           │  ╱     │   │  ╱         │   │  ╱         │   │  ╱         │
        x₃ ●─╱──────●───●─╱──────────●───●─╱──────────●───●─╱──────────● y₃
           │╱       │   │╱           │   │╱           │   │╱           │
        x₄ ●────────●───●────────────●───●────────────●───●────────────●
           ╲       ╱│   ╲           ╱│   ╲           ╱│   ╲           ╱
            ╲     ╱ │    ╲         ╱ │    ╲         ╱ │    ╲         ╱
             ╲   ╱  │     ╲       ╱  │     ╲       ╱  │     ╲       ╱
              ╲ ╱   │      ╲     ╱   │      ╲     ╱   │      ╲     ╱
               ●    │       ●───●    │       ●───●    │       ●───●
               │    │       │   │    │       │   │    │       │
               ●    │       ●───●    │       ●───●    │       ●
               │    │       │   │    │       │   │    │
               ●────────────●───●────────────●───●────────────●

     Features:     128 neurons    256 neurons     128 neurons    Classes:
     • x₁ (age)      ReLU           ReLU            ReLU         • Class A
     • x₂ (income)   Dropout(0.2)   Dropout(0.3)    Dropout(0.2)  • Class B
     • x₃ (score)    L2 Reg.        L2 Reg.         L2 Reg.      • Class C
     • x₄ (rating)

Mathematical Formulation:
┌─────────────────────────────────────────────────────────────┐
│  h⁽ˡ⁺¹⁾ = f(W⁽ˡ⁾h⁽ˡ⁾ + b⁽ˡ⁾)                              │
│                                                             │
│  Where:                                                     │
│  • h⁽ˡ⁾ = hidden layer l                                    │
│  • W⁽ˡ⁾ = weight matrix                                     │
│  • b⁽ˡ⁾ = bias vector                                       │
│  • f() = activation function (ReLU, Sigmoid, Tanh)         │
└─────────────────────────────────────────────────────────────┘
```

### Key Components:
- **Forward Propagation**: Data flows left to right
- **Weights (W)**: Connection strengths between neurons
- **Biases (b)**: Threshold adjustments for each neuron
- **Activation Functions**: Introduce non-linearity
- **Regularization**: Prevent overfitting (L2, Dropout)

---

## 2. Convolutional Neural Network (CNN) Architecture

```
INPUT IMAGE → FEATURE EXTRACTION → CLASSIFICATION

   28×28×1        Conv+Pool Layers      Fully Connected
  ┌────────┐
  │ ░░░░░░ │     ┌─────┐  ┌─────┐        ┌─────┐  ┌─────┐  ┌─────┐
  │ ░░██░░ │ →   │CONV │→ │POOL │   →   │ FC  │→ │DROP │→ │ FC  │
  │ ░██░██ │     │3×3  │  │2×2  │  ╲    │ 128 │  │ 0.5 │  │ 10  │
  │ ░░░░░░ │     │ReLU │  │Max  │   ╲   │     │  │     │  │     │
  └────────┘     └─────┘  └─────┘    ╲  └─────┘  └─────┘  └─────┘
                                      ╲      ↓       ↓       ↓
                ┌─────┐  ┌─────┐       ╲ ┌─────┐ ┌─────┐ ┌─────┐
                │CONV │→ │POOL │        →│ReLU │ │ReLU │ │ Soft│
                │3×3  │  │2×2  │         │     │ │     │ │ max │
                │ReLU │  │Max  │         └─────┘ └─────┘ └─────┘
                └─────┘  └─────┘              ↓       ↓      ↓
                     ↓       ↓            Activation Dropout Classes
                ┌─────────┐  ┌─────────┐
                │26×26×32 │  │13×13×32 │
                └─────────┘  └─────────┘
                     ↓           ↓
                ┌─────────┐  ┌─────────┐
                │24×24×64 │  │12×12×64 │
                └─────────┘  └─────────┘
                                ↓
                          ┌─────────┐
                          │Flatten  │ → 9216 features
                          └─────────┘

CONVOLUTION OPERATION:
    Input        Filter      Output
┌─ 1  2  3 ─┐  ┌─1  0─┐   ┌─ 4 ─┐
│  4  5  6   │  │ 0  1 │ = │  8  │  (4×1 + 5×0 + 7×0 + 8×1 = 4+8 = 12)
└─ 7  8  9 ─┘  └─────┘    └────┘

KEY CNN CONCEPTS:
• Convolution: Feature detection using filters/kernels
• Pooling: Dimensionality reduction (Max, Average)
• Parameter Sharing: Same filter applied across input
• Translation Invariance: Features detected regardless of position
• Hierarchical Features: Low-level → High-level features
```

---

## 3. Recurrent Neural Network (RNN) & LSTM

### Simple RNN Architecture:
```
TIME STEPS:    t=1      t=2      t=3      t=4      t=5
              ────────────────────────────────────────→

INPUT:         x₁   →   x₂   →   x₃   →   x₄   →   x₅
               │        │        │        │        │
               ▼        ▼        ▼        ▼        ▼
RNN CELL:    ┌───┐    ┌───┐    ┌───┐    ┌───┐    ┌───┐
             │RNN│ →  │RNN│ →  │RNN│ →  │RNN│ →  │RNN│
             └───┘    └───┘    └───┘    └───┘    └───┘
               │        │        │        │        │
HIDDEN:       h₁   →   h₂   →   h₃   →   h₄   →   h₅
               │        │        │        │        │
OUTPUT:        y₁       y₂       y₃       y₄       y₅

Mathematical Formulation:
• hₜ = tanh(Wₕₕhₜ₋₁ + Wₓₕxₜ + bₕ)
• yₜ = Wₕᵧhₜ + bᵧ
```

### LSTM (Long Short-Term Memory) Cell:
```
                    LSTM CELL ARCHITECTURE
                   ┌─────────────────────────┐
                   │                         │
    Cₜ₋₁ ────────→ │    ┌─ × ─┐  ┌─ + ─┐   │ ────→ Cₜ
                   │    │  f  │  │     │   │   (Cell State)
    hₜ₋₁ ──┐       │    └─────┘  └─────┘   │
           │       │        │       ╱     │
    xₜ ────┤───────│────────┼──────╱──────│────→ hₜ
           │       │    ┌─ × ─┐  ┌─ × ─┐   │   (Hidden State)
           │       │    │  i  │  │ tanh│   │
           │       │    └─────┘  └─────┘   │
           │       │        │       │     │
           └───────│────────┼───────┼─────│───┐
                   │    ┌─ × ─┐      │     │   │
                   │    │  o  │      │     │   │
                   └────└─────┘──────┼─────┘   │
                            │        │         │
                            │    ┌───────┐    │
                            │    │ tanh  │    │
                            │    └───────┘    │
                            │        │        │
                            └────────×────────┘

GATES:
• f = Forget Gate    (What to forget from cell state)
• i = Input Gate     (What new info to store)
• o = Output Gate    (What to output)
• tanh = New values  (Candidate values)

EQUATIONS:
• fₜ = σ(Wf·[hₜ₋₁,xₜ] + bf)     Forget Gate
• iₜ = σ(Wi·[hₜ₋₁,xₜ] + bi)     Input Gate
• C̃ₜ = tanh(WC·[hₜ₋₁,xₜ] + bC)  Candidate
• Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ      Cell State
• oₜ = σ(Wo·[hₜ₋₁,xₜ] + bo)     Output Gate
• hₜ = oₜ * tanh(Cₜ)            Hidden State
```

---

## 4. Transfer Learning Architecture

```
PRE-TRAINED MODEL              NEW TASK ADAPTATION
(e.g., ResNet-50)

┌─────────────────┐            ┌─────────────────┐
│ INPUT LAYER     │ ──────────→│ SAME INPUT      │
│ (224×224×3)     │            │ (224×224×3)     │
└─────────────────┘            └─────────────────┘
         │                              │
┌─────────────────┐  ❄️ FROZEN   ┌─────────────────┐
│ CONV BLOCK 1    │ ──────────→ │ CONV BLOCK 1    │
│ (112×112×64)    │             │ (112×112×64)    │
└─────────────────┘             └─────────────────┘
         │                              │
┌─────────────────┐  ❄️ FROZEN   ┌─────────────────┐
│ CONV BLOCK 2    │ ──────────→ │ CONV BLOCK 2    │
│ (56×56×128)     │             │ (56×56×128)     │
└─────────────────┘             └─────────────────┘
         │                              │
┌─────────────────┐  ❄️ FROZEN   ┌─────────────────┐
│ CONV BLOCK 3    │ ──────────→ │ CONV BLOCK 3    │
│ (28×28×256)     │             │ (28×28×256)     │
└─────────────────┘             └─────────────────┘
         │                              │
┌─────────────────┐  🔥 FINE-TUNE ┌─────────────────┐
│ CONV BLOCK 4    │ ──────────→ │ CONV BLOCK 4    │
│ (14×14×512)     │             │ (14×14×512)     │
└─────────────────┘             └─────────────────┘
         │                              │
┌─────────────────┐  ✂️ REMOVE    ┌─────────────────┐
│ GLOBAL POOL     │    ╱╱╱╱╱╱╱╱→ │ GLOBAL POOL     │
│ (2048)          │              │ (2048)          │
└─────────────────┘              └─────────────────┘
         │                              │
┌─────────────────┐  ✂️ REMOVE           │
│ DENSE (1000)    │              ┌─────────────────┐
│ ImageNet Classes│              │ 🆕 DENSE (512)   │
└─────────────────┘              │ + ReLU + Dropout│
                                 └─────────────────┘
                                          │
                                 ┌─────────────────┐
                                 │ 🆕 DENSE (256)   │
                                 │ + ReLU + Dropout│
                                 └─────────────────┘
                                          │
                                 ┌─────────────────┐
                                 │ 🆕 OUTPUT (N)    │
                                 │ Your Classes    │
                                 └─────────────────┘

TRANSFER LEARNING STRATEGIES:

1. FEATURE EXTRACTION          2. FINE-TUNING              3. FULL TRAINING
┌─────────────────────┐       ┌─────────────────────┐     ┌─────────────────────┐
│❄️ Freeze all layers  │       │❄️ Freeze early layers│     │🔥 Train everything   │
│🆕 Train classifier   │       │🔥 Unfreeze top layers│     │🎲 Random weights     │
│⚡ Fast & Stable      │       │🎯 Better performance │     │💪 Maximum flexibility│
│📊 Small datasets     │       │📈 Medium datasets    │     │🗃️ Large datasets     │
│🎯 85-90% accuracy    │       │🎯 90-95% accuracy    │     │🎯 Variable accuracy  │
└─────────────────────┘       └─────────────────────┘     └─────────────────────┘

PERFORMANCE COMPARISON:
Method              Time    Data Needed    Typical Accuracy
────────────────────────────────────────────────────────────
Feature Extraction   Fast    Small (100s)     85-90%
Fine-tuning         Medium   Medium (1000s)   90-95%
From Scratch        Slow     Large (10000s)   Variable
```

---

## 5. Architecture Comparison Summary

```
┌─────────────┬──────────────┬──────────────┬─────────────┬──────────────┐
│ NETWORK     │ BEST FOR     │ KEY FEATURE  │ PARAMETERS  │ APPLICATIONS │
├─────────────┼──────────────┼──────────────┼─────────────┼──────────────┤
│ DNN/MLP     │ Tabular Data │ Fully Conn.  │ High        │ Classification│
│             │ Structured   │ Universal    │             │ Regression   │
│             │ Features     │ Approximator │             │ Prediction   │
├─────────────┼──────────────┼──────────────┼─────────────┼──────────────┤
│ CNN         │ Images       │ Local Conn.  │ Efficient   │ Computer     │
│             │ Spatial Data │ Convolution  │ Parameters  │ Vision       │
│             │ Grid-like    │ Translation  │             │ Image Class. │
├─────────────┼──────────────┼──────────────┼─────────────┼──────────────┤
│ RNN/LSTM    │ Sequences    │ Memory       │ Moderate    │ NLP          │
│             │ Time Series  │ Temporal     │             │ Speech       │
│             │ Text         │ Dependencies │             │ Time Series  │
├─────────────┼──────────────┼──────────────┼─────────────┼──────────────┤
│ Transfer    │ Limited Data │ Pre-trained  │ Low         │ Fine-tuning  │
│ Learning    │ New Domains  │ Features     │ Training    │ Domain Adapt │
│             │ Fast Deploy  │ Knowledge    │ Time        │ Quick Deploy │
└─────────────┴──────────────┴──────────────┴─────────────┴──────────────┘

CHOOSING THE RIGHT ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────────────┐
│ Data Type           → Recommended Architecture                          │
├─────────────────────────────────────────────────────────────────────────┤
│ 📊 Tabular/Numeric  → Deep Neural Network (MLP)                        │
│ 🖼️ Images            → Convolutional Neural Network (CNN)               │
│ 📝 Text/Sequences    → Recurrent Neural Network (RNN/LSTM)             │
│ 🎵 Time Series       → RNN/LSTM or 1D CNN                              │
│ 🔄 Limited Data      → Transfer Learning                                │
│ 🎯 Multi-modal       → Ensemble/Hybrid Architectures                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways for Week 5 - Regularization Context

### Overfitting in Different Architectures:

**DNNs**: Use Dropout, L2 regularization, Early stopping
**CNNs**: Data augmentation, Dropout, Batch normalization
**RNNs**: Dropout (input/output), Gradient clipping, Regularization
**Transfer Learning**: Inherently regularized through pre-trained features

### Best Practices:
1. **Start Simple**: Begin with basic architecture, add complexity gradually
2. **Monitor Validation**: Track train vs. validation performance gap
3. **Regularization Mix**: Combine multiple techniques (L2 + Dropout + Early stopping)
4. **Architecture Choice**: Match network type to data characteristics
5. **Transfer First**: Try transfer learning before training from scratch

---

*Created for 21CSE558T Deep Neural Network Architectures*
*M.Tech Course - SRM University*
*Week 5, Day 4: Overfitting & Regularization Tutorial*