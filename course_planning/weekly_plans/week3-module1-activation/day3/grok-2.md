### Cheat Sheet Handout: Activation Functions Quick Reference

Print this as a one-page PDF or handout. Use a clean layout with bold headings, equations in LaTeX-style formatting if possible (e.g., via tools like Overleaf), and small icons for pros/cons (e.g., thumbs up/down).

| Activation Function | Mathematical Definition | Output Range | Gradient (Derivative) | Pros | Cons |
|---------------------|--------------------------|--------------|-----------------------|------|------|
| **Sigmoid** | σ(x) = 1 / (1 + e^{-x}) | [0, 1] | σ'(x) = σ(x) * (1 - σ(x))<br>(Max: 0.25 at x=0) | - Smooth and differentiable<br>- Interpretable as probability (great for binary classification outputs)<br>- Introduces non-linearity | - Vanishing gradients for large \|x\| (slow learning in deep nets)<br>- Not zero-centered (outputs always positive, can cause inefficient updates)<br>- Computationally expensive (exponential) |
| **Tanh** | tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x}) | [-1, 1] | tanh'(x) = 1 - tanh²(x)<br>(Max: 1 at x=0) | - Zero-centered (better for symmetric data, reduces bias in updates)<br>- Stronger gradients than sigmoid<br>- Similar smoothness to sigmoid | - Still suffers from vanishing gradients for large \|x\|<br>- Computationally expensive (exponentials) |
| **ReLU** | f(x) = max(0, x) | [0, ∞) | f'(x) = 1 if x > 0<br>0 if x ≤ 0 | - No vanishing gradients for x > 0 (faster training in deep nets)<br>- Sparse activation (efficient, as many neurons output 0)<br>- Very fast computation (simple max operation) | - Dying ReLU problem (neurons can get stuck at 0 if inputs always negative)<br>- Not zero-centered<br>- Undefined gradient at x=0 (but handled in practice) |
| **Leaky ReLU** (and Variants) | f(x) = x if x > 0<br>αx if x ≤ 0 (α=0.01 typically)<br>Variants: PReLU (learnable α), ELU (e^x - 1 for x<0) | (-∞, ∞) for Leaky/PReLU<br>[-1, ∞) for ELU | f'(x) = 1 if x > 0<br>α if x ≤ 0 | - Fixes dying ReLU by allowing small negative gradients<br>- Maintains ReLU's speed and non-vanishing benefits<br>- Variants like ELU are smoother and zero-centered | - Small leak (α) may not fully prevent issues in very deep nets<br>- Additional hyperparameter (α) to tune<br>- Still not fully symmetric |

**Selection Criteria Tips:**
- For output layers: Sigmoid (binary probs), Tanh (symmetric outputs), Softmax (multi-class – mention as extension).
- For hidden layers: Prefer ReLU/Leaky ReLU in modern deep nets to avoid vanishing gradients; use sigmoid/tanh only for shallow nets or specific needs.
- Universal Approximation: Non-linear activations let MLPs approximate any function – key for XOR-like problems.
- Real-World Note: In XOR, linear = fail; non-linear = success via warped boundaries.

### Slide Deck Suggestions (for Projector)

Create slides in PowerPoint/Google Slides. Aim for 10-15 slides total, minimal text, large fonts. Embed plots generated from the Jupyter code below (save as images via Matplotlib's `plt.savefig()`). Use a clean theme (e.g., blue tones for "neural" feel). Include slide numbers tied to timeline.

1. **Title Slide (Opening)**: "Session 7: Activation Functions Mastery" – Include objective, timeline overview.
2. **Hook Slide (Opening)**: XOR Recap – Image of linear perceptron failing XOR (decision line) vs. MLP succeeding (curved boundaries). Text: "Why non-linearity matters: From AND/OR to XOR magic."
3. **Why Activations? (Opening)**: Bullet points on collapse to linear models without them. Brief Universal Approximation mention. Analogy: "Like adding curves to a straight road for complex paths."
4. **Sigmoid Intro (Classical)**: Equation, properties. Embed plot of σ(x) and σ'(x).
5. **Sigmoid Derivation (Classical)**: Step-by-step math (use whiteboard for live, but slide has key steps: d/dx = ... → σ(1-σ)).
6. **Sigmoid Pros/Cons & Example (Classical)**: Bullets + spam classifier example. Plot showing vanishing gradients (flat tails).
7. **Tanh Intro (Classical)**: Equation, properties. Overlay plot with sigmoid.
8. **Tanh Gradient & Pros/Cons (Classical)**: Derivation steps, bullets. Sentiment analysis example.
9. **ReLU Intro (Modern)**: Equation, properties. Plot of f(x) and gradient.
10. **ReLU Pros/Cons & Example (Modern)**: Bullets + computer vision example. Matrix demo of sparsity (e.g., inputs [-2,3,-1,4] → outputs [0,3,0,4]).
11. **Leaky ReLU & Variants (Modern)**: Equations, properties. Comparison plot vs. ReLU.
12. **Leaky Pros/Cons & Example (Modern)**: Bullets + GANs/audio example.
13. **Key Takeaways (Wrap-Up)**: Selection criteria bullets. Teaser for Session 8.
14. **Q&A/Quiz (Wrap-Up)**: Poll questions: "Why sigmoid for binary output?" "Compute tanh'(0)." Homework teaser.

For polls: Use tools like Mentimeter or in-class raise hands.

### Demo Tool: Jupyter Notebook for Live Plotting

Share this via Google Colab (create a new notebook, paste code, share link). Run live during session for interactivity – e.g., change parameters, recompute gradients. Notebook structure:

- Cell 1: Imports
- Cell 2-5: Individual function plots
- Cell 6: Combined comparison plot

Here's the complete code (tested mentally – it's standard Matplotlib). Copy-paste into Colab.

```python
# Cell 1: Imports
import numpy as np
import matplotlib.pyplot as plt

# Cell 2: Sigmoid and Gradient
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return sigmoid(x) * (1 - sigmoid(x))

x = np.linspace(-10, 10, 400)
plt.figure(figsize=(8, 4))
plt.plot(x, sigmoid(x), label='Sigmoid σ(x)')
plt.plot(x, sigmoid_grad(x), label="Gradient σ'(x)")
plt.title('Sigmoid Function and Gradient')
plt.xlabel('x')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()  # For slides: plt.savefig('sigmoid_plot.png')

# Cell 3: Tanh and Gradient
def tanh(x):
    return np.tanh(x)

def tanh_grad(x):
    return 1 - tanh(x)**2

plt.figure(figsize=(8, 4))
plt.plot(x, tanh(x), label='Tanh(x)')
plt.plot(x, tanh_grad(x), label="Gradient tanh'(x)")
plt.title('Tanh Function and Gradient')
plt.xlabel('x')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()  # For slides: plt.savefig('tanh_plot.png')

# Cell 4: ReLU and Gradient
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return np.where(x > 0, 1, 0)

plt.figure(figsize=(8, 4))
plt.plot(x, relu(x), label='ReLU(x)')
plt.plot(x, relu_grad(x), label="Gradient ReLU'(x)")
plt.title('ReLU Function and Gradient')
plt.xlabel('x')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()  # For slides: plt.savefig('relu_plot.png')

# Cell 5: Leaky ReLU and Gradient (α=0.01)
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_grad(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

plt.figure(figsize=(8, 4))
plt.plot(x, leaky_relu(x), label='Leaky ReLU(x)')
plt.plot(x, leaky_relu_grad(x), label="Gradient Leaky ReLU'(x)")
plt.title('Leaky ReLU Function and Gradient (α=0.01)')
plt.xlabel('x')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()  # For slides: plt.savefig('leaky_relu_plot.png')

# Cell 6: Comparison Overlay
plt.figure(figsize=(10, 5))
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, tanh(x), label='Tanh')
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, leaky_relu(x), label='Leaky ReLU')
plt.title('Activation Functions Comparison')
plt.xlabel('x')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()  # For slides: plt.savefig('comparison_plot.png')
```

**Usage Tips:** In session, run cells live. For sparsity demo (ReLU), add a cell: `inputs = np.array([-2, 3, -1, 4]); print('Outputs:', relu(inputs))` → Shows [0,3,0,4]. For vanishing gradients, zoom into tails (e.g., `x = np.linspace(-5,5,400)` for focus).

### Additional Preparation Notes

- **Whiteboard Derivations:** Practice these: For sigmoid gradient, start with quotient rule: Let u=1, v=1+e^{-x}; du=0, dv=-e^{-x}; then simplify to σ(1-σ). Similar for tanh.
- **Engagement Hooks:** Prepare analogies (e.g., sigmoid as "soft switch" like dimmer light; ReLU as "on/off" like motion sensor). Poll via sticky notes or app.
- **Assessment:** Quick quiz questions: 1. sigmoid'(0) = 0.25 (compute: σ(0)=0.5, 0.5*0.5=0.25). 2. Preferred for deep XOR: ReLU (avoids vanishing).
- **Timing Check:** Rehearse to fit 1 hour – prioritize visuals over text.

This setup ties perfectly to prior knowledge, making the session engaging and educational! If you need tweaks, like adding ELU code or pre-made plot images, let me know.