# Week 4 — Day 3: Gradient Descent Theory (Instructor Slides)

## 1) Story Hook — The Night Hike
- Feel the slope → step opposite → check → repeat.

$$
\textbf{Hiker's Rule:}\quad \theta \leftarrow \theta - \alpha\,\nabla J(\theta)\ \ \text{until}\ \ \lVert\nabla J(\theta)\rVert < \varepsilon
$$

Notes: Valley = low loss; slope = gradient; stride = learning rate.

---

## 2) Training as Optimization
- Goal: find parameters (weights/biases) that minimize loss.

$$
\min_{\theta} \ J(\theta; \mathcal{D})
$$

Where \(\theta\) are model parameters and \(\mathcal{D}\) is data.

---

## 3) Loss Landscapes (Intuition)
- Non-convex surfaces: local minima, saddles, plateaus.
- Height = error; position = weights.

No formula — use 2D contour/height map visuals during class.

---

## 4) Common Cost Functions
- Regression (MSE):

$$
J(\theta) = \frac{1}{2m}\sum_{i=1}^{m} \big(f_\theta(x^{(i)}) - y^{(i)}\big)^2
$$

- Binary Classification (Logistic, cross-entropy):

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y^{(i)}\,\log\hat{y}^{(i)} + (1-y^{(i)})\,\log\big(1-\hat{y}^{(i)}\big) \Big]
$$

- Multiclass (Softmax cross-entropy):

$$
J(\theta) = -\frac{1}{m}\sum_{i=1}^{m} \sum_{k=1}^{K} \mathbf{1}[y^{(i)}=k]\,\log\hat{p}_k^{(i)}
$$

---

## 5) Gradients — Direction of Steepest Ascent
- Definition (vector of partial derivatives):

$$
\nabla J(\theta) = \left[ \frac{\partial J}{\partial \theta_1},\ldots,\frac{\partial J}{\partial \theta_n} \right]^\top
$$

- We step opposite the gradient to decrease the loss.

---

## 6) Gradient Descent Update Rule
- Iterative improvement:

$$
\theta_{t+1} = \theta_t - \alpha\,\nabla J(\theta_t)
$$

- Stop when gradient is small or loss change is below a threshold.

---

## 7) Learning Rate (\(\alpha\)) Effects
- Too small → slow progress; too large → overshoot/diverge.

$$
\text{Choose } \alpha \text{ s.t. } J(\theta_{t+1}) < J(\theta_t) \quad \text{(most steps)}
$$

- Practical: start moderate; adjust based on loss curve behavior.

---

## 8) Quick Visualization Demo (1D Quadratic)
- Function: \(f(x)=x^2\), gradient: \(g(x)=2x\).
- Live plot: steps shrink as \(|g|\) shrinks.

```python
import numpy as np, matplotlib.pyplot as plt
f = lambda x: x**2
g = lambda x: 2*x

def gd(x0=-6.0, alpha=0.5, iters=8):
    xs = [x0]
    for _ in range(iters):
        xs.append(xs[-1] - alpha * g(xs[-1]))
    return np.array(xs)

xs = np.linspace(-7, 7, 400)
path = gd()
plt.figure(figsize=(6,4))
plt.plot(xs, f(xs), 'k-', lw=2, label='f(x)=x^2')
plt.plot(path, f(path), 'ro-')
plt.title("Feel slope → Step opposite → Check → Repeat")
plt.xlabel("x"); plt.ylabel("loss"); plt.grid(True); plt.tight_layout(); plt.show()
```

---

## 9) Neural Network Training Loop (Mapping)
- Forward → Loss → Backprop (gradients) → Update.

$$
\theta \leftarrow \theta - \alpha\,\widehat{\nabla J(\theta)}
$$

Where \(\widehat{\nabla J(\theta)}\) is computed via backprop over data.

```python
for epoch in range(E):
    y_hat = model(X)
    loss = criterion(y_hat, y)
    loss.backward()      # compute gradients
    optimizer.step()     # θ ← θ − α·∇J(θ)
    optimizer.zero_grad()
```

---

## 10) Quick Checks & Exit Ticket
- Why subtract the gradient? What does \(\alpha\) control?
- What happens near a saddle vs a plateau?

$$
\text{Remember:}\quad \theta \leftarrow \theta - \alpha\,\nabla J(\theta)\quad \text{until}\quad \lVert\nabla J\rVert \approx 0
$$

---

## 11) Teaser for Day 4
- Full-batch vs stochastic vs mini-batch.

$$
\text{BGD: use all } m,\quad \text{SGD: use 1},\quad \text{Mini-batch: use } b \ll m
$$

Noise helps escape poor minima; batches leverage vectorization and GPUs.



# Learning problem + loss landscapes

```python
import numpy as np, matplotlib.pyplot as plt

def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

x = np.linspace(-6, 6, 400)
y = np.linspace(-6, 6, 400)
X, Y = np.meshgrid(x, y)
Z = himmelblau(X, Y)

plt.figure(figsize=(6,5))
cs = plt.contour(X, Y, Z, levels=30, cmap='viridis')
plt.clabel(cs, inline=1, fontsize=8)
mins = [(3.0, 2.0), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)]
for a,b in mins:
    plt.plot(a, b, 'ro')
plt.title('Loss Landscape Analogy: Multiple Minima & Saddles')
plt.xlabel('θ₁'); plt.ylabel('θ₂'); plt.grid(True, alpha=0.2)
plt.tight_layout(); plt.show()
```



#  Cost functions: MSE vs cross-entropy
- Classification: use Cross-Entropy (CE) with sigmoid (binary) or softmax (multiclass).

Definitions

- MSE (Regression):
J(θ) = (1/2m) Σ_i (ŷ_i − y_i)^2
- Binary CE (Sigmoid):
J(θ) = −(1/m) Σ_i [y_i log ŷ_i + (1−y_i) log(1−ŷ_i)]
where ŷ_i = σ(z_i)
- Multiclass CE (Softmax):
J(θ) = −(1/m) Σ_i Σ_k 1[y_i=k] log p_i,k
where p_i = softmax(z_i)

Gradients (Why CE is preferred for classification)

- MSE w.r.t. prediction: ∂J/∂ŷ = (ŷ − y). With sigmoid/softmax output, chain rule can shrink gradients near saturation.
- Binary CE + sigmoid: neat cancellation → ∂J/∂z = (ŷ − y). Stronger gradients when wrong and confident.
- Softmax CE: ∂J/∂z_k = (p_k − 1[y=k]). Simple and stable for backprop.

Behavior & Intuition

- MSE:
    - Penalizes squared error; symmetric; sensitive to outliers.
    - Works best when Gaussian noise assumptions hold.
    - For classification, slower convergence and poor probabilistic calibration compared to CE.
- Cross-Entropy:
    - Measures dissimilarity between true and predicted distributions.
    - Rewards putting probability mass on the correct class.
    - Encourages calibrated probabilities and faster learning signals when predictions are confidently wrong.

Numerical Stability Tips

- Binary CE: use stable APIs (e.g., BCEWithLogitsLoss / BinaryCrossentropy(from_logits=True)) to avoid log(0).
- Softmax CE: use CrossEntropyLoss / CategoricalCrossentropy(from_logits=True) which combine log-softmax + NLL.
- Clip probabilities when implementing manually (e.g., ε = 1e-7).

Class Imbalance & Regularization

- Use class weights or focal loss for imbalanced data.
- Label smoothing for softmax: replace one-hot y with (1−ε) for correct class and ε/(K−1) for others to reduce overconfidence.

Quick Rule of Thumb

- Continuous targets → MSE.
- Binary class → BCE with logits.
- Multiclass → Softmax CE with logits.
- Imbalanced → CE + class weights or focal loss.





# Gradients: Intuition & Notation

- Gradient meaning: vector of local slopes; points toward steepest increase. Use negative gradient for steepest decrease.
- Notation: For scalar loss J(θ) with θ ∈ R^n, ∇J(θ) = [∂J/∂θ1, …, ∂J/∂θn]^T. Stationary point when ||∇J|| ≈ 0.
- Directional derivative: Change of J along unit direction u is D_u J(θ) = ∇J(θ) · u. Max decrease occurs at u = -∇J/||∇J||.
- Quadratic example: J(θ) = 1/2 ||Aθ − b||^2 → ∇J(θ) = A^T(Aθ − b); explains smooth, convex valleys.
- Chain rule (backprop core): If y = g(z) and z = h(θ), then ∂J/∂θ = (∂J/∂y)(∂y/∂z)(∂z/∂θ). Deep nets apply this layer-by-layer.
- Linear layer gradients: For y = W x + b, upstream gradient δ = ∂J/∂y:
    - ∂J/∂W = δ x^T
    - ∂J/∂b = δ
    - ∂J/∂x = W^T δ
- Activation gradients: For elementwise a = φ(z), ∂J/∂z = (∂J/∂a) ⊙ φ′(z). Examples: ReLU′(z)=1[z>0], Sigmoid′(z)=σ(z)(1−σ(z)).
- Shapes matter: Track tensor shapes to avoid mistakes. Gradients match parameter shapes.
- Non-differentiable points: ReLU at 0 uses a subgradient; frameworks handle this consistently.
- Numerical check (sanity): Finite-difference around θ:
    - ∂J/∂θ_i ≈ (J(θ + ε e_i) − J(θ − ε e_i)) / (2ε); compare to autograd (ε ~ 1e−5).


# GD algorithm: update rule + loop
## **Planning GD Algorithm Explanation
GD Algorithm: Update Rule + Loop

- Objective: minimize J(θ) over data D by iterative updates using local slope.
- Core update:
    - θ_{t+1} = θ_t − α ∇J(θ_t)
    - With L2 (λ): θ_{t+1} = (1 − αλ)θ_t − α ∇J_data(θ_t)
- Basic loop (pseudo):
    - initialize θ
    - repeat: predict → compute loss J → backprop to get ∇J → θ ← θ − α∇J
    - until stop criterion met
- Terminology:
    - Batch: subset of data used for one update
    - Step/Iteration: one parameter update
    - Epoch: full pass over training set
- Convergence/Stopping:
    - Max epochs or steps
    - Gradient norm small: ||∇J|| < ε
    - Validation loss plateau (early stopping with patience)
    - Loss decrease per epoch < δ
- Monitoring:
    - Track train/val loss and accuracy each epoch
    - Log learning rate, gradient norms; visualize curves
- Minimal code patterns:
    - PyTorch:
    - for epoch in E: for X,y in loader: loss = crit(model(X), y); opt.zero_grad(); loss.backward(); opt.step()
- Keras:
    - model.compile(optimizer, loss, metrics); model.fit(X_train, y_train, validation_data=..., callbacks=[EarlyStopping])
- Practical tips:
    - Shuffle data each epoch; set model to train/eval modes appropriately
    - Use “with logits” losses to avoid instability (log-softmax inside)
    - Normalize inputs; standardize targets for regression
- Common gotchas:
    - Forgetting zero_grad (accumulates gradients)
    - Mixing probabilities with logits in CE
    - Wrong reduction (mean vs sum) changes effective α
- Complexity:
    - Per step ≈ forward + backward cost; batch size trades steps vs work/step
- Rule of thumb
    - Start with mini-batch, α tuned on val set; add early stopping and LR schedule if plateau

# “Learning rate: small vs large”?
Learning Rate: Small vs Large

- Stability insight: For smooth convex quadratics, need 0 < α < 2/L (L ≈ largest curvature). In deep nets, treat α as tuned empi
Symptoms

- Too small: very slow loss decrease; tiny weight updates; many epochs.
- Too large: loss oscillates or explodes; NaNs/inf; accuracy stuck or erratic.
- Just right: steady, mostly monotonic train-loss drop; faster early gains, slower later.

Quick Heuristics

- If loss rises or NaNs in first 100 steps → divide α by 10.
- If loss zigzags strongly → reduce α by 2–5×.
- If loss flat for many steps → increase α by 2×.
- Track update ratio: mean(|Δθ|)/mean(|θ|) target ~ 1e−3 to 1e−2.

Rules of Thumb

- Optimizer defaults: Adam 1e−3, SGD+momentum 0.1 (normalized inputs).
- Batch-size relation: larger batches tolerate larger α (approx. linear scaling).
- Normalize inputs; consider gradient clipping if spikes occur.

Schedules

- Step decay: α ← α·γ every s epochs (e.g., γ=0.1).
- Exponential decay: α_t = α_0·exp(−kt).
- Cosine decay (often with warmup): smooth to near-zero by end.
- Reduce-on-plateau: shrink α when val loss stalls (patience-based).

LR Range Test (1–2 minutes to run on small batch)

- Linearly increase α from very small to large over a few epochs; plot loss vs α; pick α near the largest value before loss shoo
ts up.

Pseudo:

- α from 1e−6 → 1
- Update α each batch; record loss
- Choose α ≈ 1/5 to 1/10 of the divergence point

Snippets

- PyTorch (ReduceLROnPlateau):
    - scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    - scheduler.step(val_loss)
- Keras:
    - callbacks=[tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)]



# Visualization demo: 1D quadratic GD
Goal

- Visualize GD on f(x)=x² with different learning rates to show slow, optimal, and divergent behavior.
Key Facts

- Grad: f′(x)=2x. Update: x_{t+1} = x_t − α·2x_t = (1−2α) x_t.
- Stability: need |1−2α| < 1 ⇒ 0 < α < 1. Best contraction near α≈0.5.

Code: Compare Learning Rates
import numpy as np, matplotlib.pyplot as plt

f = lambda x: x**2
g = lambda x: 2*x

def gd_path(x0, alpha, iters):
    xs = [x0]
    for _ in range(iters):
        xs.append(xs[-1] - alpha * g(xs[-1]))
    return np.array(xs)

x0 = -6.0
alphas = [0.05, 0.5, 1.2]  # small, near-optimal, divergent
labels  = ["α=0.05 (slow)", "α=0.5 (fast)", "α=1.2 (diverges)"]
colors  = ["tab:blue", "tab:green", "tab:red"]

xs = np.linspace(-7, 7, 400)
plt.figure(figsize=(8,5))
plt.plot(xs, f(xs), 'k-', lw=2, label='f(x)=x²')

for a, lab, c in zip(alphas, labels, colors):
    path = gd_path(x0, a, iters=12)
    plt.plot(path, f(path), 'o-', color=c, label=lab, alpha=0.9)

plt.title("1D GD on f(x)=x²: effect of learning rate")
plt.xlabel("x"); plt.ylabel("loss"); plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
Talking Points

- Small α: steps shrink, but many iterations; monotone decrease.
- α≈0.5: near-optimal for this quadratic; rapid convergence.
- Large α (≥1): flips across the minimum; oscillations grow → divergence.

Optional: Iteration Table (print for one α)
alpha = 0.5
path = gd_path(x0, alpha, 8)
for t, x in enumerate(path):
    print(f"t={t:2d} | x={x:7.3f} | f(x)={f(x):7.3f}")
Instructor Prompts

- “Predict the next point given current x and α.”
- “How does changing α alter (1−2α) and the contraction speed?”
- “What visual evidence shows divergence vs convergence?”




# NN training loop connection
NN Training Loop Connection

- Forward pass: compute predictions ŷ = f_θ(X); keep intermediates for backprop.
- Loss: pick task-appropriate loss (MSE for regression; CE with logits for classification).
- Backward pass: autograd applies chain rule to get ∇_θ J(θ) from the loss.
- Update: optimizer applies θ ← θ − α∇J(θ) (plus any regularization/momentum).
- Batching: iterate mini-batches for speed and stable updates; shuffle each epoch.
- Modes: set model.train() for training (enables dropout/BN updates) and model.eval() for validation.

Minimal PyTorch Pattern
model.train()
for X, y in train_loader:
    optimizer.zero_grad()
    logits = model(X)                  # forward
    loss = criterion(logits, y)        # e.g., CrossEntropyLoss (expects logits)
    loss.backward()                    # compute ∇J
    optimizer.step()                   # θ ← θ − α∇J
Validation:
model.eval()
with torch.no_grad():
    for Xv, yv in val_loader:
        logits = model(Xv)
        val_loss += criterion(logits, yv).item()
Common gotchas:

- Use “with logits” losses; don’t apply softmax/sigmoid before CE-with-logits.
- Always optimizer.zero_grad() before backward() (grads accumulate).
- Normalize inputs; track metrics and early stopping on val loss.

Keras Equivalent
model.compile(optimizer="adam",
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=E,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])