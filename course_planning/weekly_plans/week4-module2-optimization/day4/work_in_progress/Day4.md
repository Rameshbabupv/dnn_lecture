
Assessment

- Quick quiz:
    - Gradient: direction of steepest increase; use −∇J for descent.
    - Update rule: θ ← θ − α∇J(θ). Stop when ||∇J|| < ε or ΔJ < δ.
    - MSE vs CE: MSE for regression; CE (with logits) for classification; CE gives stronger gradients when confidently wrong.
    - Learning rate: too small = slow; too large = oscillation/divergence.
    - Learning rate: too small = slow; too large = oscillation/divergence.
-
Exit ticket (2–3 minutes):
    - Write the update for a single weight w given gradient g and α.
    - Name two landscape features that slow GD and why (plateaus, saddles).
    - If loss oscillates between epochs, what change would you try first? (reduce α).
-
Mini exercise (sanity check):
    - Using f(x)=x², x0=−6: compute next two steps for α=0.5 and α=1.2; predict converge vs diverge.
    - Explain how the same logic applies to a NN’s last layer with CE loss.

Day 4 Preview

- Goals:
    - Compare Batch, Stochastic, and Mini-batch GD in theory and code.
    - Visualize convergence patterns and analyze compute/memory trade-offs.
    - Tune batch size and learning rate; introduce simple schedules.
    - Tune batch size and learning rate; introduce simple schedules.
-
Plan:
    - Implement three variants on a toy regression set; plot cost vs epoch.
    - Discuss when to use each method; introduce class defaults (mini-batch ~32).
    - Practical heuristics: scale α with batch size; shuffle; monitor stability.
    - Tease advanced optimizers (momentum, Adam) and why they help.


  # Variants overview: Batch vs SGD vs Mini-batch
    □ Batch GD details
    □ SGD details
    □ Mini-batch details
    □ Implementation + comparison
    □ Hyperparameter tuning
    □ Assessment + takeaways


## **Variants Overview

- Batch GD: uses full dataset per update.
    - Update: θ ← θ − α · (1/m) Σ_i ∇_θ 𝓛(f_θ(x_i), y_i)
    - Pattern: smooth convergence; 1 update/epoch.
- SGD: updates per single example.
    - Update: θ ← θ − α · ∇_θ 𝓛(f_θ(x_i), y_i) for each i
    - Pattern: noisy zigzag; many updates/epoch.
- Mini-batch GD: updates per small batch b.
    - Update: θ ← θ − α · (1/b) Σ_{i∈B} ∇_θ 𝓛(f_θ(x_i), y_i)
    - Pattern: balanced noise/speed; GPU-friendly.
Mini-batch GD: updates per small batch b.
    - Update: θ ← θ − α · (1/b) Σ_{i∈B} ∇_θ 𝓛(f_θ(x_i), y_i)
    - Pattern: balanced noise/speed; GPU-friendly.
-
Trade-offs:
    - Stability: Batch > Mini-batch > SGD
    - Speed/Wall time: Mini-batch ≥ SGD > Batch
    - Memory: SGD < Mini-batch < Batch
-
When to use:
    - Small datasets, deterministic runs → Batch GD.
    - Streaming/online or tight memory → SGD.
    - Most practical deep learning → Mini-batch (e.g., 32–128).


## **Batch GD details
Batch GD Details

- Definition: Uses the entire training set per update: θ ← θ − α (1/m) Σ_i ∇_θ 𝓛(f_θ(x_i), y_i).
- Convergence: Smooth, deterministic path; for convex/quadratic losses, converges if 0 < α < 2/L (L = Lipschitz constant of ∇J).
- Pros: Stable updates; exact gradient; easy to reason/debug; reproducible.
- Cons: One update per epoch → slow on large datasets; high memory footprint; poor hardware utilization vs mini-batching.
-
Pros: Stable updates; exact gradient; easy to reason/debug; reproducible.
-
Cons: One update per epoch → slow on large datasets; high memory footprint; poor hardware utilization vs mini-batching.
-
Pseudocode:
    - initialize θ
    - repeat for epochs:
    - compute predictions over all m samples
    - compute mean loss J(θ)
    - compute gradient ∇J(θ) over all m
    - update θ ← θ − α ∇J(θ)

- Complexity: Each step costs one full forward + backward pass over m; wall-time heavy when m is large.
Complexity: Each step costs one full forward + backward pass over m; wall-time heavy when m is large.
-
When To Use: Small datasets (<10k), deterministic research settings, or to benchmark expected convergence behavior.
-
Numerical Tips:
    - Normalize/standardize inputs; use “with logits” losses for CE.
    - Regularization: L2 adds −αλθ term to update.
    - If loss increases, reduce α; consider line search for convex problems.
-
Framework Notes:
    - PyTorch: load whole dataset into a single batch (may be memory-bound); otherwise, simulate by aggregating all gradients before step.
    - Keras: set batch_size = len(X_train) for full-batch (watch RAM/GPU limits).


## **SGD details
- Definition: Update per single example i:
    - θ ← θ − α ∇θ 𝓛(fθ(x_i), y_i) after each sample (randomly ordered).
- Convergence: Noisy path; for convex problems converges in expectation with diminishing α (e.g., α_t ∝ 1/t).
- Pros: Very frequent updates; memory-light; supports online/streaming data; noise can escape shallow minima/saddles.
- Cons: High variance gradients; sensitive to α; loss curve fluctuates; slower final convergence without variance reduction.
-
Pros: Very frequent updates; memory-light; supports online/streaming data; noise can escape shallow minima/saddles.
-
Cons: High variance gradients; sensitive to α; loss curve fluctuates; slower final convergence without variance reduction.
- Working (7s • Esc to interrupt)
Practical setup
    - Shuffle each epoch (random reshuffling) rather than pure sampling with replacement.
    - Start with smaller α than batch GD; consider decay: α_t = α0/(1+kt) or ReduceLROnPlateau.
    - Normalize inputs; consider gradient clipping if spikes occur.
-
Pseudocode
    - for epoch: shuffle indices; for i in indices:
    - g ← ∇θ 𝓛(fθ(x_i), y_i)
    - θ ← θ − α g

- Variance control
    - Polyak/Ruppert averaging: keep running average of θ to stabilize final iterate.
    - Mini-batch b=1 is exact SGD; in practice b∈{4,8} often outperforms strict b=1.
    - Mini-batch b=1 is exact SGD; in practice b∈{4,8} often outperforms strict b=1.
-
Framework notes
    - PyTorch: use DataLoader with batch_size=1; remember zero_grad() each step.
    - Keras: model.fit(..., batch_size=1). Beware of poor device utilization.
-
When to use
    - Streaming data, severe memory limits, or when intentional noise aids exploration.
    - As a teaching/diagnostic baseline before moving to mini-batch with momentum.


## **Mini-batch details
Mini-batch Details

- Definition: Update per batch B of size b:
θ ← θ − α · (1/b) Σ_{i∈B} ∇θ 𝓛(fθ(x_i), y_i)
- Why it’s default: Balanced noise/smoothness; efficient vectorization; excellent GPU utilization; manageable memory.
- Typical sizes: 16–256 (commonly 32, 64, 128). Choose by VRAM/CPU RAM and stability.
- Convergence pattern: Smoother than SGD, faster than full-batch; controlled stochasticity helps escape saddles.
- Pseudocode:
    - for epoch: shuffle; for batches B:
    - g ← (1/b) Σ_{i∈B} ∇θ 𝓛(fθ(x_i), y_i)
    - θ ← θ − α g
- Practical tips:
    - Shuffle each epoch; set drop_last=True for consistent batch norms.
    - Linear scaling rule: if b → k·b, try α → k·α with short warmup.
    - Gradient accumulation simulates large b when memory-limited (accumulate g for k steps before step()).
    - BatchNorm needs sufficiently large b; for tiny b, consider GroupNorm/LayerNorm.
- Framework notes:
    - PyTorch DataLoader: batch_size=b, shuffle=True, num_workers>0, pin_memory=True (GPU).
    - Keras: model.fit(..., batch_size=b); monitor steps_per_epoch for dataset iterators.
- When to use: Almost always in deep learning; start with b≈32–128, tune by utilization and loss smoothness.