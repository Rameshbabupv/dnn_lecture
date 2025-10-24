---
tags:
  - week4
  - LaTex
---

# Gradient Descent Update Rule

$$
\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t)
$$


# Convergence Checks

$$
\|\nabla J(\theta)\| < \varepsilon 
\quad \text{or} \quad
|J_{t+1} - J_t| < \delta
$$


# L2 Regularization (Weight Decay)
$$
\theta_{t+1} = (1 - \alpha \lambda)\,\theta_t - \alpha \nabla_\theta J(\theta_t)
$$



# Backpropagation (Chain Rule Example)
$$
\frac{\partial J}{\partial a} 
= \frac{\partial J}{\partial y} 
  \cdot \frac{\partial y}{\partial z} 
  \cdot \frac{\partial z}{\partial a}
$$
![[output (1).png]]