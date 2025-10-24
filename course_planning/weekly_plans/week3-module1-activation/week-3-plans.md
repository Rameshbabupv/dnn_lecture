---
tags:
  - week3
---


# My Plans 

| Time | Description | what should be covered |






1. grok: https://grok.com/chat/f981e111-9bb1-4548-b543-d083bcc7c8b4
2. colab: https://colab.research.google.com/drive/1IdG1Lxq2GDQMMzzpF8awRsVC1Advef7d#scrollTo=FpOnhKQAr8tI
3. ChatGPT : https://chatgpt.com/share/68acfc5b-6ad0-8013-a9c5-4b4c3bee6e59



$$
A =
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}, \quad
B =
\begin{bmatrix}
5 & 6 \\
7 & 8
\end{bmatrix}
$$

$$
C = A \times B =
\begin{bmatrix}
1\cdot 5 + 2\cdot 7 & 1\cdot 6 + 2\cdot 8 \\
3\cdot 5 + 4\cdot 7 & 3\cdot 6 + 4\cdot 8
\end{bmatrix}
=
\begin{bmatrix}
19 & 22 \\
43 & 50
\end{bmatrix}
$$



![[Books#**Week-3 Sessions 7,8,9**]]





$$
\begin{array}{|c|c|c|c|c|c|}
\hline
\textbf{Activation Function} & \textbf{Mathematical Definition} & \textbf{Output Range} & \textbf{Gradient (Derivative)} & \textbf{Pros} & \textbf{Cons} \\
\hline
\text{Sigmoid} & 
\sigma(x) = \frac{1}{1+e^{-x}} &
[0,1] &
\sigma'(x) = \sigma(x)(1-\sigma(x)) \quad (\max: 0.25 \text{ at } x=0) &
\begin{array}{l}
- \text{Smooth, differentiable} \\
- \text{Probabilistic interpretation} \\
- \text{Non-linear}
\end{array} &
\begin{array}{l}
- \text{Vanishing gradients for large } |x| \\
- \text{Not zero-centered} \\
- \text{Computationally expensive}
\end{array} \\
\hline
\text{Tanh} &
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} &
[-1,1] &
\tanh'(x) = 1 - \tanh^2(x) \quad (\max: 1 \text{ at } x=0) &
\begin{array}{l}
- \text{Zero-centered} \\
- \text{Stronger gradients than Sigmoid} \\
- \text{Smooth, differentiable}
\end{array} &
\begin{array}{l}
- \text{Vanishing gradients for large } |x| \\
- \text{Expensive (exponentials)}
\end{array} \\
\hline
\text{ReLU} &
f(x) = \max(0,x) &
[0,\infty) &
f'(x) =
\begin{cases}
1 & x>0 \\
0 & x \leq 0
\end{cases} &
\begin{array}{l}
- \text{No vanishing gradient for } x>0 \\
- \text{Sparse activation (efficient)} \\
- \text{Fast computation}
\end{array} &
\begin{array}{l}
- \text{Dying ReLU (neurons stuck at 0)} \\
- \text{Not zero-centered} \\
- \text{Undefined at } x=0
\end{array} \\
\hline
\text{Leaky ReLU / Variants} &
f(x) =
\begin{cases}
x & x>0 \\
\alpha x & x \leq 0
\end{cases}, \ \alpha \approx 0.01
& (-\infty,\infty) &
f'(x) =
\begin{cases}
1 & x>0 \\
\alpha & x \leq 0
\end{cases} &
\begin{array}{l}
- \text{Fixes dying ReLU issue} \\
- \text{Maintains ReLU speed} \\
- \text{Variants (PReLU, ELU) smoother}
\end{array} &
\begin{array}{l}
- \text{Leak $\alpha$ needs tuning} \\
- \text{Still not fully symmetric} \\
- \text{ELU adds extra computation}
\end{array} \\
\hline
\end{array}
$$





Examples and Anolgies : 

![[Pasted image 20250828163303.png]]

![[Pasted image 20250828163315.png]]
![[Pasted image 20250828163332.png]]

# 🌍 **Real-World Analogies**

![[Pasted image 20250828163425.png]]
![[Pasted image 20250828163506.png]]

![[Pasted image 20250828163515.png]]


![[Pasted image 20250828163524.png]]

![[Pasted image 20250828163532.png]]


![[Pasted image 20250828163555.png]]






# **let’s slow down and actually walk through** 

![[Pasted image 20250828163722.png]]


![[Pasted image 20250828163738.png]]





![[Pasted image 20250828163802.png]]



![[Pasted image 20250828163814.png]]


![[Pasted image 20250828163826.png]]

![[Pasted image 20250828163836.png]]




![[Pasted image 20250828163917.png]]


![[Pasted image 20250828165138.png]]

$$
\textbf{Problem 1:} \quad \text{Input size = 2, Output size = 3} \\[6pt]
x = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad
W = \begin{bmatrix} 1 & -1 \\ 0 & 2 \\ 3 & 1 \end{bmatrix}, \quad
b = \begin{bmatrix} 0 \\ 1 \\ -1 \end{bmatrix}
$$

$$
\textbf{Problem 2:} \quad \text{Input size = 3, Output size = 2} \\[6pt]
x = \begin{bmatrix} 2 \\ -1 \\ 0 \end{bmatrix}, \quad
W = \begin{bmatrix} 1 & 0 & -2 \\ -1 & 2 & 1 \end{bmatrix}, \quad
b = \begin{bmatrix} 1 \\ -2 \end{bmatrix}
$$

$$
\textbf{Problem 3:} \quad \text{Input size = 2, Output size = 2} \\[6pt]
x = \begin{bmatrix} 3 \\ 1 \end{bmatrix}, \quad
W = \begin{bmatrix} 2 & -1 \\ -3 & 4 \end{bmatrix}, \quad
b = \begin{bmatrix} -1 \\ 2 \end{bmatrix}
$$






![[01_relu_vs_leaky.png]]


![[04_sigmoid_derivative.png]]
