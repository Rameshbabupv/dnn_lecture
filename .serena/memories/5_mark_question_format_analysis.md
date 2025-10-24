# 5-Mark Question Format Analysis for Deep Neural Network Architecture Course

## Context
Analysis of FA-1.pdf sample from different department (Convolutional Neural Networks course) to understand proper format and expectations for 5-mark questions in our Deep Neural Network Architecture course (21CSE558T).

## Key Findings from Sample Analysis

### Question Characteristics
- **Type**: Computational/application problems requiring step-by-step mathematical solutions
- **Structure**: Concrete scenarios with specific numerical values (input vectors, weights, bias)
- **Focus**: Forward propagation calculations for single neurons
- **Bloom's Level**: BL3 (Application) - students must apply learned formulas
- **Course Outcome**: Mix of CO1/CO2 focusing on fundamental understanding

### Expected Answer Format
1. **Detailed step-by-step working** (3-4 calculation steps minimum)
2. Show dot product calculation explicitly: `Y = w*x + b`
3. Show bias addition step
4. Apply activation function with mathematical formula
5. Provide **final numerical answer** with appropriate precision (e.g., 0.9707, 0.8808)

### Sample Question Pattern
```
"You have a single neuron with input vector X = [1, 2], weights W = [0.5, -1.0], and bias = 0.5. The activation function is sigmoid.
Q1. What is the output of the neuron?"

Expected Solution:
- Dot product: 0.5(1) + (-1)(2) + 0.5 = 0.5 - 2 + 0.5 = -1
- Apply sigmoid: σ(-1) = 1/(1+e^(-(-1))) ≈ 0.2689
- Answer: ≈ 0.2689
```

## Recommendations for Our Course

### Question Categories to Include
1. **TensorFlow/Keras Implementation Problems** (matching our practical focus)
2. **Multi-layer Network Calculations** (forward/backward pass)
3. **Optimization Algorithm Applications** (gradient descent steps)
4. **Loss Function Computations** (with specific datasets)
5. **Regularization Technique Applications** (dropout, batch norm effects)

### Question Structure Template
```
Scenario: [Specific technical context]
Given: [Concrete numerical values/parameters]
Task: [Calculate/Determine/Apply specific operation]
Show: [Require step-by-step working]
```

### Alignment Requirements
- Use **TensorFlow/Keras syntax** in questions where applicable
- Include **Module 1-2 concepts**: Perceptron calculations, MLP forward pass, gradient descent steps, activation function comparisons
- Ensure questions test **application rather than memorization**
- Must align with Week 1-6 lecture content (no general knowledge)

### Assessment Criteria
- **Marks Distribution**: Likely 1 mark per major step + 1 mark for final answer
- **Expected Student Work**: 40-60 words of explanation plus mathematical steps
- Clear intermediate calculations required
- Final answer with appropriate units/format

## Next Steps
- Create 10 five-mark questions following this computational problem-solving approach
- Ensure all questions use content from delivered Week 1-6 lectures only
- Include TensorFlow/Keras practical implementation aspects
- Maintain step-by-step calculation requirement for full marks

## Key Difference from Previous Work
- 5-mark questions are **computational** (not explanatory like 2-mark)
- Require **mathematical calculations** with specific numerical answers
- Focus on **application of formulas** rather than conceptual explanations
- Must show **detailed working** for full credit