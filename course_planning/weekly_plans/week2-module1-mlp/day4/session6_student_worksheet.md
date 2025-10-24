# Session 6 Student Worksheet
## Data Structures in TensorFlow - Neural Network Building Blocks

**Name:** ________________________  
**Roll Number:** __________________  
**Date:** August 22, 2025  
**Time:** 1 Hour  

---

## 📋 **Pre-Session Checklist**

Before starting, ensure you have:
- [ ] TensorFlow 2.x installed and working
- [ ] Python environment set up (Google Colab recommended)
- [ ] Access to the exercise files: `t3_tensorflow_basic_operations_exercises.py`
- [ ] Completed Session 5 materials on TensorFlow basics

---

## 🎯 **Learning Objectives - Self Assessment**

After completing this session, rate your understanding (1-5 scale):

| Objective | Before | After |
|-----------|--------|-------|
| Create and manipulate TensorFlow tensors | __ | __ |
| Implement basic tensor operations | __ | __ |
| Build neural network forward pass | __ | __ |
| Apply activation functions correctly | __ | __ |
| Debug tensor shape issues | __ | __ |

---

## 🔨 **Practical Exercises - Completion Log**

Mark each exercise as completed and note any issues:

### Exercise 1: Tensor Creation and Manipulation
- [ ] **Completed** | **Time taken:** ______ minutes
- **Key learning:** ________________________________________________
- **Issues faced:** ________________________________________________
- **Code snippet that worked well:**
```python
# Write your best code example here


```

### Exercise 2: Mathematical Operations  
- [ ] **Completed** | **Time taken:** ______ minutes
- **Key learning:** ________________________________________________
- **Issues faced:** ________________________________________________
- **Most useful operation:** ________________________________________

### Exercise 3: Activation Functions
- [ ] **Completed** | **Time taken:** ______ minutes
- **Key learning:** ________________________________________________
- **Which activation gave most interesting results:** ________________
- **Observation about softmax:** ____________________________________

### Exercise 4: Reduction Operations
- [ ] **Completed** | **Time taken:** ______ minutes
- **Key learning:** ________________________________________________
- **Most confusing operation:** _____________________________________
- **Real-world application you can think of:** _____________________

### Exercise 5: Neural Network Forward Pass
- [ ] **Completed** | **Time taken:** ______ minutes
- **Key learning:** ________________________________________________
- **Challenges with matrix dimensions:** ____________________________
- **Output values obtained:** _______________________________________

### Exercise 6: XOR Problem Implementation
- [ ] **Completed** | **Time taken:** ______ minutes
- **Key learning:** ________________________________________________
- **Accuracy achieved:** ____________________________________________
- **Understanding of hidden layer role:** ____________________________

### Exercise 7: Data Preprocessing
- [ ] **Completed** | **Time taken:** ______ minutes
- **Key learning:** ________________________________________________
- **Most useful preprocessing technique:** ___________________________
- **When would you use each technique:** _____________________________

### Exercise 8: Debugging and Error Handling
- [ ] **Completed** | **Time taken:** ______ minutes
- **Key learning:** ________________________________________________
- **Most common error encountered:** ________________________________
- **Best debugging strategy learned:** _______________________________

---

## 🧠 **Conceptual Understanding Questions**

Answer the following questions based on your practical experience:

### 1. Tensor Shapes and Neural Networks
**Question:** Why are tensor shapes so important in neural network operations?

**Your Answer:**
___________________________________________________________________
___________________________________________________________________
___________________________________________________________________

### 2. Variables vs Constants
**Question:** When would you use tf.Variable vs tf.constant in a neural network?

**Your Answer:**
___________________________________________________________________
___________________________________________________________________
___________________________________________________________________

### 3. Matrix Multiplication in Neural Networks
**Question:** Explain how matrix multiplication tf.matmul() relates to neural network computations.

**Your Answer:**
___________________________________________________________________
___________________________________________________________________
___________________________________________________________________

### 4. Activation Functions Purpose
**Question:** Based on your experiments, why do we need activation functions in neural networks?

**Your Answer:**
___________________________________________________________________
___________________________________________________________________
___________________________________________________________________

### 5. XOR Problem Insight
**Question:** How did the hidden layer help solve the XOR problem? What did you observe?

**Your Answer:**
___________________________________________________________________
___________________________________________________________________
___________________________________________________________________

---

## 💡 **Code Troubleshooting Log**

Document any errors you encountered and how you solved them:

### Error 1:
**Error Message:** ___________________________________________________
**Cause:** _________________________________________________________
**Solution:** ______________________________________________________
**Learning:** ______________________________________________________

### Error 2:
**Error Message:** ___________________________________________________
**Cause:** _________________________________________________________
**Solution:** ______________________________________________________
**Learning:** ______________________________________________________

### Error 3:
**Error Message:** ___________________________________________________
**Cause:** _________________________________________________________
**Solution:** ______________________________________________________
**Learning:** ______________________________________________________

---

## 🔍 **Shape Debugging Practice**

Complete this debugging exercise:

```python
# Given these tensors, predict the shapes and identify any errors:
a = tf.constant([[1, 2, 3], [4, 5, 6]])  # Shape: ____________
b = tf.constant([[1, 2], [3, 4], [5, 6]])  # Shape: ____________

# Will this work? Circle: YES / NO
result1 = tf.matmul(a, b)  # Predicted shape: ____________

# Will this work? Circle: YES / NO  
result2 = tf.matmul(b, a)  # Predicted shape: ____________

# How would you fix any issues?
# Solution: ___________________________________________________
```

---

## 🚀 **Extension Activities (Optional)**

If you finish early, try these challenges:

### Challenge 1: Custom Activation Function
- [ ] **Attempted** | **Completed:** Yes / No
Create your own activation function using basic tensor operations.

**Code:**
```python
def my_activation(x):
    # Your custom activation here
    pass
```

### Challenge 2: Batch Processing
- [ ] **Attempted** | **Completed:** Yes / No
Modify the XOR network to handle batches of inputs instead of single examples.

**Observation:** ___________________________________________________

### Challenge 3: Network Visualization
- [ ] **Attempted** | **Completed:** Yes / No
Create a function that prints the shape flow through your network.

**Most interesting finding:** _____________________________________

---

## 📊 **Session Self-Evaluation**

### Overall Session Rating: ⭐⭐⭐⭐⭐ (Circle your rating)

### What went well?
___________________________________________________________________
___________________________________________________________________

### What was challenging?
___________________________________________________________________
___________________________________________________________________

### What would you like to explore further?
___________________________________________________________________
___________________________________________________________________

### How confident do you feel about using TensorFlow for neural networks?
**Scale 1-10:** ______

### One thing you'll remember from this session:
___________________________________________________________________
___________________________________________________________________

---

## 🔄 **Connection to Previous Learning**

### How does this session build on Session 5?
___________________________________________________________________
___________________________________________________________________

### What from Session 4 (MLP/XOR) did you see in practice today?
___________________________________________________________________
___________________________________________________________________

### How do you think this prepares you for Week 3 (Activation Functions)?
___________________________________________________________________
___________________________________________________________________

---

## 📝 **Action Items for Next Session**

Before the next class, I will:
- [ ] Review any concepts I found difficult
- [ ] Practice the exercises I struggled with
- [ ] Read ahead about activation functions
- [ ] Experiment with the optional challenges
- [ ] Prepare questions for the next session

**Specific questions for next session:**
1. ________________________________________________________________
2. ________________________________________________________________
3. ________________________________________________________________

---

## 👥 **Peer Learning (If working in pairs/groups)**

**Partner(s):** _____________________________________________________

### What did you learn from your peer(s)?
___________________________________________________________________
___________________________________________________________________

### What did you help explain to others?
___________________________________________________________________
___________________________________________________________________

### Best collaborative moment:
___________________________________________________________________
___________________________________________________________________

---

## 🎯 **Instructor Use Only**

**Completion Time:** _________ minutes  
**Participation Level:** Excellent / Good / Satisfactory / Needs Improvement  
**Technical Understanding:** Excellent / Good / Satisfactory / Needs Improvement  
**Code Quality:** Excellent / Good / Satisfactory / Needs Improvement  

**Specific Feedback:**
___________________________________________________________________
___________________________________________________________________

**Recommended Focus Areas:**
___________________________________________________________________
___________________________________________________________________

**Grade/Assessment:** _______________

---

**End of Worksheet - Great job on completing Session 6! 🎉**