# Robust CNN Training with VerifAI and Alpha-CROWN  
**CS781: [Formal Methods in Machine Learning — Autumn 2025](https://www.cse.iitb.ac.in/~supratik/courses/cs781/)**

**Instructor:** [*Prof. Supratik Chakraborty*](https://www.cse.iitb.ac.in/~supratik/)  

**Authors:**  
- **[Aman Moon](https://in.linkedin.com/in/aman-moon-b07653256) (22B1216)**  
- **[Arin Weling](https://in.linkedin.com/in/arin-weling-584a39252) (22B1230)**  

---

## Project Overview

This project explores **iterative re-training** of a Convolutional Neural Network (CNN) for **robust car detection** using:

- **CARLA Simulator** (for generating semantic scene variations)  
- **VerifAI Toolkit** (probabilistic scene specification & falsification-driven sampling)  
- **Alpha-CROWN (auto_LiRPA)** (formal verification of neural network robustness)  

The central goal is to show that **data generated from adversarial or low-confidence regions**, discovered via semantic-level falsification, improves **local robustness** of the CNN over iterations.

---

## Objective

1. Train a baseline CNN to classify between number of cars on road scenes (**no-car, one-car or two-cars**) .  
2. Use **VerifAI** to sample scenes with semantic variations:
   - Lighting  
   - Weather  
   - Car distance  
   - Orientation
   - Number of Cars
3. Identify **samples on which the model performs poorly** (semantic counterexamples).  
4. Retrain the CNN on these samples.  
5. After each iteration, use **Alpha-CROWN** to compute local robustness:
   - Radius of the largest **Lₚ-ball** around a given input image  
   - For which the predicted class remains unchanged  
6. Observe **robustness improvement** over iterations.

---

## Why Alpha-CROWN and Not Alpha-Beta-CROWN?

We initially attempted verification using **alpha-beta-CROWN** (branch-and-bound), but:

- It **requires very high VRAM** for CNN verification  
- Even small models caused memory overflows on available GPUs  

Thus, we switched to:

- **Alpha-CROWN**  
- Implemented using the **auto_LiRPA** library  
- Produces **looser bounds**, but scales better  
- Still sufficient for **local robustness metrics**

---

## System Pipeline

### **1. Scene Generation (CARLA → VerifAI)**
VerifAI samples scenes and retrieves rendered images from CARLA.

### **2. Model Evaluation**
Each sampled image is passed through the CNN:
- Misclassified samples  
- Low-confidence predictions  
- Samples with *small robustness bounds*

These are collected for retraining.

### **3. CNN Re-training**
Each iteration adds the newly collected counterexamples to the training set.

### **4. Formal Robustness Estimation (Alpha-CROWN)**
For a chosen norm (we used **L2**) and selected images:
- Compute the **local robustness radius**  
- Track the change over iterations

---

## Outcome

- The CNN should become **more robust** to semantic perturbations.  
- Alpha-CROWN bounds showed a **monotonic increase** across iterations.  
- Falsification-driven data augmentation yields **better generalization** compared to random sampling.

---

## Experimental Notes

- **Chosen Lp norm:** `L2` for simpler bound formulation and stable convergence in Alpha-CROWN.  
- **Model architecture:** A lightweight **CNN**, chosen specifically to keep verification computationally feasible.  
- **Iterative robustness improvement:** After **4-5 re-training iterations**, we observed a clear **increase in local robustness radii** computed by Alpha-CROWN.  
- **Effectiveness of semantic sampling:** VerifAI’s **semantic-level sampler** successfully exposed rare and challenging corner-case scenes (e.g., occluded vehicles, extreme lighting, odd car poses). These counterexamples notably improved generalization and robustness when added to the training set.

---

## References

- **VerifAI Toolkit**  
  Berkeley LearnVerify Group. *VerifAI: A Toolkit for the Design and Analysis of Autonomous Systems*.  
  GitHub: https://github.com/BerkeleyLearnVerify/VerifAI  

- **α/β-CROWN (Alpha-Beta CROWN)**  
  *General and Efficient Robust Neural Network Verification via α, β-CROWN*.  
  GitHub: https://github.com/Verified-Intelligence/alpha-beta-CROWN  

- **auto_LiRPA (Alpha-CROWN / LiRPA Library)**  
  *auto_LiRPA: A Unified Library for Training and Verifying Neural Networks with Certified Robustness*.  
  GitHub: https://github.com/KaidiXu/auto_LiRPA  

- **CARLA Autonomous Driving Simulator**  
  *CARLA: An Open Urban Driving Simulator*.  
  Website: https://carla.org/

---

## Acknowledgements

We thank **Prof. Supratik Chakraborty** for his continuous support and guidance throughout CS781 — *Formal Methods in Machine Learning* (Autumn 2025).  
We also acknowledge the creators and maintainers of **VerifAI**, **CARLA**, **alpha-beta-CROWN**, and **auto_LiRPA**, whose open-source tools made this project possible.
