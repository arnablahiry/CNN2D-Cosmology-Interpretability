# Interpreting Cosmological Information from Neural Networks in the Hydrodynamic Universe

This repository contains the implementation and analysis code for the research project  
**“Interpreting Cosmological Information from Neural Networks in the Hydrodynamic Universe”**,  
authored by **Arnab Lahiry**, under the supervision of **Adrian E. Bayer** and **Francisco Villaescusa-Navarro**.  
The work investigates how convolutional neural networks (CNNs) extract cosmological information from simulated universes, particularly within the hydrodynamic simulations of the **CAMELS** project.

---

## Abstract

Neural networks have proven capable of inferring cosmological parameters from simulated matter density fields, even in the presence of complex baryonic processes.  
This project explores which regions of the cosmic web — such as voids, filaments, and halos — provide the most cosmological information to CNNs.  
Through interpretability analyses (Saliency Maps, Integrated Gradients, and GradientSHAP), this work reveals that both high- and low-density regions contribute significantly to parameter estimation.  
The results demonstrate that robust cosmological inference can be achieved with minimal loss in accuracy, even after applying aggressive Fourier- and density-scale cuts, suggesting strategies for astrophysical robustness in future surveys.

---

## Methodology Overview

### Data
This study uses data from the **IllustrisTNG Latin Hypercube (LH) set** of the **CAMELS** project, which includes over 16,000 hydrodynamic and gravity-only simulations.  
Each simulation varies two cosmological parameters, Ωₘ (total matter density) and σ₈ (amplitude of linear fluctuations), as well as four astrophysical feedback parameters:
ASN1, ASN2, AAGN1, and AAGN2.  

From each simulation, two-dimensional total matter density maps of size **256 × 256** at **z = 0** are extracted.  
These maps serve as inputs to the CNN.  

Case-wise **data modification, retraining, and testing** are performed to compare the resulting **R² scores** across different conditions.  
The cases include:  
1. **Density-based modifications**, where pixels above or below specified overdensity or underdensity thresholds are masked.  
2. **Fourier scale cuts**, implemented using the **Pylians3** library ([documentation](https://pylians3.readthedocs.io/en/master/)), where top-hat **kₘₐₓ** cuts are applied at various scales to limit information to specific Fourier modes.



---

### Neural Network Architecture
The CNN follows the architecture proposed in *Villaescusa-Navarro et al. (2021)*, consisting of:
- Several convolutional layers with kernel size 4, stride 2, and padding 1.
- Batch normalization and LeakyReLU activation functions.
- Two fully connected layers producing posterior means and variances for the inferred parameters.

Given an input image **X**, the network predicts the mean (**μᵢ**) and variance (**σᵢ²**) of the marginal posterior distribution **p(θᵢ | X)** for each parameter θᵢ.

$$
\mu_i(X) = \int_{\theta_i} p(\theta_i | X) \, \theta_i \, d\theta_i
$$

$$
\sigma_i^2(X) = \int_{\theta_i} p(\theta_i | X) (\theta_i - \mu_i)^2 d\theta_i
$$

---

### Loss Function

The CNN is trained to minimize a custom loss function that accounts for both prediction accuracy and variance estimation.  
For each parameter **θᵢ**, the loss function is defined as:

$$L = \sum_{i=1}^{6} \log\left( \sum_{j \in \text{batch}} (\theta_{i,j} - \mu_{i,j})^2 \right) + \sum_{i=1}^{6} \log\left( \sum_{j \in \text{batch}} ((\theta_{i,j} - \mu_{i,j})^2 - \sigma_{i,j}^2)^2 \right)$$

This formulation encourages the network to predict both accurate mean values and consistent uncertainty estimates for each parameter.

---

### Training Procedure
- The dataset is divided into **training, validation, and testing** subsets in a 90:5:5 ratio, ensuring that maps from each simulation box are used exclusively in one split.  
- Data augmentation (rotations and flips) enforces invariance under symmetry transformations.  
- Hyperparameters, including learning rate, dropout, and weight decay, are optimized using **Optuna** with the **TPE sampler**.

The model’s accuracy is evaluated using the **R² score**:

$$
R^2 = 1 - \frac{\sum_i (\theta_i - \mu_i)^2}{\sum_i (\theta_i - \bar{\theta})^2}
$$

---

### Interpretability
To interpret which regions of the input maps contribute most to cosmological inference, three interpretability techniques are employed:

1. **Saliency Maps** 
   $$\left(S_i(x) = \frac{\partial f(x)}{\partial x_i}\right)$$
    Measures local sensitivity of the output to each input pixel.

2. **Integrated Gradients** 
   $$\left(IG_i(x) = (x_i - x'_i) \int_0^1 \frac{\partial f(x' + \alpha(x - x'))}{\partial x_i} d\alpha\right)$$
    Provides a global attribution score from a baseline input \( x' \) (e.g., a blank or noisy image) to the actual input.

3. **GradientSHAP** 
   $$\left(\phi_i(x) = \mathbb{E}_{x' \sim p(x')} \left[ (x_i - x'_i) \int_0^1 \frac{\partial f(x' + \alpha(x - x'))} {\partial x_i} d\alpha \right]\right)$$ Extends Integrated Gradients by averaging over multiple baselines, capturing feature interactions and uncertainty.

All interpretability analyses are implemented using the **Captum** library in PyTorch.

---

## Key Results

- CNNs successfully infer cosmological parameters (Ωₘ, σ₈) while marginalizing over astrophysical feedback parameters.  
- Information is extracted from both overdense regions (halos, filaments) and underdense regions (voids).  
- Overdense regions provide **more information per pixel**, while underdense regions contribute **significantly due to their large spatial coherence**.  
- Cosmological information saturates by \( k_{\text{max}} \approx 20\,h/\text{Mpc} \); removing smaller scales causes minimal loss.  
- The network’s predictions remain consistent between hydrodynamic and N-body simulations, indicating robustness to baryonic effects.  
- Minimal degradation in cosmological constraining power is observed even after aggressive Fourier or density cuts.

---

## Citation

If you use this code or results, please cite:

> Lahiry, A., Bayer, A. E., & Villaescusa-Navarro, F. (2025).  
> *Interpreting Cosmological Information from Neural Networks in the Hydrodynamic Universe*. 


---

## Acknowledgements

This work was conducted at:
- Foundation for Research and Technology – Hellas (FORTH)
- University of Crete
- Flatiron Institute, Center for Computational Astrophysics
- Princeton University

Supported by the **TITAN ERA Chair Project (Horizon Europe)** and the **Simons Foundation**.  
CAMELS simulation data provided by the [CAMELS Project](https://www.camel-simulations.org).

---

## Contact

**Author:** Arnab Lahiry  
**Email:** [arnablahiry08@gmail.com](mailto:arnablahiry08@gmail.com)
