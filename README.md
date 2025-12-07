# MCMC Sampler for E1 + M1 Gamma Strength Function Fitting

This repository implements a fully custom **block-wise Metropolis–Hastings sampler**  
for extracting the E1 and M1 components of the γ-ray strength function (ySF),  
combining data-driven likelihoods with TALYS theoretical priors.

The sampler handles:

- **7 correlated physical parameters**
- **block updates**:  
  - Block 1 → E1 parameters (3D)  
  - Block 2 → M1 parameters (4D)
- **log-transform for scale parameters**
- **hard physical boundaries**
- **multivariate Gaussian priors**
- **Roberts–Rosenthal optimal proposal scaling**
- **Cholesky-based correlated Gaussian proposals**
- **adaptive scalar tuning during burn-in**
- **optional mixture proposals for slow directions**
- **high-quality FFT-based autocorrelation diagnostics**

This README emphasizes the **main intended use**:  
### ➤ **Full E1+M1 combined sampling.**  
Testing-only modes (pure E1 or M1 fits) are described briefly at the end.

---

# 1. Overview of the Model

We fit **7 parameters** describing the E1 and M1 components of the γSF:

| Group | Param | Meaning |
|------|-------|---------|
| **E1 block (3 params)** | `a0` | width scaling (dimensionless) |
| | `a1` | energy shift |
| | `a2` | amplitude scaling |
| **M1 block (4 params)** | `scale` | upbend overall scale (sampled in log-space) |
| | `p1` | exponential energy damping |
| | `p2` | deformation sensitivity |
| | `p3` | deformation magnitude |

The **model prediction** is

\[
y(E) = E1(E; a_0,a_1,a_2) + M1(E; \text{scale},p_1,p_2,p_3)
\]

Each component uses interpolation of TALYS output plus parametric corrections.

---

# 2. Internal vs. Model Parameter Spaces

To ensure positivity of sensitive parameters (like M1 scale), the sampler works in an **internal space**, where:

- Most parameters: **identity transform**
- M1 scale:  
  - model: `scale`  
  - internal: `θ = log(scale)`

We provide two helpers:

```python
internal_to_model(params_internal)
model_to_internal(params_model, log_params=[3])
