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
```
which guarantee the sampler never proposes invalid scale ≤ 0.

3. Priors
Each block has a correlated multivariate Gaussian prior:
- E1 prior: 3D Gaussian in model-space
- M1 prior: 4D Gaussian in internal-space, plus Jacobian term θ

Priors enforce:
- soft correlations from TALYS output
- hard physical boundaries (positivity, allowed deformation range, etc.)

The priors are dispatched using:
```python
check_prior(prior, params_current, prior_arguments)
```
which automatically performs:
- log-transform for M1
- hard boundary checks
- correct covariance/precision math

4. Likelihood

The log-likelihood is Gaussian:
```python
loglikelihood_ysf(params, arguments)
```
which:
- evaluates E1 and M1 models
- adds them to create the observable
- masks out invalid interpolations
- computes:


5. Proposal Covariance and Optimal Scaling

Proposals use multivariate Gaussian updates:

5.1 Base Covariance (user-defined or diagonal)

You may specify:

- diagonal covariances from step sizes, or

- fully correlated covariances estimated from a pilot run.

5.2 Roberts–Rosenthal Optimal Scaling

For a block of dimension d, the theoretically optimal factor is:


implemented via:
```python
roberts_scaled_cov(cov_base, d)
```
This ensures ~25% acceptance in high dimensions.

5.3 Cholesky Factor

We factor each block covariance:
```python
L = cholesky(cov_scaled)
```
because:

- it guarantees positive-definite Gaussian proposals

- sampling L @ z is stable and fast

- preserves correlations exactly

6. Adaptive Scalar Tuner (burn-in only)

Each block has a ScalarTuner that adjusts proposal scale:
```python
alpha ← alpha × (1 + c·(accept_rate - target))
```
Why we need it:

- prevents the sampler from getting “stuck” if initial step sizes are poor

- steers each block toward its own acceptance target (usually 0.234)

- adaptation stops after burn-in (preserves detailed balance)

The sampler uses:
```python
tuners[b].get_L(L_bases[b])
```
to produce:

7. Metropolis–Hastings Sampler (block-wise)

The core engine:
```python
metropolis(...)
```
Features:
- block-wise updates
- internal-space sampling
- adaptive tuning in burn-in
- mixture proposal for slow M1 log-scale
- model-space storage of samples
- total and per-block acceptance diagnostics

8. Diagnostics

We provide a full FFT-based diagnostic system:
```python
mcmc_diagnostics_array(...)
```
Computes:
- autocorrelation functions
- τ_int (integrated autocorrelation time)
- ESS (effective sample size)
- per-parameter masking of non-finite samples

The FFT approach makes ACF computation extremely fast even for 1e6 samples.


