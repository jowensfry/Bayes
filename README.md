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
Statist. Sci. 16(4): 351-367 (November 2001). DOI: 10.1214/ss/1015346320

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

9. Recommended Workflow for MCMC Posterior Exploration

This section describes the correct way to use the sampler, following a two-phase workflow:

1. Pilot run (short exploratory chain)

2. Production run (long final chain)

This approach is standard in adaptive Metropolis–Hastings and ensures statistically valid sampling while achieving efficient mixing.

9.1 Phase I — Pilot Run (Exploratory)

The pilot run is a calibration step.
It is not used for inference.

The goals are to:

- map out the approximate region of high posterior density,

- estimate empirical covariances for each block of parameters,

- determine correlations among parameters,

-prepare proposal distributions for the long production run.

The sampler uses only diagonal proposals here (from the step_size vector).
Adaptation is used only to stabilize acceptance.

Example
```python
burn = 10000
num_iter = 50000   # short exploratory run

results_pilot = metropolis(
    burn=burn,
    data=data,
    sigma=error,
    prior=all_prior,
    prior_arguments=prior_arguments_all,
    likelihood=loglikelihood_ysf,
    model=all_model,
    num_iterations=num_iter,
    step_size=all_steps,
    block_idxs=[[0,1,2],[3,4,5,6]],
    cov_bases=None,            # diagonal proposals only
    adapt_interval=5000,
    adapt_window=5000,
    target_accept=0.25,
    log_params=[3],
    random_seed=42,
)
chains_pilot = results_pilot[0]
```
Build empirical block covariances
```python
cov_E1 = np.cov(chains_pilot[:, 0:3].T)
cov_M1 = np.cov(chains_pilot[:, 3:7].T)

cov_bases = [cov_E1, cov_M1]
```
These will be used for the production run.

9.2 Phase II — Production Run (Final Sampling)

In the production run we use:

- empirical blockwise covariances from the pilot,
- Roberts–Rosenthal optimal scaling (2.38²/d) internally,
- blockwise Cholesky proposals,
- a short burn-in with scalar tuning,
- no adaptation during sampling.

Only samples from this phase are used for parameter inference.

Example
```python
burn2 = 20000
num_iter2 = 500000   # long chain

results_final = metropolis(
    burn=burn2,
    data=data,
    sigma=error,
    prior=all_prior,
    prior_arguments=prior_arguments_all,
    likelihood=loglikelihood_ysf,
    model=all_model,
    num_iterations=num_iter2,
    step_size=all_steps,        # still required, but ignored if cov_bases given
    block_idxs=[[0,1,2],[3,4,5,6]],
    cov_bases=cov_bases,        # empirical covariances from pilot
    adapt_interval=500,
    adapt_window=2000,
    target_accept=0.234,
    log_params=[3],
    random_seed=123,
)
chains_final = results_final[0]
```
These samples form the final posterior.

9.3 Why This Two-Phase Method Is Correct

- Avoids invalid adaptation:
  Adaptation only occurs before sampling begins, preserving correct stationary distribution.

- Efficient proposals:
Pilot covariances allow the sampler to account for parameter correlations.

- Optimal scaling:
The sampler internally applies the 2.38²/d scaling factor recommended for multivariate Metropolis.

- Stable long-run behavior:
The production phase is non-adaptive, ensuring valid MCMC.

- Large improvements in ESS and τ_int:
The jump from diagonal proposals to blockwise empirical proposals typically reduces autocorrelation by an order of magnitude.


9.4 Full Workflow Summary

| Phase                | Purpose                 | Proposal Type   | Adaptation       | Output                      |
| -------------------- | ----------------------- | --------------- | ---------------- | --------------------------- |
| **Pilot Run**        | Locate posterior region | Diagonal        | Yes              | Empirical block covariances |
| **Covariance Build** | Learn correlations      | n/a             | n/a              | `cov_bases`                 |
| **Production Run**   | Final inference         | Full covariance | Light tuner only | Posterior samples           |
| **Diagnostics**      | Validate mixing         | n/a             | n/a              | ESS, τ_int, trace plots     |

Use only the production-run samples for inference.

9.5 Optional: Local Mixture Proposals

Some parameters (e.g., the log-scale parameter θ₀) may mix poorly due to highly skewed posterior shapes.

You may optionally add a small-scale local proposal (e.g., 10% of iterations) to help explore narrow directions:
```python
if rng.uniform() < 0.1:
    j = 3  # log-scale parameter
    cand_internal = params_current_internal.copy()
    cand_internal[j] += rng.normal(scale=0.3)
    cand_model = internal_to_model(cand_internal)
```
This “mixture kernel” often significantly improves τ_int for difficult parameters.
