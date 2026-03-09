# Adaptive MCMC for TALYS
This repository implements an adaptive, block-wise Metropolis–Hastings sampler for Bayesian calibration of TALYS nuclear models. It supports:

- Nuclear Level Density (NLD) parameter inference
- Gamma Strength Function (ySF) inference (E1 + M1 components)
- Adaptive covariance scaling during burn-in
- Blocked proposals
- Log-space sampling for selected parameters
- Empirical covariance construction from pilot runs
- Diagnostics (ESS, τ_int)
  
## Repository Structure
```
Adaptive_MCMC_for_TALYS/
│
├── main.py
├── posterior_from_talys.py
│
├── Talys_Models/
│   ├── NLD_models/
│   │   └── ld5_59Cu.csv
│   └── ySF_models/
│       └── s10m3un.csv
│
└── src/
    ├── sampler.py
    ├── priors.py
    ├── models.py
    ├── likelihood.py
    ├── covarience.py
    ├── converters.py
    ├── diagnostics.py
    ├── visualization.py
    └── __init__.py
```

## Overview of the Workflow

The typical workflow is:

- Load base TALYS models
- Load experimental data
- Define priors and model parameterization
- Run adaptive Metropolis–Hastings sampling
- Analyze posterior chains
- Generate diagnostics and visualizations

## Reading Base TALYS Models

The script first loads tabulated TALYS predictions.

Example:
```python
df_nld_talys = pd.read_csv("Talys_Models/NLD_models/ld5_59Cu.csv")
energy_nld_talys = df_nld_talys["Ex"].values
nld_talys = df_nld_talys["NLD"].values
```
## Loading Experimental Data

Experimental NLD and gamma strength data are loaded from external files.

The script reconstructs the energy axis using:
```python
energy_axis(m, b, data_size)
```
where
E = m*i + b
This reproduces the energy binning used in the experimental analysis. m and b are taken from the Oslo Method fitting procedure.

## Running the MCMC Sampler

Sampling is performed using metropolis_unified() located in src/sampler.py.
The sampler returns:

- posterior parameter samples
- posterior probabilities
- chi-squared values
- acceptance statistics

metropolis_unified() is a block-adaptive Metropolis–Hastings sampler with proposal tuning during burn-in.
Function signature
```python
metropolis_unified(
    burn,
    data,
    sigma,
    prior,
    prior_arguments,
    likelihood,
    model,
    model_args,
    num_iterations,
    step_size,
    block_idxs=None,
    cov_bases=None,
    adapt_interval=1000,
    adapt_window=1000,
    target_accept=0.234,
    log_params=None,
    mixture_local=None,
    thin=1,
    random_seed=None,
)
```
### Parameter Descriptions
#### burn
burn : int
Number of burn-in iterations.
During burn-in:
- proposal scale is adaptively tuned
- samples are not stored
  
After burn-in, adaptation stops and the chain becomes stationary.

#### data

data

data : list

Input experimental data used by the likelihood function.

Structure:

data = [energy_values, measurements]

#### sigma

sigma : array-like

Measurement uncertainties used in the likelihood calculation.

These are used when computing the Gaussian likelihood or χ² statistic.

#### prior

prior : list

Specifies which prior functions to use.

Examples:

- nld_prior
- E1_prior
- M1_prior

The sampler calls check_prior() to evaluate the log-prior.


#### prior_arguments

prior_arguments : list

Contains:

[initial_parameters, covariance_matrices]

These are passed to the prior evaluation functions.

#### likelihood

likelihood : function

The log-likelihood function used to evaluate candidate parameters.

In this repository the primary function is:

loglikelihood_general()

which computes the likelihood of the data given the model parameters.


#### model

model : list

Functions used to transform the base TALYS models using candidate parameters.

Examples:

- nld_model
- E1_model
- M1_model

#### model_args

model_args : list of dictionaries

Contains the base TALYS model arrays that the model functions modify.
This allows the likelihood function to compute predictions for a given parameter set.  

#### num_iterations

num_iterations : int

Number of MCMC samples stored after burn-in.

#### step_size

step_size : array

Initial proposal step sizes for each parameter.
If no covariance matrices are provided, these are used to construct a diagonal proposal covariance.

#### block_idxs

block_idxs : list of lists

Defines parameter blocks for joint updates.

Example:

[[0,1,2],[3],[4],[5,6]]

This means:

- parameters 0–2 updated together
- parameter 3 updated alone
- parameter 4 updated alone
- parameters 5–6 updated together

Blocking helps sampling efficiency when parameters are correlated.

#### cov_bases

cov_bases : list of covariance matrices

Optional covariance matrices used for block proposals.
If not provided, they are constructed from step_size.
Each block covariance is scaled using:

roberts_scaled_cov()

which implements the optimal scaling rule for Metropolis algorithms.

#### adapt_interval

adapt_interval : int

Number of iterations between proposal scaling updates.
Acceptance statistics are collected over this interval.

#### target_accept
target_accept : float

Target acceptance probability.

Typical theoretical values:

  | dimension	| optimal acceptance|
  |-----------|-------------------|
  | 1D	| ~0.44 |
  | multi-dim	| ~0.234|

The sampler defaults to 0.234.

#### log_params

log_params : list

Indices of parameters sampled in log space.

This is used for parameters that:

- must remain positive
- vary over many orders of magnitude

Conversion is handled automatically by:

model_to_internal()
internal_to_model()


#### mixture_local

Optional mechanism to occasionally apply local perturbations to a specific parameter.

Structure:
```python
{
    "index": parameter_index,
    "prob": probability,
    "scale": standard_deviation
}
```
This can improve mixing for parameters with long autocorrelation times.


#### random_seed

random_seed : int

Seed for the random number generator to ensure reproducible sampling.

### Sampler Output
The function returns:

- params_list
- posterior_list
- chi2_list
- accept_percent_total

#### params_list

shape = (num_iterations, n_parameters)

Posterior samples in model space.

#### posterior_list

Posterior probability values for each sample.

#### chi2_list

Chi-squared statistic corresponding to each sample.

#### accept_percent_total

Overall acceptance rate across the chain.


### Modules in src/
Each module in src/ provides a specific component of the inference framework.

*src/sampler.py*

Implements the adaptive block Metropolis–Hastings algorithm.

Responsibilities:

- Proposal generation
- Block updates
- Burn-in adaptation
- Acceptance testing
- Log-space parameter handling

*src/models.py

Defines parameterized transformations of the base TALYS models.

Examples include:

- Nuclear level density modification
- E1 gamma strength modification
- M1 gamma strength modification

These functions take model parameters and produce predicted observables.

#### src/priors.py

Defines prior probability distributions for the model parameters.

Examples:

nld_prior()
E1_prior()
M1_prior()

Also contains:

check_prior()

which evaluates the combined prior probability.

#### src/likelihood.py

Implements likelihood evaluation.

The main function:

loglikelihood_general()

computes the log-likelihood using the model prediction and experimental data.

#### src/covarience.py

Utility functions for constructing proposal covariance matrices.

Includes:

cov_mtx_general()
roberts_scaled_cov()
diag_cov_from_steps()

These help generate stable proposal distributions.

#### src/converters.py

Handles parameter transformations between different representations.

Includes:

model_to_internal()
internal_to_model()
ScalarTuner

ScalarTuner is responsible for adjusting the proposal scale during burn-in.

#### src/diagnostics.py

Provides MCMC convergence diagnostics.

Includes:

mcmc_diagnostics_array()

which estimates:

- autocorrelation time
- effective sample size (ESS)

#### src/visualization.py

Contains plotting tools for posterior analysis and model comparison.

These may include plots such as:

- posterior distributions
- parameter correlations
- model fits to data

### TALYS Model Inputs

Base TALYS model tables are stored in:

Talys_Models/

These files contain tabulated predictions that are modified during sampling.

### Diagnostics

After sampling, diagnostics can be computed using:

mcmc_diagnostics_array()

Key metrics:

- Effective Sample Size (ESS)
- Integrated autocorrelation time

Large ESS values (>1000) indicate good sampling efficiency.

### Summary

This repository implements a flexible adaptive MCMC framework designed for Bayesian calibration of TALYS nuclear models.

Key features include:

- Block Metropolis–Hastings sampling
- Adaptive proposal scaling
- Log-space parameter support
- Modular likelihood and model functions
- Built-in convergence diagnostics

The modular structure allows the framework to be extended to additional nuclear models or experimental datasets.
