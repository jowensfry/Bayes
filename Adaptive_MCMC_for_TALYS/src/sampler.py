# ------------------------------------#
# Adaptive Metropolis-Hastings sampler# 
# ------------------------------------#

import numpy as np
from src.covarience import roberts_scaled_cov, diag_cov_from_steps
from src.converters import ScalarTuner, internal_to_model, model_to_internal
from src.priors import check_prior

def metropolis_unified(
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
):
    """
    Unified Metropolis sampler.
    This function calls `check_prior(prior, params_model, prior_arguments)` to evaluate priors.
    
    Metropolis-Hastings with block proposals and scalar tuners (burn-in adaptation).

    Parameters
    ----------
    burn : int
        Number of burn-in iterations (adaptation happens during burn).
    data: list or array-like (length = 2) 
		Holds energy points in [0] and measurement in [1]
    sigma: list or array-like (length = 1)
		Holds uncertainty of measurement
    prior: list or array-like (length = 1 or 2)
		Tells sampler which prior to use
    prior_arguments: list or array-like (length = 2)
		Vector that holds initial parameter array and covarience matrix arrays
    likelihood: 
		Which likelihood function to use
    model : list or array-like (length = 1 or 2)
		What models (E1,M1,or NLD) used to manipulate the original tabulated TALYS model
	model_args: list or array-like of dictionary
		Passes base TALYS model into likelihood for parameter manipulation
    num_iterations : int
        Number of iterations to record (after burn).
    step_size : list or array-like (length = n_params)
        Default diagonal step sizes used if cov_bases not supplied.
    block_idxs : list of lists
        Example: [[0,1,2],[3,4,5,6]] to update two blocks sequentially.
        If None, defaults to single block of all parameters.
    cov_bases : list of cov_base matrices corresponding to block_idxs
        If provided, each must be a covariance matrix (not precision).
        If None, diagonal cov from step_size will be used per-block.
    adapt_interval : int
        How often (in iterations) to update the tuner alpha using recent window acceptance.
    adapt_window : int
        Window size used to compute recent acceptance fraction for tuner.
    target_accept : float
        Target acceptance fraction per block for tuning during burn-in.
    log_params : list of int
        Indices of params to sample in log-space (internal representation).
    random_seed : int or None

    Returns
    -------
    params_list : np.ndarray shape (num_iterations, n_params)
        Samples returned in model space (exp applied to log params).
    posterior_list : np.ndarray shape (num_iterations,)
    chi2_list : np.ndarray shape (num_iterations,)
    accept_percent_total : float percentage
    """

    rng = np.random.default_rng(random_seed)

    step_size = np.asarray(step_size, dtype=float)
    initial_parameters = np.asarray(prior_arguments[0], dtype=float)

    n_params = len(initial_parameters)
    if block_idxs is None:
        block_idxs = [list(range(n_params))]
    n_blocks = len(block_idxs)

    if cov_bases is None:
        cov_bases = []
        for idxs in block_idxs:
            cov_bases.append(diag_cov_from_steps(step_size, idxs))
    else:
        if len(cov_bases) != n_blocks:
            raise ValueError("cov_bases must match block_idxs.")

    L_bases = []
    for idxs, cov_base in zip(block_idxs, cov_bases):
        d = len(idxs)
        cov_scaled = roberts_scaled_cov(np.asarray(cov_base, dtype=float), d)
        cov_scaled += 1e-12 * np.eye(d)
        try:
            L_base = np.linalg.cholesky(cov_scaled)
        except np.linalg.LinAlgError:
            cov_scaled += 1e-8 * np.eye(d)
            L_base = np.linalg.cholesky(cov_scaled)
        L_bases.append(L_base)

    if np.isscalar(target_accept):
        targets = [float(target_accept)] * n_blocks
    else:
        targets = list(target_accept)

    tuners = [ScalarTuner(alpha0=0.1, target=target_accept[b]) for b in range(n_blocks)]
    accept_counts_window = [0] * n_blocks
    propose_counts_window = [0] * n_blocks
    accept_counts_total = [0] * n_blocks
    propose_counts_total = [0] * n_blocks

    if log_params is None:
        log_params = []
    params_current_model = initial_parameters.copy()
    for j in log_params:
        if params_current_model[j] <= 0:
            params_current_model[j] = 1e-12
    params_current_internal = model_to_internal(params_current_model.copy(), log_params)

    pri = check_prior(prior, params_current_model, prior_arguments)
    if pri == -np.inf:
        raise ValueError("Initial parameters have -inf prior.")
    posterior_current = likelihood(params_current_model, [data, model, sigma, model_args]) + pri

    params_list = []
    posterior_list = []
    chi2_list = []

    # Burn-in
    total_burn = int(burn)
    print("Burn-in/adapt phase")
    for it in range(total_burn):
        for b, idxs in enumerate(block_idxs):
            d = len(idxs)
            L_propose = tuners[b].get_L(L_bases[b])
            z = rng.normal(size=d)
            cand_internal = params_current_internal.copy()
            cand_internal_block = cand_internal[idxs] + L_propose @ z
            cand_internal[idxs] = cand_internal_block

            cand_model = internal_to_model(cand_internal, log_params)

            pri = check_prior(prior, cand_model, prior_arguments)
            if pri != -np.inf:
                llh = likelihood(cand_model, [data, model, sigma, model_args])
            else:
                llh = -np.inf
            posterior_cand = llh + pri

            ratio = np.exp(posterior_cand - posterior_current) if np.isfinite(posterior_cand) else 0.0
            acceptance_prob = min(1.0, float(ratio)) if not (np.isnan(ratio)) else 0.0

            propose_counts_total[b] += 1
            propose_counts_window[b] += 1

            if rng.uniform() < acceptance_prob:
                params_current_internal = cand_internal
                params_current_model = cand_model
                posterior_current = posterior_cand
                accept_counts_total[b] += 1
                accept_counts_window[b] += 1

        if mixture_local is not None and rng.uniform() < mixture_local.get('prob', 0.0):
            j = mixture_local['index']
            cand_internal = params_current_internal.copy()
            cand_internal[j] += rng.normal(scale=mixture_local.get('scale', 0.1))
            cand_model = internal_to_model(cand_internal, log_params)
            pri = check_prior(prior, cand_model, prior_arguments)
            if pri != -np.inf:
                llh = likelihood(cand_model, [data, model, sigma, model_args])
            else:
                llh = -np.inf
            posterior_cand = llh + pri
            ratio = np.exp(posterior_cand - posterior_current) if np.isfinite(posterior_cand) else 0.0
            acceptance_prob = min(1.0, float(ratio)) if not (np.isnan(ratio)) else 0.0
            if rng.uniform() < acceptance_prob:
                params_current_internal = cand_internal
                params_current_model = cand_model
                posterior_current = posterior_cand

        if (it + 1) % adapt_interval == 0:
            for b in range(n_blocks):
                if propose_counts_window[b] > 0:
                    recent_frac = accept_counts_window[b] / propose_counts_window[b]
                else:
                    recent_frac = 0.0
                tuners[b].update(recent_frac)
                accept_counts_window[b] = 0
                propose_counts_window[b] = 0

    # Sampling phase
    print("Sampling phase (no adaptation)")
    total_iters = int(num_iterations)
    for it in range(total_iters):
        for b, idxs in enumerate(block_idxs):
            d = len(idxs)
            L_propose = tuners[b].get_L(L_bases[b])
            z = rng.normal(size=d)
            cand_internal = params_current_internal.copy()
            cand_internal_block = cand_internal[idxs] + L_propose @ z
            cand_internal[idxs] = cand_internal_block

            if mixture_local is not None and np.random.rand() < mixture_local.get('prob', 0.0):
                j = mixture_local['index']
                cand_internal = params_current_internal.copy()
                cand_internal[j] += np.random.normal(scale=mixture_local.get('scale', 0.1))

            cand_model = internal_to_model(cand_internal, log_params)
            ####### Diagnostics ##########
            '''
            if it < 50:   # print only a little for diagnosis
                print("\nCandidate MODEL params:", cand_model)
            
            pri_test = check_prior(prior, cand_model, prior_arguments)
            if np.isinf(pri_test):
                print("PRIOR returned -inf at iteration", it)
            else:
                print("PRIOR OK", pri_test)
            
            ll_test = likelihood(cand_model, [data, model, sigma, model_args])
            if np.isinf(ll_test) or np.isnan(ll_test):
                print("LIKELIHOOD returned -inf or nan at iteration", it)
            else:
                print("LIKELIHOOD OK", ll_test)
            mE = model[0](cand_model[:3], data[0])
            mM = model[1](cand_model[3:], data[0])
            print("Model E1:", mE)
            print("Model M1:", mM)
            print("Combined:", mE + mM)
            print("Data:", data)
            '''
            pri = check_prior(prior, cand_model, prior_arguments)
            if pri != -np.inf:
                llh = likelihood(cand_model, [data, model, sigma, model_args])
            else:
                llh = -np.inf
            posterior_cand = llh + pri

            ratio = np.exp(posterior_cand - posterior_current) if np.isfinite(posterior_cand) else 0.0
            acceptance_prob = min(1.0, float(ratio)) if not (np.isnan(ratio)) else 0.0

            propose_counts_total[b] += 1

            if np.random.rand() < acceptance_prob:
                params_current_internal = cand_internal
                params_current_model = cand_model
                posterior_current = posterior_cand
                accept_counts_total[b] += 1
                accept_counts_window[b] += 1

        # store
        params_list.append(params_current_model.copy())
        posterior_list.append(posterior_current)
        chi2_list.append(likelihood(params_current_model, [data, model, sigma, model_args]))

    per_block_accept_pct = [
        100.0 * (acc / max(1, prop)) for acc, prop in zip(accept_counts_total, propose_counts_total)
    ]
    total_accepts = sum(accept_counts_total)
    total_props = sum(propose_counts_total)
    overall_accept_pct = 100.0 * (total_accepts / max(1, total_props))

    print("Per-block accept pct:", per_block_accept_pct)
    

    return np.array(params_list), np.array(posterior_list), np.array(chi2_list), overall_accept_pct

