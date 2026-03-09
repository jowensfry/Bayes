####### This file contains the functions used to calculate the prior for the sampler #############

import numpy as np

def nld_prior(params_vals, arguments):
    # arguments = (params0, inv_cov)   # inv_cov = Σ^{-1}
    params0, inv_cov = arguments

    p_limit = 5
    c_limit = 10

    # ----- HARD BOUNDARIES / HEAVYSIDE FUNCTIONS -----
    # If outside allowed region → impossible → log-prior = -∞
    if (abs(params_vals[0]) > p_limit or
        abs(params_vals[1]) > c_limit):
	#or params_vals[0] + min(data[0]) > max(data[0]) or
        #params_vals[0] + min(data[0]) < 0
        return -np.inf
    
    # ----- GAUSSIAN LOG PRIOR -----
    mu = np.asarray(params_vals) - np.asarray(params0)
    d = len(mu)

    # Quadratic form
    quad = mu @ (inv_cov @ mu)

    # log(det(inv_cov)) in a stable way
    sign, logdet_inv = np.linalg.slogdet(inv_cov)
    if sign <= 0:
        return -np.inf   # inv_cov must be positive definite

    # Log probability of multivariate Gaussian
    log_prior = -0.5 * (d * np.log(2*np.pi) - logdet_inv + quad)

    return log_prior



def E1_prior(params_vals, arguments):
    """
    E1 prior expecting:
      params_vals: array-like [a0, a1, a2]  (model/internal space)
      arguments: (params0, cov_matrix) where cov_matrix is the covariance Σ (not precision)
    Returns log-prior (float) or -np.inf for impossible.
    """
    # TALYS 2.0 parameter  bounds 
    upper_limit = 10.0
    w_low_limit = 0.0
    E_low_limit = -10.0
    f_low_limit = 0.1

    # Unpack and ensure numpy arrays
    params_vals = np.asarray(params_vals, dtype=float)
    params0, cov_matrix = arguments
    params0 = np.asarray(params0, dtype=float)
    cov_matrix = np.asarray(cov_matrix, dtype=float)

    # Make Hard boundaries
    if (np.any(params_vals > upper_limit) or
        params_vals[0] < w_low_limit or
        params_vals[1] < E_low_limit or
        params_vals[2] <= f_low_limit):
        return -np.inf

    # Ensure cov_matrix is square and matches dimension
    d = len(params_vals)
    if cov_matrix.shape != (d, d):
        raise ValueError("E1_prior: cov_matrix shape mismatch")

    # Invert covariance to get precision (Σ⁻¹), with a tiny jitter if needed
    try:
        precision = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # Add tiny jitter to diagonal and try again
        jitter = 1e-8 * np.eye(d)
        precision = np.linalg.inv(cov_matrix + jitter)

    # Quadratic form mu^T Σ⁻¹ mu
    mu = params_vals - params0
    quad = float(mu @ (precision @ mu))

    # log determinant of covariance Σ
    sign, logdet_cov = np.linalg.slogdet(cov_matrix)
    if sign <= 0:
        # If cov_matrix is not Positive Definite, return impossible
        return -np.inf

    # Gaussian log density (correct form using covariance Σ)
    # log p = -0.5 * ( d*log(2π) + logdet(Σ) + mu^T Σ⁻¹ mu )
    log_prior = -0.5 * (d * np.log(2.0 * np.pi) + logdet_cov + quad)
    return float(log_prior)


def M1_prior(params_internal, arguments):
    """
    M1 prior expecting:
      params_internal: [theta0, p1, p2, p3]  (theta0 = log(scale))
      arguments: (params0, cov_matrix) where cov_matrix is covariance Σ in INTERNAL space
    Returns log-prior (float) or -np.inf.
    """
    params_internal = np.asarray(params_internal, dtype=float)
    params0, cov_matrix = arguments
    params0 = np.asarray(params0, dtype=float)
    cov_matrix = np.asarray(cov_matrix, dtype=float)

    # unpack internal
    if len(params_internal) != 4:
        raise ValueError("M1_prior: params_internal must have length 4")

    theta0, p1, p2, p3 = params_internal
    scale = np.exp(theta0)   # model-space value

    # ---- Hard boundaries in MODEL SPACE ----
    if (
        scale <= 0 or scale > 1e-5 or
        p1 < 0 or p1 > 10 or
        abs(p2) > 10 or
        p3 < 0 or p3 > 1.5
    ):
        return -np.inf

    # Ensure covariance shape matches
    d = 4
    if cov_matrix.shape != (d, d):
        raise ValueError("M1_prior: cov_matrix shape mismatch")

    # Invert covariance to get precision
    try:
        precision = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        jitter = 1e-8 * np.eye(d)
        precision = np.linalg.inv(cov_matrix + jitter)

    # Quadratic form in internal space
    mu = params_internal - params0
    quad = float(mu @ (precision @ mu))

    # logdet of covariance Σ
    sign, logdet_cov = np.linalg.slogdet(cov_matrix)
    if sign <= 0:
        return -np.inf

    gauss_log = -0.5 * (d * np.log(2.0 * np.pi) + logdet_cov + quad)

    # Jacobian term (d scale / d theta0) = scale = exp(theta0) → log|J| = theta0
    log_jac = float(theta0)

    return float(gauss_log + log_jac)


def check_prior(prior, params_current, prior_arguments):
    """
    Dispatch wrapper for E1_prior, M1_prior, nld_prior, or combined [E1_prior, M1_prior].
    """

    # ===========================================================
    # 1. SINGLE-MODEL PRIORS
    # ===========================================================
    if callable(prior):

        # -------------------------
        # (a) E1 prior
        # -------------------------
        if prior is E1_prior:
            if len(prior_arguments[0]) > 3:
                raise ValueError("Too many parameters for E1 prior. Expected 3 (w,E,f).")
            return prior(params_current, prior_arguments)

        # -------------------------
        # (b) M1 prior
        # -------------------------
        elif prior is M1_prior:
            params_current_arr = np.asarray(params_current, dtype=float).copy()

            if len(prior_arguments[0]) > 4:
                raise ValueError("Too many parameters for M1 prior (expected 4).")

            # scale > 0 check
            if params_current_arr[0] <= 0:
                return -np.inf

            # convert scale → log(scale)
            params_internal = params_current_arr.copy()
            params_internal[0] = float(np.log(params_current_arr[0]))

            return prior(params_internal, prior_arguments)

        # -------------------------
        # (c) NLD prior
        # -------------------------
        elif prior is nld_prior:
            if len(params_current) != 2:
                raise ValueError("NLD prior expects exactly 2 parameters (p_table, c_table).")
            return prior(params_current, prior_arguments)

        # -------------------------
        # (d) No match
        # -------------------------
        else:
            raise ValueError("Unknown single model type (expected E1_prior, M1_prior, or nld_prior).")


    # ===========================================================
    # 2. MULTI-MODEL PRIOR (E1 + M1)
    # ===========================================================
    elif isinstance(prior, (list, tuple)) and len(prior) == 2:

        if prior[0] is not E1_prior or prior[1] is not M1_prior:
            raise ValueError("Invalid combined prior: must be [E1_prior, M1_prior].")

        E1_current = params_current[:3]
        M1_current = params_current[3:]

        # Unpack the argument structure:
        # prior_arguments = [all_params_mu, [cov_E1, cov_M1]]
        all_mu, cov_list = prior_arguments
        cov_E1, cov_M1 = cov_list

        # Build E1-specific argument tuple
        E1_args = [all_mu[:3], cov_E1]

        # Build M1-specific argument tuple (internal scale)
        M1_model = np.asarray(M1_current, dtype=float)
        if M1_model[0] <= 0:
            return -np.inf
        M1_internal = M1_model.copy()
        M1_internal[0] = float(np.log(M1_model[0]))
        M1_args = [all_mu[3:], cov_M1]

        # Compute both priors
        pri_E1 = prior[0](E1_current, E1_args)
        pri_M1 = prior[1](M1_internal, M1_args)

        return pri_E1 + pri_M1

    # ===========================================================
    # 3. Invalid specification
    # ===========================================================
    else:
        raise ValueError("Invalid prior format.")

