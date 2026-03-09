##### Log-Likelihood function used by the sampler ######

import numpy as np

def loglikelihood_general(params_model, arguments):
    """
    arguments = [data, model, sigmas]
      - data: tuple/list (energy_data, observed_points)
      - model: callable OR list/tuple of two callables [E1_model, M1_model]
    """
    data, model, sigmas, model_args = arguments
    energy_data, points = data

    if callable(model):
        X = model(params_model, energy_data,model_args,1)
        valid_mask = np.isfinite(X)
        obs = X[valid_mask]
    elif isinstance(model, (list, tuple)) and len(model) == 2:
        params_E1 = params_model[:3].copy()
        params_M1 = params_model[3:].copy()
        E1 = model[0](params_E1, energy_data,model_args,len(model))
        M1 = model[1](params_M1, energy_data,model_args,len(model))
        valid_mask = np.isfinite(E1) & np.isfinite(M1)
        obs = E1[valid_mask] + M1[valid_mask]
    else:
        raise ValueError("loglikelihood_general: model must be callable or [E1_model, M1_model]")

    points_valid = points[valid_mask]
    sigmas_valid = sigmas[valid_mask]

    if obs.size == 0:
        return -np.inf

    resid = (points_valid - obs) / sigmas_valid
    ll = -0.5 * (np.sum(resid ** 2) + np.sum(np.log(2 * np.pi * (sigmas_valid ** 2))))
    return float(ll)

