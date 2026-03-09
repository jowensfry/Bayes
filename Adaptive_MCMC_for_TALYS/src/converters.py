#Functions for converting small parameters (e.g. upbendc) in and out of log space 

import numpy as np

# -----------------------
# Scalar tuner
# -----------------------
class ScalarTuner:
    def __init__(self, alpha0=1.0, target=0.234, min_alpha=1e-8, max_alpha=1e8):
        self.alpha = float(alpha0)
        self.target = float(target)
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)

    def update(self, recent_accept_frac):
        if recent_accept_frac <= 0.0:
            self.alpha *= 0.5
        else:
            err = recent_accept_frac - self.target
            factor = 1.0 + 0.3 * err
            factor = np.clip(factor, 0.5, 1.5)
            self.alpha *= factor
        self.alpha = float(np.clip(self.alpha, self.min_alpha, self.max_alpha))
        return self.alpha

    def get_L(self, L_base):
        return np.sqrt(self.alpha) * L_base

# -----------------------
# Transform helpers
# -----------------------
def internal_to_model(params_internal, log_params=None):
    params_internal = np.asarray(params_internal, dtype=float).copy()
    if log_params is None:
        log_params = []
    params_model = params_internal.copy()
    for j in log_params:
        params_model[j] = float(np.exp(params_internal[j]))
    return params_model

def model_to_internal(params_model, log_params=None):
    params_model = np.asarray(params_model, dtype=float).copy()
    if log_params is None:
        log_params = []
    params_internal = params_model.copy()
    for j in log_params:
        if params_model[j] <= 0:
            params_internal[j] = np.log(1e-12)
        else:
            params_internal[j] = float(np.log(params_model[j]))
    return params_internal


