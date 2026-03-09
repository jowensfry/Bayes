# -----------------------#
# Covariance utilities   #
# -----------------------#

import numpy as np

def cov_mtx_general(sigmas, rhos=None):
    """
    General covariance builder for small blocks (len(sigmas) in {2,3,4}).
    - sigmas: array-like of standard deviations
    - rhos: optional matrix or list of off-diagonal correlations (upper triangular)
    If rhos is None, uses default correlations.
    """
    s = np.asarray(sigmas, dtype=float)
    d = len(s)
    if d < 2 or d > 4:
        raise ValueError("cov_mtx_general: supports 2,3,4-d blocks only.")

    # default correlation patterns (keep your previous defaults)
    if d == 2:
        rho12 = 0.7 if rhos is None else rhos[0]
        cov = np.array([[s[0]**2, rho12*s[0]*s[1]],
                        [rho12*s[0]*s[1], s[1]**2]])
    elif d == 3:
        rho12, rho13, rho23 = (0.7, 0.0, 0.0) if rhos is None else rhos
        cov = np.array([
            [s[0]**2, rho12*s[0]*s[1], rho13*s[0]*s[2]],
            [rho12*s[0]*s[1], s[1]**2, rho23*s[1]*s[2]],
            [rho13*s[0]*s[2], rho23*s[1]*s[2], s[2]**2]
        ])
    elif d == 4:
        rho12, rho13, rho14, rho23, rho24, rho34 = (0.0, 0.0, 0.0, 0.3, 0.7, 0.7) if rhos is None else rhos
        cov = np.array([
            [s[0]**2, rho12*s[0]*s[1], rho13*s[0]*s[2], rho14*s[0]*s[3]],
            [rho12*s[0]*s[1], s[1]**2, rho23*s[1]*s[2], rho24*s[1]*s[3]],
            [rho13*s[0]*s[2], rho23*s[1]*s[2], s[2]**2, rho34*s[2]*s[3]],
            [rho14*s[0]*s[3], rho24*s[1]*s[3], rho34*s[2]*s[3], s[3]**2]
        ])
    # Ensure Positive Definite
    eig = np.linalg.eigvalsh(cov)
    if np.any(eig <= 0):
        cov += (1e-8 - np.min(eig) + 1e-12) * np.eye(d)
    return cov

def diag_cov_from_steps(step_sizes, idxs):
    steps = np.asarray([step_sizes[i] for i in idxs], dtype=float)
    return np.diag(steps ** 2)

def roberts_scaled_cov(cov_base, d):
    s2 = (2.38 ** 2) / float(d)
    return s2 * cov_base
