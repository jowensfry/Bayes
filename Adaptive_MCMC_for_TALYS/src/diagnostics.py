############### This file contains all of the diagnostic functions used by the sampler #########
import numpy as np

def _autocorr_fft(x, max_lag=None):
    """Fast, stable autocorrelation (returns acf[0..max_lag] with acf[0]==1)."""
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = x.shape[0]
    if max_lag is None:
        max_lag = n - 1
    # next power-of-two padding (for numerical stability / speed)
    m = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=2*m)
    acf_all = np.fft.irfft(f * np.conjugate(f), n=2*m)[:n]
    acf_all /= acf_all[0]
    return acf_all[: (max_lag + 1)]

def _integrated_time_from_acf(acf, max_lag=None):
    """
    Robust τ estimator:
      τ = 1 + 2 * sum_{t=1..T} rho_t
    where T is chosen by the initial positive sequence / Sokal-like window:
      - find first t where rho_t < 0 and truncate before it
      - otherwise sum up to max_lag
    """
    acf = np.asarray(acf, dtype=float)
    if acf.size <= 1:
        return 1.0
    if max_lag is None:
        max_lag = acf.size - 1
    # Look for first negative autocorrelation (exclude lag 0)
    neg_idx = np.where(acf[1:(max_lag+1)] <= 0)[0]
    if neg_idx.size > 0:
        T = neg_idx[0]  # zero-based into acf[1:]
        tau = 1.0 + 2.0 * np.sum(acf[1:(T+1)])
        return max(1.0, float(tau))
    # No negative up to max_lag — use available values
    tau = 1.0 + 2.0 * np.sum(acf[1:(max_lag+1)])
    return max(1.0, float(tau))

def mcmc_diagnostics_array(chains,
                           log_params=None,
                           burn=0,
                           chi2_col=None,
                           max_lag=None,
                           min_samples=50):
    """
    Compute ACF, τ_int, and ESS for each parameter.

    Parameters
    ----------
    chains : np.ndarray, shape (N_samples, n_params)
        Model-space samples (or post-processed internal space if you prefer).
    log_params : list-like or None
        Indices that are logically 'log' parameters. If provided, you may prefer
        to transform (for tau/trace plots) before diagnostics. This function
        does NOT auto-transform; pass chains already converted if desired.
    burn : int
        Number of initial samples to discard.
    chi2_col : int or None
        If provided, that column index will be excluded from parameter diagnostics.
    max_lag : int or None
        Max lag to compute ACF. Default = N//2.
    min_samples : int
        If a parameter has fewer than this many finite samples, returns trivial results.

    Returns
    -------
    diagnostics : dict with keys:
        'tau_int' : array of τ per parameter (in same order as param_idxs)
        'ess'     : array of ESS per parameter
        'acf'     : list of acf arrays per parameter
        'param_idxs' : list of parameter indices analyzed
        'N' : number of samples used
    """
    chains = np.asarray(chains)
    if chains.ndim != 2:
        raise ValueError("chains must be shape (N, n_params)")
    N_full, n_params = chains.shape
    N = int(N_full - burn)
    if N <= 0:
        raise ValueError("burn >= total samples; nothing to analyze")

    data = chains[burn:, :]
    if max_lag is None:
        max_lag = max(1, N // 2)

    param_idxs = list(range(n_params))
    if chi2_col is not None:
        if chi2_col in param_idxs:
            param_idxs.remove(chi2_col)

    acfs = []
    taus = []
    esses = []
    samples_per_param = []

    for idx in param_idxs:
        x = data[:, idx]
        mask = np.isfinite(x)
        n_finite = int(mask.sum())
        samples_per_param.append(n_finite)
        if n_finite < min_samples:
            # fallback: not enough samples
            acfs.append(np.array([1.0]))
            taus.append(float(N))
            esses.append(float(max(1.0, N)))
            continue

        x2 = x[mask]
        # compute acf
        L = min(max_lag, x2.shape[0] - 1)
        acf = _autocorr_fft(x2, max_lag=L)
        # enforce numerical symmetry / clamp small negatives caused by noise
        acf = np.asarray(acf, dtype=float)
        # integrated time
        tau = _integrated_time_from_acf(acf, max_lag=L)
        ess = float(x2.shape[0]) / float(tau)   # CORRECT formula: ESS = N / tau
        acfs.append(acf)
        taus.append(float(tau))
        esses.append(ess)

    return {
        "tau_int": np.array(taus),
        "ess": np.array(esses),
        "acf": acfs,
        "param_idxs": param_idxs,
        "N": N,
        "samples_per_param": samples_per_param,
    }


def convert_model_to_internal(chains_model, log_params):
    arr = np.asarray(chains_model, dtype=float).copy()
    if log_params is None:
        log_params = []
    for j in log_params:
        arr[:, j] = np.log(arr[:, j])
    return arr
