import os
import matplotlib
matplotlib.use("Agg")  # REQUIRED for multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import corner
from sklearn.mixture import GaussianMixture

def load_posterior_csv(csv_file):
    df = pd.read_csv(csv_file)

    if "LogLikelihood" not in df.columns:
        raise ValueError("CSV must contain a 'LogLikelihood' column")

    param_cols = [c for c in df.columns if c != "LogLikelihood"]

    if len(param_cols) == 0:
        raise ValueError("No parameter columns found")

    return df, param_cols


def ensure_image_dir(base_dir="Images"):
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def detect_model_type(param_cols):
    if {"ptable", "ctable"}.issubset(param_cols):
        return "NLD"
    elif len(param_cols) >= 7:
        return "ySF"
    else:
        raise ValueError(
            f"Unrecognized posterior format: {param_cols}"
        )

##########Visualization tools to inspect the posterior distributions######
#In development

# Walker
"""
The walker shows the value of each parameter step by step.
Since parameters explore differently, this can be a bit messy.
Feel free to comment out or reorganize parameters
"""    

def walker(
    csv_file,
    outdir = "Images",
    outfile="walker.pdf",
    figsize=(6, 6),
    dpi=150
):
    """
    Trace plot for all parameters in a CSV posterior file.
    """

    df, param_cols = load_posterior_csv(csv_file)
    outdir = ensure_image_dir(outdir)
    steps = np.arange(len(df))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for col in param_cols:
        ax.plot(steps, df[col].values, label=col, lw=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Parameter value")
    ax.legend(frameon=True, fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, outfile))
    plt.close(fig)

### Chi^2 vs value ###
"""
This will give you an idea of the 1D shape of the parameter.
Shows the value with the lowest chi^2.
Could be prettier, feel free to play with it
"""
def loglikelihood_vs_all_params(
    csv_file,
    outdir="Images",
    log_x_params=("upbendc",),
    figsize=(5, 5),
    dpi=150
):
    """
    Scatter plots of LogLikelihood vs parameters.
    Selected parameters are shown on a log x-axis.
    """

    outdir = ensure_image_dir(outdir)
    df, param_cols = load_posterior_csv(csv_file)

    ll = df["LogLikelihood"].values
    best_idx = ll.argmax()

    for param in param_cols:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        ax.scatter(df[param], ll, s=10, alpha=0.6)
        ax.axvline(df[param].iloc[best_idx], color="r", lw=2)

        if param in log_x_params:
            if (df[param] <= 0).any():
                raise ValueError(f"Cannot use log scale for {param}: non-positive values present")
            ax.set_xscale("log")

        ax.set_xlabel(param)
        ax.set_ylabel("LogLikelihood")
        ax.set_title(f"LogLikelihood vs {param}")

        fig.tight_layout()
        fig.savefig(f"{outdir}/loglike_vs_{param}.pdf")
        plt.close(fig)


### Corner Plot (Fishtank Python3 doesn't have this library) ###
"""
2D heat maps of each parameter with the others.
Shows correlation between parameters as well as 1D histograms.
"""
'''
def corner_plot(
    csv_file,
    outfile="corner.pdf",
    bins=40,
    smooth=1.5,
    smooth1d=1.0,
    figsize=(8, 8)
):
    """
    Corner plot from CSV posterior samples.
    """

    df, param_cols = load_posterior_csv(csv_file)

    data = df[param_cols].values

    fig = corner.corner(
        data,
        labels=param_cols,
        bins=bins,
        smooth=smooth,
        smooth1d=smooth1d,
        label_kwargs={"fontsize": 14},
        hist_kwargs={"linewidth": 2},
    )

    fig.set_size_inches(*figsize)
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)
'''
def gmm_bic_all_pairs_csv(
    csv_file,
    k_min=1,
    k_max=20,
    covariance_type="full",
    random_state=0,
    outdir="Images",
    figsize=(6, 5),
    dpi=150
):
    """
    Perform GMM+BIC clustering once, then plot clusters for all parameter pairs.
    """

    outdir = ensure_image_dir(outdir)
    df, param_cols = load_posterior_csv(csv_file)

    X = df[param_cols].values

    # ---- BIC scan ----
    bics = []
    models = []

    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=random_state
        )
        gmm.fit(X)
        bics.append(gmm.bic(X))
        models.append(gmm)

    best_idx = np.argmin(bics)
    best_k = k_min + best_idx
    best_gmm = models[best_idx]

    print(f"Best number of clusters (lowest BIC) = {best_k}")

    labels = best_gmm.predict(X)
    colors = plt.get_cmap("tab20", best_k).colors

    # ---- Pairwise projections ----
    for x_param, y_param in itertools.combinations(param_cols, 2):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for i in range(best_k):
            mask = labels == i
            ax.scatter(
                df.loc[mask, x_param],
                df.loc[mask, y_param],
                s=12,
                alpha=0.5,
                color=colors[i],
                label=f"Cluster {i}"
            )

        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        ax.set_title(f"GMM clusters: {x_param} vs {y_param}")
        ax.legend(frameon=True, fontsize=8)
        ax.grid(True)

        fig.tight_layout()
        fname = f"gmm_{x_param}_vs_{y_param}.pdf"
        fig.savefig(os.path.join(outdir, fname))
        plt.close(fig)

    return {
        "best_k": best_k,
        "labels": labels,
        "bics": bics,
        "model": best_gmm
    }

def plot_post(
    posterior_csv,
    fit_dataset,
    default_talys_file,
    Oslo_slope,
    Oslo_int,
    *,
    comparison_set = None,
    scale_factor=1,
    sample_frac=0.1,
    ci_levels=(0.68, 0.95),
    ref_adapters=None,
    random_seed=142857,
    plot_components=False,
    # --- convergence options ---
    check_convergence=False,
    batch_size=100,
    max_draws=10000,
    tol_median=0.01,
):
    """
    Plot posterior predictive model with credible intervals.

    Supports:
      - γSF posterior (E1 + M1, ≥7 parameters)
      - NLD posterior (ptable, ctable)

    Model type is inferred from posterior column schema.
    """

    # ------------------------------------------------------------
    # Load experimental data
    # ------------------------------------------------------------
    fit_dataset = pd.read_csv(fit_dataset, header=None, index_col=False)
    half = len(fit_dataset) // 2
    raw_data = np.array(fit_dataset[:half]).flatten()
    raw_error = np.array(fit_dataset[half:]).flatten()

    # ------------------------------------------------------------
    # Load posterior
    # ------------------------------------------------------------
    df, param_cols = load_posterior_csv(posterior_csv)
    chains = df[param_cols].values

    # ------------------------------------------------------------
    # Load TALYS grid (used by both models)
    # ------------------------------------------------------------
    df_talys = pd.read_csv(default_talys_file, header=0)
    

    # ------------------------------------------------------------
    # Detect model type from posterior schema
    # ------------------------------------------------------------
    if {"ptable", "ctable"}.issubset(param_cols):
        model_type = "NLD"
    elif len(param_cols) >= 7:
        model_type = "ySF"
    else:
        raise ValueError(
            f"Unrecognized posterior format with columns: {param_cols}"
        )

    # ------------------------------------------------------------
    # γSF physics models
    # ------------------------------------------------------------
    if model_type == "ySF":
        data = raw_data * scale_factor
        error = raw_error * scale_factor
        energy_grid = df_talys["E"].values
        ysf_E1_talys = df_talys["f(E1)"].values
        ysf_M1_talys = df_talys["f(M1)"].values
        center_E = np.max(energy_grid[np.argmax(ysf_E1_talys)])

        def E1_model(params):
            w, E_shift, scale = params
            mapped_energy = center_E + w * (energy_grid - center_E) + E_shift
            e1 = np.interp(
                mapped_energy,
                energy_grid,
                ysf_E1_talys,
                left=np.nan,
                right=np.nan,
            )
            return scale * e1

        def M1_model(params):
            p0, p1, p2, p3 = params
            return (
                p0
                * np.exp(-p1 * energy_grid)
                * np.exp(-p2 * p3) 
            )

        def eval_model(params):
            return E1_model(params[0:3]) + M1_model(params[3:7])+ ysf_M1_talys

    # ------------------------------------------------------------
    # NLD physics model
    # ------------------------------------------------------------
    else:
        data = raw_data
        error = raw_error
        energy_grid = df_talys["Ex"].values
        energy_nld_talys = energy_grid
        nld_talys = df_talys["NLD"].values

        p_idx = param_cols.index("ptable")
        c_idx = param_cols.index("ctable")

        def eval_model(params):
            p = params[p_idx]
            c = params[c_idx]

            shifted_energy = energy_nld_talys - p
            shifted_nld = np.interp(
                shifted_energy,
                energy_nld_talys,
                nld_talys,
                left=np.nan,
                right=np.nan,
            )

            exp_term = np.exp(
                c * np.sqrt(np.clip(shifted_energy, 0, None))
            )

            return exp_term * shifted_nld

    # ------------------------------------------------------------
    # Energy axis for experimental data
    # ------------------------------------------------------------
    def energy_axis(m, b, n):
        return m * np.arange(n) + b

    # ------------------------------------------------------------
    # Credible interval helper
    # ------------------------------------------------------------
    def posterior_ci(samples, level):
        alpha = 0.5 * (1.0 - level)
        lo = np.percentile(samples, 100 * alpha, axis=0)
        med = np.percentile(samples, 50, axis=0)
        hi = np.percentile(samples, 100 * (1 - alpha), axis=0)
        return lo, med, hi

    # ------------------------------------------------------------
    # Optional convergence-based draw selection
    # ------------------------------------------------------------
    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(len(chains))

    evaluated = []
    prev_median = None
    n_used = 0

    if check_convergence:
        for n in range(batch_size, max_draws + 1, batch_size):
            sel = perm[n_used:n]
            n_used = n

            new_samples = np.array(
                [eval_model(p) for p in chains[sel]]
            )
            evaluated.append(new_samples)

            stacked = np.vstack(evaluated)
            _, med, _ = posterior_ci(stacked, ci_levels[0])

            if prev_median is not None:
                rel = np.abs(med - prev_median) / np.maximum(
                    np.abs(prev_median), 1e-12
                )
                if np.max(rel) < tol_median:
                    break

            prev_median = med

        total_samples = stacked

    else:
        n_draw = int(len(chains) * sample_frac)
        sel = rng.choice(len(chains), n_draw, replace=False)
        total_samples = np.array(
            [eval_model(p) for p in chains[sel]]
        )

    # ------------------------------------------------------------
    # Compute credible intervals
    # ------------------------------------------------------------
    ci_results = {
        level: posterior_ci(total_samples, level)
        for level in ci_levels
    }

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)

    ax.errorbar(
        energy_axis(Oslo_slope, Oslo_int, len(data)),
        data,
        yerr=error,
        fmt="o",
        color="black",
        capsize=5,
        label="Fit data",
    )

    for level in sorted(ci_levels, reverse=True):
        lo, med, hi = ci_results[level]
        alpha = 0.25 if level > 0.8 else 0.45
        ax.fill_between(
            energy_grid,
            lo,
            hi,
            alpha=alpha,
            label=f"{int(level*100)}% credible interval",
        )

    ax.plot(
        energy_grid,
        ci_results[max(ci_levels)][1],
        lw=1.5,
        label="Posterior median",
    )

    if ref_adapters is not None:
        for adapter in ref_adapters:
            adapter(ax)

    ax.set_yscale("log")
    

    if model_type == "ySF":
        if comparison_set != None:
            labels = [r'56Fe LaBr3 A.C.Larsen et al., J. Phys. G: Nucl. Part. Phys. 44, 064005 (2017)',
                      r'56Fe NaI A.C.Larsen et al., J. Phys. G: Nucl. Part. Phys. 44, 064005 (2017)',
                      r'57Fe LaBr3 A.C.Larsen et al., J. Phys. G: Nucl. Part. Phys. 44, 064005 (2017)',
                      r'57Fe NaI A.C.Larsen et al., J. Phys. G: Nucl. Part. Phys. 44, 064005 (2017)',
                      r'64Ni L. Crespo Campo et al., Phys. Rev.94, 044321 (2016)'
                      ]
            colors = ['violet','grey','darkviolet','silver','saddlebrown']
            i=0
            for gsf in comparison_set:
                comp = pd.read_csv(gsf,sep='\t',engine="python",comment="#",header=1)
                print(comp.head())
                plt.errorbar(comp['Eg(MeV)'].values, comp['f(MeV^-3)'].values, 
                        comp['df(MeV^-3)'].values,fmt='d',
                        label= labels[i],
                        color = colors[i])
                i=i+1
        ax.set_ylabel(r"$\gammaSF [MeV^{-3}]", fontsize=16)
        ax.set_xlabel("Energy [MeV]")
        ax.set_title(r"$\gamma$SF posterior predictive")
        ax.set_ylim(1e-9, 1e-6)
        ax.set_xlim(0, 12)
        ax.legend(frameon=True)
        fig.savefig(r"Images/gammaSF_fit.pdf")
        print(r"ySF pdf saved")
    else:
        ax.set_ylabel("Level Density")
        ax.set_xlabel("Energy [MeV]")
        ax.set_title("NLD posterior predictive")
        ax.set_ylim(1e-2,1e5)
        ax.set_xlim(0, 12)
        ax.legend(frameon=True)
        fig.savefig("Images/NLD_fit.pdf")
        print(r"NLD pdf saved")