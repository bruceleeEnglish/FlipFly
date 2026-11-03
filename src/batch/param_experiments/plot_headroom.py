"""§5.2.3 fig:headroom + tab:headroom-table + tab:proxy-agreement

Verifies Corollary 3: a coordinate ordering sorted by per-dim variance
nu_i^2 of the SRHT-rotated data delivers polynomial-decay headroom on
the partial-sum variance.

Pipeline (per dataset):
  1. Stream S<ds>_base.fvecs and accumulate per-coordinate statistics:
         mu_i      = E[(Sx)_i]
         sigma_i^2 = E[(Sx)_i^2]
         nu_i^2    = sigma_i^2 - mu_i^2
  2. Proxy permutation pi      = argsort sigma_i^2 desc (used by SDA)
     Optimal permutation pi^*  = argsort nu_i^2     desc
     Kendall's tau quantifies their agreement.
  3. Fit polynomial decay nu_(i)^2 ~ C * i^{-alpha} on the
     pi^*-sorted spectrum via linear regression in log-log space.
  4. For ell/d_tilde in {1/16, 1/8, 1/4}, compute:
       observed headroom = ell_eff^{pi^*}(ell) / ell
                         = (smallest k such that the top-k slice of
                            the pi^*-sorted nu^2 spectrum captures at
                            least (ell/d_tilde) * total_variance) / ell
       predicted headroom from Corollary 3:
                         = (ell / d_tilde)^{alpha / (1 - alpha)}
  5. Emit
       output/headroom.pdf            -- predicted vs observed bars
       output/headroom_table.tex      -- per-dataset alpha + headroom
       output/proxy_agreement.tex     -- tau and bias/variance ratio

Output:
    output/headroom.pdf
    output/headroom_table.tex
    output/proxy_agreement.tex
"""
import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import kendalltau
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from common import (
    DATASETS,
    DATASET_TITLES,
    DATASET_FILES,
    stream_srht_base,
)


# Three ell-ratios from the paper's TODO outline.
DEFAULT_ELL_RATIOS = [1.0 / 16, 1.0 / 8, 1.0 / 4]


def streaming_per_coord_stats(data_root, dataset, chunk_size=10000, max_n=None):
    """Returns (mu, sigma2, nu2, N, D) for the SRHT-rotated base file."""
    sum_x = None
    sum_x2 = None
    N = 0
    for chunk in stream_srht_base(data_root, dataset,
                                   chunk_size=chunk_size, max_n=max_n):
        c = chunk.astype(np.float64, copy=False)
        if sum_x is None:
            sum_x = c.sum(axis=0)
            sum_x2 = (c * c).sum(axis=0)
        else:
            sum_x += c.sum(axis=0)
            sum_x2 += (c * c).sum(axis=0)
        N += c.shape[0]
    if N == 0:
        raise FileNotFoundError(f"empty or missing SRHT base for {dataset}")
    mu = sum_x / N
    sigma2 = sum_x2 / N
    nu2 = sigma2 - mu * mu
    nu2 = np.clip(nu2, 0.0, None)  # safety, no negative variance
    D = sigma2.shape[0]
    return mu, sigma2, nu2, N, D


def kendall_tau(a, b):
    """Kendall's tau between two arrays. Falls back to a manual O(n log n)
    implementation if scipy isn't installed."""
    if _HAS_SCIPY:
        tau, _ = kendalltau(a, b)
        return float(tau)
    # Manual fallback: count concordant - discordant pairs via merge sort
    # (Knight's algorithm). For our use case n = d_tilde up to 16384 -> ok.
    n = len(a)
    order = np.argsort(a)
    b_sorted = b[order]
    # count inversions in b_sorted -> discordant pairs
    def merge_count(arr):
        if len(arr) <= 1:
            return arr, 0
        mid = len(arr) // 2
        l, c1 = merge_count(arr[:mid])
        r, c2 = merge_count(arr[mid:])
        merged = []
        i = j = c = 0
        while i < len(l) and j < len(r):
            if l[i] <= r[j]:
                merged.append(l[i]); i += 1
            else:
                merged.append(r[j]); j += 1
                c += len(l) - i
        merged += l[i:] + r[j:]
        return merged, c1 + c2 + c
    _, inv = merge_count(b_sorted.tolist())
    total = n * (n - 1) / 2
    return (total - 2 * inv) / total


def top_ell_overlap(pi, pi_star, ell):
    """|Top_ell(pi) cap Top_ell(pi_star)| / ell."""
    ell = int(ell)
    if ell <= 0:
        return 0.0
    s1 = set(pi[:ell].tolist())
    s2 = set(pi_star[:ell].tolist())
    return len(s1 & s2) / ell


def realized_headroom(nu2, perm, ell):
    """ell_eff^perm(ell) / ell = (D / ell) * (sum nu2 at perm's top-ell) / total nu2.

    Definition 4 of the paper. Independent of whether `perm` is the
    optimal pi^* or the proxy pi: simply measures how much of the
    total variance the top-ell coords under `perm` capture, then
    rescales to the headroom interpretation.
    """
    D = len(nu2)
    ell = int(ell)
    if ell <= 0:
        return 0.0
    total = float(nu2.sum())
    if total <= 0:
        return 1.0
    captured = float(nu2[perm[:ell]].sum())
    return (D / ell) * (captured / total)


def fit_alpha(nu2_sorted_desc):
    """Fit nu_(i)^2 ~ C * i^{-alpha} via OLS in log-log space.

    Returns (alpha, C). Uses indices i = 1, 2, ..., D. Filters out
    zero/very-small entries from the tail to keep the regression
    stable."""
    D = len(nu2_sorted_desc)
    idx = np.arange(1, D + 1, dtype=np.float64)
    nu2 = np.asarray(nu2_sorted_desc, dtype=np.float64)
    # Trim the bottom 5% of coordinates (often near machine epsilon)
    keep = nu2 > nu2[0] * 1e-8
    if keep.sum() < 8:
        return 0.5, nu2[0]  # arbitrary fallback
    xs = np.log(idx[keep])
    ys = np.log(np.maximum(nu2[keep], 1e-300))
    # y = log C - alpha * x
    A = np.column_stack([np.ones_like(xs), -xs])
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    log_C, alpha = coef
    return float(alpha), float(np.exp(log_C))


def observed_headroom(nu2_sorted_desc, ell_ratios):
    """Compute ell_eff^{pi^*}(ell) / ell for each ratio.

    Definition: smallest k such that the cumulative variance of the
    top-k coords (by nu^2 desc) is at least (ell / d_tilde) * total
    variance. Headroom = k / ell.
    """
    D = len(nu2_sorted_desc)
    cum = np.cumsum(nu2_sorted_desc)
    total = cum[-1]
    out = {}
    for r in ell_ratios:
        ell = max(1, int(round(D * r)))
        target = (ell / D) * total
        # find smallest k where cum[k-1] >= target
        k = int(np.searchsorted(cum, target) + 1)
        k = min(max(k, 1), D)
        out[r] = k / ell
    return out


def predicted_headroom(alpha, ell_ratios):
    """Corollary 3 closed form: headroom(ell) ~ (ell/d_tilde)^{alpha/(1-alpha)}.

    Caveat: degenerates at alpha = 1. We clip alpha to [0, 0.99] for
    numerical safety."""
    alpha_c = min(0.99, max(0.0, alpha))
    if alpha_c < 1e-6:
        return {r: 1.0 for r in ell_ratios}  # no decay -> no headroom
    exponent = alpha_c / (1 - alpha_c)
    return {r: float(r ** exponent) for r in ell_ratios}


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot(results_by_dataset, ell_ratios, out_path):
    """1 row x len(ell_ratios) cols. Each panel groups all datasets
    on the x-axis with predicted vs observed bars side by side.
    Value labels are placed vertically above each bar to avoid
    horizontal overlap when predicted ~ observed. The legend sits
    on its own row above the panels, away from the value labels."""
    datasets = list(results_by_dataset.keys())
    ncols = len(ell_ratios)
    fig, axes = plt.subplots(
        1, ncols,
        figsize=(3.0 * ncols, 2.4),
        squeeze=False,
    )

    PRED_C = "#B07AA1"   # mauve
    OBS_C  = "#C44E52"   # brick red

    bar_w = 0.40
    x_pos = np.arange(len(datasets), dtype=np.float64)

    for col, ratio in enumerate(ell_ratios):
        ax = axes[0][col]
        pred = [results_by_dataset[ds]["predicted"][ratio] for ds in datasets]
        obs  = [results_by_dataset[ds]["observed"][ratio]  for ds in datasets]

        ax.bar(x_pos - bar_w / 2, pred, width=bar_w,
               color=PRED_C, edgecolor="black", linewidth=0.5,
               label="predicted")
        ax.bar(x_pos + bar_w / 2, obs, width=bar_w,
               color=OBS_C, edgecolor="black", linewidth=0.5,
               label="observed")

        for x, v in zip(x_pos - bar_w / 2, pred):
            ax.text(x, v + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=11, rotation=90)
        for x, v in zip(x_pos + bar_w / 2, obs):
            ax.text(x, v + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=11, rotation=90)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [DATASET_TITLES.get(ds, ds) for ds in datasets],
            fontsize=12, rotation=20, ha="right",
        )
        ax.set_ylim(0, 1.32)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(rf"$\ell/\tilde{{d}}={ratio:.3g}$", fontsize=14)
        ax.tick_params(axis="y", labelsize=12)
        if col == 0:
            ax.set_ylabel(r"$\ell_{\mathrm{eff}}/\ell$", fontsize=14)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=2,
        fontsize=13, frameon=False,
        bbox_to_anchor=(0.5, 1.05),
    )

    fig.tight_layout(w_pad=0.6)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"saved {out_path}")


# --------------------------------------------------------------------------- #
# LaTeX tables
# --------------------------------------------------------------------------- #
def write_headroom_table(results_by_dataset, ell_ratios, out_path):
    datasets = list(results_by_dataset.keys())
    lines = [
        r"\begin{tabular}{@{}l" + "c" * (1 + 2 * len(ell_ratios)) + r"@{}}",
        r"\toprule",
    ]
    head = ["Dataset", r"$\alpha$"]
    for r in ell_ratios:
        head.append(rf"obs.\,$\ell_{{\mathrm{{eff}}}}/\ell$ ({r:.3g})")
        head.append(rf"pred.\,$\ell_{{\mathrm{{eff}}}}/\ell$ ({r:.3g})")
    lines.append(" & ".join(head) + r" \\")
    lines.append(r"\midrule")
    for ds in datasets:
        res = results_by_dataset[ds]
        row = [DATASET_TITLES.get(ds, ds).replace("_", r"\_"),
               f"{res['alpha']:.3f}"]
        for r in ell_ratios:
            row.append(f"{res['observed'][r]:.3f}")
            row.append(f"{res['predicted'][r]:.3f}")
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved {out_path}")


def write_proxy_agreement_table(results_by_dataset, out_path):
    """tab:proxy-agreement: realized headroom ell_eff^pi(ell)/ell of
    the proxy permutation pi used by SDA, alongside the optimal
    ell_eff^{pi^*}(ell)/ell, at the operating depths ell in
    {d_tilde/8, d_tilde/4}. The proxy is operationally useful when its
    realized headroom exceeds the SAD baseline of 1.0 and tracks the
    optimum."""
    datasets = list(results_by_dataset.keys())
    lines = [
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Dataset"
        r" & \multicolumn{2}{c}{$\ell=\tilde{d}/8$}"
        r" & \multicolumn{2}{c}{$\ell=\tilde{d}/4$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}"
        r" & proxy $\pi$ & optimal $\pi^\star$"
        r" & proxy $\pi$ & optimal $\pi^\star$ \\",
        r"\midrule",
    ]
    for ds in datasets:
        res = results_by_dataset[ds]
        row = [
            DATASET_TITLES.get(ds, ds).replace("_", r"\_"),
            f"{res['realized_proxy'][1.0/8]:.3f}",
            f"{res['realized_opt'][1.0/8]:.3f}",
            f"{res['realized_proxy'][1.0/4]:.3f}",
            f"{res['realized_opt'][1.0/4]:.3f}",
        ]
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved {out_path}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_dataset(data_root, dataset, ell_ratios, chunk_size, max_n):
    print(f"  streaming SRHT base of {dataset} (chunk={chunk_size}, max_n={max_n}) ...")
    mu, sigma2, nu2, N, D = streaming_per_coord_stats(
        data_root, dataset, chunk_size=chunk_size, max_n=max_n,
    )
    print(f"  N = {N}, D = {D}")

    # Permutations
    pi      = np.argsort(sigma2)[::-1]
    pi_star = np.argsort(nu2)[::-1]
    tau = kendall_tau(sigma2, nu2)
    max_bias_ratio = float(np.max(mu * mu / np.maximum(sigma2, 1e-30)))
    overlap_ratios = [1.0 / 8, 1.0 / 4]
    overlaps = {r: top_ell_overlap(pi, pi_star, max(1, int(round(D * r))))
                for r in overlap_ratios}
    realized_proxy = {r: realized_headroom(nu2, pi,
                                            max(1, int(round(D * r))))
                      for r in overlap_ratios}
    realized_opt = {r: realized_headroom(nu2, pi_star,
                                          max(1, int(round(D * r))))
                    for r in overlap_ratios}
    print(f"  Kendall tau(pi, pi*) = {tau:.4f}")
    print(f"  max(mu_i^2 / sigma_i^2) = {max_bias_ratio:.4g}")
    for r in overlap_ratios:
        print(f"  ell/D={r:.4g}: Overlap={overlaps[r]:.4f}  "
              f"realized headroom proxy={realized_proxy[r]:.4f}  "
              f"optimal={realized_opt[r]:.4f}")

    # Sort nu^2 desc; fit alpha; compute headrooms
    nu2_sorted = np.sort(nu2)[::-1]
    alpha, _C = fit_alpha(nu2_sorted)
    print(f"  fitted alpha = {alpha:.4f}")

    observed = observed_headroom(nu2_sorted, ell_ratios)
    predicted = predicted_headroom(alpha, ell_ratios)
    for r in ell_ratios:
        print(f"    ell/D = {r:.4g}: observed = {observed[r]:.4f}, "
              f"predicted = {predicted[r]:.4f}")

    return {
        "mu": mu, "sigma2": sigma2, "nu2": nu2, "N": N, "D": D,
        "tau": tau, "max_bias_ratio": max_bias_ratio,
        "overlaps": overlaps,
        "realized_proxy": realized_proxy,
        "realized_opt": realized_opt,
        "alpha": alpha, "observed": observed, "predicted": predicted,
    }


def parse_float_list(s):
    out = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if "/" in token:
            a, b = token.split("/")
            out.append(float(a) / float(b))
        else:
            out.append(float(token))
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/data/dataset")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--ell-ratios",
                    default=",".join(str(r) for r in DEFAULT_ELL_RATIOS),
                    help="comma-separated ell/d_tilde ratios "
                         "(fractions like 1/16 are accepted)")
    ap.add_argument("--chunk-size", type=int, default=20000,
                    help="rows per fvecs read; trade memory for speed")
    ap.add_argument("--max-n", type=int, default=None,
                    help="cap the number of base rows processed "
                         "(useful for sanity checks; default: all rows)")
    ap.add_argument("--out-fig",
                    default=os.path.join(here, "output", "headroom.pdf"))
    ap.add_argument("--out-headroom-table",
                    default=os.path.join(here, "output", "headroom_table.tex"))
    ap.add_argument("--out-proxy-table",
                    default=os.path.join(here, "output", "proxy_agreement.tex"))
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    ell_ratios = parse_float_list(args.ell_ratios)

    os.makedirs(os.path.dirname(args.out_fig) or ".", exist_ok=True)

    results = {}
    for ds in datasets:
        if ds not in DATASET_FILES:
            print(f"WARNING: no DATASET_FILES entry for {ds}", file=sys.stderr)
            continue
        print(f"--- {ds} ---")
        try:
            results[ds] = run_dataset(
                args.data_root, ds, ell_ratios,
                args.chunk_size, args.max_n,
            )
        except FileNotFoundError as e:
            print(f"  SKIP {ds}: {e}", file=sys.stderr)

    if not results:
        print("ERROR: no datasets succeeded", file=sys.stderr)
        sys.exit(1)

    plot(results, ell_ratios, args.out_fig)
    write_headroom_table(results, ell_ratios, args.out_headroom_table)
    write_proxy_agreement_table(results, args.out_proxy_table)


if __name__ == "__main__":
    main()
