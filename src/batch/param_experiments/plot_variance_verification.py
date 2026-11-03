"""§5.2.2 fig:variance-verification + tab:walsh-flatness

Empirical vs theoretical variance of the partial-distance estimator
under SRHT projection, validating Theorem 1.

For each (dataset, ell/d_tilde ratio) pair the script

  1. samples M_q (base, query) pairs from disk;
  2. pads x - q to the next power of two (so SRHT applies);
  3. draws R independent SRHT diagonals and projects x - q under
     each one, computing
        Delta_ell = (d_tilde / ell) * sum_{i=1..ell} (S(x-q))_i^2
     for each draw;
  4. takes the empirical variance Var_emp[Delta_ell] over the R
     draws (one variance per pair);
  5. compares to the Haar reference
        Var_Haar[Delta_ell] = 2 * ||x-q||^4 * (d_tilde - ell)
                              / (ell * (d_tilde + 2))
     plotted as a scatter, expected to lie on the diagonal.

The Walsh-flatness parameter eta_ell is reported as the median of
Var_emp / Var_Haar across pairs at each ell ratio.

Output:
    output/variance_verification.pdf
    output/walsh_flatness.tex
"""
import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter

from common import (
    DATASETS,
    DATASET_TITLES,
    DATASET_FILES,
    pad_to_pow2,
    srht_project,
    next_power_of_two,
    read_dataset_base,
    read_dataset_query,
)


PALETTE = [
    ("Random Orth.", "#B07AA1"),   # mauve, theoretical reference
    ("SRHT",         "#C44E52"),   # brick red, our method
]


def _choose_scale_exp(data_min, data_max):
    """Pick base exponent k so that tick mantissae x/10^k mostly
    sit in [1, 100). We anchor on the floor of the geometric mean
    so the brightest range of the data lands at single- or
    two-digit numbers like 1, 2, 5, 10, 20, 50."""
    log_lo = np.log10(max(data_min, 1e-30))
    log_hi = np.log10(max(data_max, 1e-30))
    return int(np.floor((log_lo + log_hi) / 2.0))


def _mantissa_fmt(scale_exp):
    """Return a FuncFormatter that prints x/10^scale_exp as a
    short integer or 1-decimal value. The common 10^scale_exp
    factor is shown once per panel as an in-axes annotation
    (see plot()) rather than repeated on every tick."""
    scale = 10.0 ** scale_exp

    def _fmt(x, _pos):
        if x <= 0:
            return ""
        m = x / scale
        if m >= 10:
            return f"${m:.0f}$"
        if m >= 1:
            return f"${int(round(m))}$" if abs(m - round(m)) < 0.05 else f"${m:.1f}$"
        if m >= 0.1:
            return f"${m:.2f}$"
        return f"${m:.3f}$"

    return _fmt


def haar_variance(diff_sq_norm, ell, D):
    """Var_Haar[Delta_ell] for an isotropic Haar rotation.

    Derivation: partial = sum_{i<=ell} ((S r)_i)^2 with r = x-q follows
    a (||r||^2) * Beta(ell/2, (D-ell)/2) law under Haar; rescaling by
    D/ell yields the formula below.
    """
    return 2.0 * (diff_sq_norm ** 2) * (D - ell) / (ell * (D + 2))


def compute_pair_variances(diff_vecs, R, ell, rng):
    """For each row of diff_vecs (shape (M, D)), draw R independent
    SRHT diagonals, compute Delta_ell under each draw, return the
    sample variance over the R draws."""
    M, D = diff_vecs.shape
    estimators = np.empty((R, M), dtype=np.float64)
    for r in range(R):
        rad = rng.choice([-1, 1], size=D).astype(np.float32)
        Z = srht_project(diff_vecs, rad)
        partial = np.sum(Z[:, :ell].astype(np.float64) ** 2, axis=1)
        estimators[r] = partial * (D / ell)
    return np.var(estimators, axis=0, ddof=1)


def run_dataset(data_root, dataset, M_q, R, ell_ratios, seed):
    """Returns dict[ell_ratio] -> dict('emp': arr, 'th': arr, 'eta': float)."""
    rng = np.random.default_rng(seed)

    print(f"  reading {dataset} base (up to {M_q} rows)...")
    base = read_dataset_base(data_root, dataset, max_n=M_q * 2)
    print(f"  reading {dataset} query...")
    queries = read_dataset_query(data_root, dataset)

    D_in = base.shape[1]
    D = next_power_of_two(D_in)
    print(f"  D_in = {D_in}, padded D = {D}")

    # Pad both and pair up
    M = min(M_q, len(queries), len(base))
    base_idx = rng.choice(len(base), M, replace=False)
    base_sel = base[base_idx]
    queries_sel = queries[:M]
    diff = (base_sel.astype(np.float32) - queries_sel.astype(np.float32))
    diff_padded = pad_to_pow2(diff)

    results = {}
    for ratio in ell_ratios:
        ell = max(1, int(round(D * ratio)))
        print(f"  ell/D = {ratio} (ell = {ell}): {R} SRHT draws over {M} pairs...")
        emp_var = compute_pair_variances(diff_padded, R, ell, rng)
        diff_sq_norm = np.sum(diff_padded.astype(np.float64) ** 2, axis=1)
        th_var = np.array([haar_variance(s, ell, D) for s in diff_sq_norm])

        # eta_ell = median ratio (we use median for robustness to outliers)
        eta = float(np.median(emp_var / np.maximum(th_var, 1e-30)))
        results[ratio] = {"emp": emp_var, "th": th_var, "eta": eta, "D": D, "ell": ell}
    return results


def plot(results_by_dataset, ell_ratios, out_path):
    """3 rows (ell ratios) x 4 cols (datasets), scatter emp vs th."""
    datasets = list(results_by_dataset.keys())
    nrows, ncols = len(ell_ratios), len(datasets)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.6 * ncols, 2.6 * nrows),
        squeeze=False,
    )

    for row, ratio in enumerate(ell_ratios):
        for col, dataset in enumerate(datasets):
            ax = axes[row][col]
            res = results_by_dataset[dataset].get(ratio)
            if res is None:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            emp = res["emp"]
            th = res["th"]
            ax.loglog(th, emp, "o", color="#C44E52", ms=2.2, alpha=0.4,
                      markeredgewidth=0)
            lo = max(1e-30, min(th.min(), emp.min()))
            hi = max(th.max(), emp.max())
            ax.loglog([lo, hi], [lo, hi], "--", color="#444444", lw=1.0,
                      label="$y = x$")

            if row == 0:
                ax.set_title(DATASET_TITLES.get(dataset, dataset), fontsize=13)
            if col == 0:
                # ylabel: full word "empirical variance", no
                # abbreviation, no ratio (the ratio moves into
                # the in-panel annotation block below).
                ax.set_ylabel("empirical variance", fontsize=12)
            if row == nrows - 1:
                ax.set_xlabel("Haar theory variance", fontsize=12)
            ax.tick_params(labelsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Common scale factor: x and y are both variance (same
            # units, same magnitude), so we share one 10^k factor
            # across both axes. Tick labels display the mantissa
            # (1, 2, 5, 10, 20, 50, ...) and the multiplier appears
            # once at the bottom-right corner of the panel. This
            # keeps every panel readable even when the data range
            # is narrow (<1 decade).
            data_lo = max(1e-30, min(th.min(), emp.min()))
            data_hi = max(th.max(), emp.max())
            k_scale = _choose_scale_exp(data_lo, data_hi)
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(
                    LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=15)
                )
                axis.set_major_formatter(FuncFormatter(_mantissa_fmt(k_scale)))
                axis.set_minor_locator(
                    LogLocator(base=10,
                               subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0),
                               numticks=60)
                )
                axis.set_minor_formatter(NullFormatter())
            ax.text(
                0.97, 0.04,
                f"$\\times 10^{{{k_scale}}}$",
                transform=ax.transAxes,
                fontsize=9, ha="right", va="bottom",
                color="#666666",
            )
            # Two-line in-panel annotation: the (dataset, ell)
            # identifier on top, the empirical Walsh-flatness
            # factor on bottom. Replaces the row-label that used
            # to sit inside the ylabel.
            ax.text(
                0.04, 0.96,
                f"$\\ell/\\tilde{{d}} = {ratio}$\n"
                f"$\\eta_\\ell = {res['eta']:.3f}$",
                transform=ax.transAxes,
                fontsize=11, ha="left", va="top",
                color="#222222",
                linespacing=1.15,
            )

    fig.tight_layout(w_pad=0.5, h_pad=0.6)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"saved {out_path}")


def write_walsh_flatness_table(results_by_dataset, ell_ratios, out_path):
    """Emit a LaTeX tabular for tab:walsh-flatness."""
    datasets = list(results_by_dataset.keys())
    lines = []
    lines.append(r"\begin{tabular}{@{}l" + "c" * len(ell_ratios) + r"@{}}")
    lines.append(r"\toprule")
    header = ["Dataset"]
    for r in ell_ratios:
        header.append(rf"$\eta_{{\tilde{{d}}\cdot{r}}}$")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for ds in datasets:
        row = [DATASET_TITLES.get(ds, ds).replace("_", r"\_")]
        for r in ell_ratios:
            res = results_by_dataset[ds].get(r)
            row.append("--" if res is None else f"{res['eta']:.3f}")
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"saved {out_path}")


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
    ap.add_argument("--data-root", default="/data/dataset",
                    help="parent directory containing per-dataset subfolders")
    ap.add_argument("--datasets", default=",".join(DATASETS),
                    help="comma-separated dataset names")
    ap.add_argument("--M-q", type=int, default=200,
                    help="number of (base, query) pairs sampled per dataset")
    ap.add_argument("--R", type=int, default=64,
                    help="number of independent SRHT draws per pair")
    ap.add_argument("--ell-ratios", default="1/8,1/4,1/2",
                    help="comma-separated ell/d_tilde ratios "
                         "(fractions like 1/8 are accepted)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-fig",
                    default=os.path.join(here, "output", "variance_verification.pdf"))
    ap.add_argument("--out-table",
                    default=os.path.join(here, "output", "walsh_flatness.tex"))
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    ell_ratios = parse_float_list(args.ell_ratios)

    os.makedirs(os.path.dirname(args.out_fig) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_table) or ".", exist_ok=True)

    results = {}
    for ds in datasets:
        if ds not in DATASET_FILES:
            print(f"WARNING: no DATASET_FILES entry for {ds}, skipping",
                  file=sys.stderr)
            continue
        print(f"--- {ds} ---")
        try:
            results[ds] = run_dataset(
                args.data_root, ds, args.M_q, args.R, ell_ratios,
                args.seed,
            )
        except FileNotFoundError as e:
            print(f"  SKIP {ds}: {e}", file=sys.stderr)

    if not results:
        print("ERROR: no datasets succeeded", file=sys.stderr)
        sys.exit(1)

    plot(results, ell_ratios, args.out_fig)
    write_walsh_flatness_table(results, ell_ratios, args.out_table)


if __name__ == "__main__":
    main()
