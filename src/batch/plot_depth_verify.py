"""
Theorem 1 + Corollary 3 verification via the scan-depth survival
function. Designed for Sec. 5.2 (parameter / theory verification),
NOT Sec. 5.3 (end-to-end method comparison).

Three curves only:

    1. Random Orthogonal  -- the *theoretical reference* from
       Theorem 1, namely the partial-sum variance of a Haar rotation
       followed by ADSampling-style early termination. The op 3
       (ADSampling) log is used as the empirical realisation of
       this reference because its rotation is in fact drawn from
       the Haar measure; we relabel it to make the conceptual
       role (a reference, not a competing method) explicit.

    2. FlipFly            -- our data-oblivious method. Should
       overlap with the Random Orthogonal curve on every dataset,
       confirming Theorem 1: SRHT replicates the partial-sum
       variance of a Haar rotation up to a small Walsh-flatness
       factor.

    3. SRHT+PCA           -- our data-aware method. Should sit
       below the Random Orthogonal curve, confirming Corollary 3:
       the variance-sorted coordinate permutation gives
       polynomial-decay headroom on the partial-sum variance.

This plot is *not* meant to rank methods against each other; for
that, see the QPS-Recall figure in Sec. 5.3. DADE, DDCres,
PDScanning are *deliberately* omitted -- they are alternative
methods, not theoretical references.

Usage:
  python3 src/batch/plot_depth_verify.py --K 20,100
  python3 src/batch/plot_depth_verify.py --K 100 --linear

Output (default):
  src/batch/query_performance/depth_verify_K<Ks>.pdf
"""
import argparse
import os
import re
import sys
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
    }
)


DATASETS = [
    "trevi",
    "arxiv-for-fanns-large",
    "XUltra",
    "OpenAI3072",
]

DATASET_TITLES = {
    "trevi": "Trevi",
    "arxiv-for-fanns-large": "Arxiv",
    "XUltra": "XUltra",
    "OpenAI3072": "OpenAI3072",
}

# (label, operator id, color, marker, linestyle).
# Colours are kept exactly in sync with plot_qps_right.py so that
# any cross-figure colour cue (FlipFly = brick red, SRHT+PCA =
# near-black, op-3 = mauve) is consistent across the paper. The
# op-3 (ADSampling) log is the empirical realisation of a Haar
# rotation, so we relabel it as "Random Orth." to mark its
# conceptual role (a theoretical reference, not a method we're
# benchmarking against). The label is shortened from "Random
# Orthogonal" so the legend does not overflow a narrow column.
PALETTE = [
    (r"Random Orth.", 3, "#B07AA1", "o", "--"),
    (r"FlipFly",     18, "#C44E52", "s", "-"),
    (r"FlipFly-D",     20, "#2F2F2F", "^", "-"),
]

HIST_RE = re.compile(r'^"(\d+)"\s*:\s*([\d.]+)')


def parse_hist(path):
    hist = OrderedDict()
    with open(path) as f:
        for line in f:
            m = HIST_RE.match(line.strip())
            if m:
                hist[int(m.group(1))] = float(m.group(2))
    return hist


def survival_curve(hist):
    """S(d_k) = (sum_{j >= k} count[j]) / total."""
    depths = sorted(hist.keys())
    counts = [hist[d] for d in depths]
    total = sum(counts)
    if total <= 0:
        return [], []
    surv = []
    cum_below = 0.0
    for c in counts:
        surv.append((total - cum_below) / total)
        cum_below += c
    return depths, surv


def plot(log_dir, datasets, k_values, ops, out_path, ylog):
    nrows, ncols = len(k_values), len(datasets)
    # Compact per-panel footprint for single-column placement
    # (`\includegraphics[width=\columnwidth]`). Narrow width
    # (1.35 in/panel) keeps the x-axis from stretching across the
    # page; the corresponding aspect is close to square per panel.
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(1.35 * ncols, 2.1 * nrows),
        squeeze=False,
    )

    legend_handles = {}
    found_total = 0

    for row_idx, k_value in enumerate(k_values):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx][col_idx]
            found_here = 0
            # ADSampling rotates at the native dimension while FlipFly and
            # FlipFly-D rotate at the padded one. Normalizing each curve by its
            # own last bin would compare them at different absolute depths, so
            # every curve in a panel uses the widest denominator.
            panel_dmax = 0
            for _lbl, op, _c, _m, _ls in PALETTE:
                if op not in ops:
                    continue
                lg = os.path.join(
                    log_dir,
                    f"hnsw_{dataset}_K{k_value}_operator{op}_search_log.txt",
                )
                if os.path.exists(lg):
                    h = parse_hist(lg)
                    if h:
                        panel_dmax = max(panel_dmax, max(h))
            for label, op, color, marker, ls in PALETTE:
                if op not in ops:
                    continue
                log = os.path.join(
                    log_dir,
                    f"hnsw_{dataset}_K{k_value}_operator{op}_search_log.txt",
                )
                if not os.path.exists(log):
                    print(
                        f"skip {dataset} K={k_value} op={op}: {log} not found",
                        file=sys.stderr,
                    )
                    continue

                hist = parse_hist(log)
                if not hist:
                    print(
                        f"skip {dataset} K={k_value} op={op}: no histogram bins",
                        file=sys.stderr,
                    )
                    continue

                depths, surv = survival_curve(hist)
                if not depths:
                    continue

                d_max = panel_dmax or depths[-1]
                xs = [d / d_max for d in depths]

                markevery = max(1, len(xs) // 10)
                (line,) = ax.plot(
                    xs,
                    surv,
                    color=color,
                    linestyle=ls,
                    lw=1.0,
                    marker=marker,
                    markersize=3.2,
                    markevery=markevery,
                    markeredgecolor="white",
                    markeredgewidth=0.4,
                    label=label,
                )
                legend_handles.setdefault(label, line)
                found_here += 1
                found_total += 1

            # Title split across two lines so it never overflows the
            # narrow per-panel width.
            ax.set_title(
                f"{DATASET_TITLES.get(dataset, dataset)}\n"
                f"$K{{=}}{k_value}$"
            )
            # Symbols match Table~\ref{tab:notation}: \ell is the
            # partial-scan depth and \tilde{d} is the padded dimension.
            ax.set_xlabel(r"$\ell/\tilde{d}$")
            # ylabel only on the leftmost column of each row, so
            # matplotlib places it naturally adjacent to the axis
            # rather than drifting into the figure margin (which is
            # what fig.supylabel does in a multi-row grid).
            if col_idx == 0:
                ax.set_ylabel(r"$\Pr[\text{depth} \geq \ell]$", fontsize=10)
            # Keep the full 0..1 horizontal range but use only three
            # ticks {0, 0.5, 1.0}; the y-axis decade below 1e-2 still
            # carries no data so we clip it.
            ax.set_xlim(0, 1.02)
            ax.set_xticks([0.0, 0.5, 1.0])
            if ylog:
                ax.set_yscale("log")
                ax.set_ylim(1e-2, 1.5)
            else:
                ax.set_ylim(0, 1.05)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if found_here == 0:
                ax.text(
                    0.5,
                    0.5,
                    "no logs",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=14,
                )

    if found_total == 0:
        print("ERROR: no log files yielded a histogram.", file=sys.stderr)
        sys.exit(1)

    handles = [
        legend_handles[lbl]
        for lbl, _, _, _, _ in PALETTE
        if lbl in legend_handles
    ]
    labels = [h.get_label() for h in handles]
    # bbox_inches='tight' crops the saved figure to its content. With
    # an ylabel on the leftmost panels the saved canvas is no longer
    # symmetric in the matplotlib figure coordinate; the legend at
    # x=0.5 ends up visually left-of-centre after cropping. The
    # anchor is shifted slightly right so the legend lines up with
    # the panel-grid centre in the cropped output.
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        handlelength=1.8,
        handletextpad=0.4,
        columnspacing=1.0,
        fontsize=9,
        bbox_to_anchor=(0.54, 0.985),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=0.5, h_pad=1.2)
    fig.subplots_adjust(wspace=0.75, hspace=0.9)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"saved {out_path}")


def parse_int_list(value):
    return [int(x) for x in value.split(",") if x.strip()]


def parse_str_list(value):
    return [x.strip() for x in value.split(",") if x.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log_dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "query_performance"
        ),
    )
    ap.add_argument("--K", default="20,60, 100", help="comma-separated K values")
    ap.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="comma-separated dataset names (panel columns)",
    )
    ap.add_argument(
        "--ops",
        default="3,18,20",
        help="operator IDs (default = Random Orthogonal [op3], "
        "FlipFly [op18], SRHT+PCA [op20])",
    )
    ap.add_argument(
        "--linear",
        action="store_true",
        help="use a linear y-axis instead of log scale",
    )
    ap.add_argument("--out", default=None, help="output PDF path")
    args = ap.parse_args()

    datasets = parse_str_list(args.datasets)
    ops = parse_int_list(args.ops)
    k_values = parse_int_list(args.K)

    out = args.out
    if out is None:
        k_suffix = "_".join(str(k) for k in k_values)
        out = os.path.join(args.log_dir, f"depth_verify_K{k_suffix}.pdf")

    plot(args.log_dir, datasets, k_values, ops, out, ylog=not args.linear)
