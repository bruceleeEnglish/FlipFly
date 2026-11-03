"""
Bar chart of per-query rotation (projection) time, log y-axis.

This isolates the projection-cost story: ADSampling / DADE /
DDCres carry an O(d_tilde^2) matrix-vector projection, SRHT and
SRHT+PCA carry an O(d_tilde log d_tilde) FWHT. On a log y-axis
the gap is two-plus decades and unmistakable.

Methods with rotation_time == 0 (FDScanning, PDScanning,
PDScanning+) are skipped by default since they do not project.

Per-query rotation cost is computed as
    rotation_per_query = rotation_total / M
to match the convention used in plot_qps_right.py.

rotation_time does not depend on K (it is the cost of the data-
independent query_transform step that runs before the ANN search,
see hnsw_search.cpp:139--146 in the C++ benchmark), so we plot a
single K-slice. --K only selects which log file to read; values
are constant across K to within measurement noise.

Usage:
  python3 src/batch/plot_rotation_only.py
  python3 src/batch/plot_rotation_only.py --K 20

Output (default):
  src/batch/query_performance/rotation_only_K<K>.pdf
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 13,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
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

# (label, operator id, color) - methods that actually project (rotation > 0).
# Palette: Seaborn "deep" for the three baselines (cool, restrained) and a
# two-tone red+near-black for the two SRHT methods so they pop visually.
PALETTE = [
    ("ADSampling",  3, "#4C72B0"),   # steel blue
    ("DADE",        4, "#55A868"),   # forest green
    ("DDCres",      5, "#937860"),   # taupe / muted brown
    ("SRHT-AD",    18, "#C44E52"),   # brick red  (ours)
    ("SRHT-DA",    20, "#2F2F2F"),   # near-black (ours)
]


def read_rotation_total(path):
    with open(path) as f:
        for line in f:
            if line.startswith("rotation_time:"):
                return float(line.split()[1])
    return None


def plot(log_dir, datasets, k_value, query_count, ops, out_path, ncols=2):
    # Rotation time is the cost of the data-independent
    # query_transform step (hnsw_search.cpp:139--146 in the C++
    # benchmark) that runs *before* the ANN search, so it does not
    # depend on K. Empirically the rotation_time field is constant
    # across K to within measurement noise (<10% for matrix-vector
    # baselines, <2% for FWHT-based methods). We therefore plot only
    # a single K-slice; --K controls which log file is read.
    #
    # ncols defaults to 2 so a 4-dataset run produces a 2x2 grid
    # that fits naturally inside a double-column figure* float.
    nrows = (len(datasets) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 4.2 * nrows),
        squeeze=False,
    )

    found_total = 0

    for ds_idx, dataset in enumerate(datasets):
        row_idx = ds_idx // ncols
        col_idx = ds_idx % ncols
        ax = axes[row_idx][col_idx]
        labels, values, colors = [], [], []

        for label, op, color in PALETTE:
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

            rot_total = read_rotation_total(log)
            if rot_total is None:
                continue
            rot_per_q = rot_total / query_count
            if rot_per_q <= 0:
                print(
                    f"skip {dataset} K={k_value} op={op}: rotation == 0",
                    file=sys.stderr,
                )
                continue

            labels.append(label)
            values.append(rot_per_q)
            colors.append(color)
            found_total += 1

        x = list(range(len(labels)))
        ax.bar(
            x,
            values,
            color=colors,
            edgecolor="black",
            linewidth=0.6,
            width=0.72,
        )

        for xi, v in zip(x, values):
            ax.text(
                xi,
                v * 1.18,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=12,
            )

        ax.set_title(
            f"({chr(ord('a') + ds_idx)}) "
            f"{DATASET_TITLES.get(dataset, dataset)}"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("per-query rotation (ms)")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10)), numticks=60))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if values:
            vmin = max(min(values), 1e-6)
            vmax = max(values)
            ax.set_ylim(vmin * 0.3, vmax * 6.0)

        if not labels:
            ax.text(
                0.5,
                0.5,
                "no logs",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
            )

    # Hide any unused axes (e.g. when len(datasets) is not a multiple
    # of ncols).
    for empty_idx in range(len(datasets), nrows * ncols):
        axes[empty_idx // ncols][empty_idx % ncols].set_visible(False)

    if found_total == 0:
        print("ERROR: no log files yielded a rotation_time.", file=sys.stderr)
        sys.exit(1)

    fig.tight_layout(w_pad=0.5, h_pad=1.2)
    fig.subplots_adjust(wspace=0.28, hspace=0.55)
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
    ap.add_argument(
        "--K",
        type=int,
        default=100,
        help="K value of the log file to read. rotation_time is "
        "essentially constant across K (it is the cost of the "
        "data-independent query_transform step), so any K with a "
        "log present works; we default to K=100 for consistency "
        "with the rest of the paper.",
    )
    ap.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="comma-separated dataset names (panel columns)",
    )
    ap.add_argument("--M", type=int, default=100, help="number of queries per log")
    ap.add_argument(
        "--ops",
        default="3,4,5,18,20",
        help="operator IDs (default = methods with rotation > 0)",
    )
    ap.add_argument("--out", default=None, help="output PDF path")
    args = ap.parse_args()

    datasets = parse_str_list(args.datasets)
    ops = parse_int_list(args.ops)

    out = args.out
    if out is None:
        out = os.path.join(args.log_dir, f"rotation_only_K{args.K}.pdf")

    plot(args.log_dir, datasets, args.K, args.M, ops, out)
