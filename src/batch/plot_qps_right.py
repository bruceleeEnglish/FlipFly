"""
Plot the right-side figure from plot_qps.py for four datasets.

Edit K near the top to switch the log filename pattern, or override it with
one K value or a comma-separated K list:
  python3 src/batch/plot_qps_right.py --K 100
  python3 src/batch/plot_qps_right.py --K 20,100

Output:
  src/batch/query_performance/qps_right_K<K>.pdf
  src/batch/query_performance/qps_right_K<K1>_<K2>.pdf
"""
import argparse
import os
import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
    }
)


# Default K values plotted as rows of the figure grid. Comma-separated
# string so a single invocation `python3 plot_qps_right.py` produces
# the full multi-K figure used in Sec. 5.3 of the paper. Edit this
# string or pass `--K 20,100` to override.
K = "20,60,100"

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

# Per-dataset Recall@K (x-axis) override. Arxiv's curves all cluster
# in the high-recall region (~0.8 onwards), so zooming the x-axis
# avoids wasting horizontal space on the empty 0--0.8 stretch.
DATASET_XLIM = {
    "arxiv-for-fanns-large": (0.8, 1.0),
}

# method label, operator id, color, marker
PALETTE = [
    ("FDScanning", 0, "#5A5A5A", "x", "--"),
    ("PDScanning", 1, "#4E79A7", "+", "--"),
    ("PDScanning+", 2, "#59A14F", "v", "--"),
    ("ADSampling", 3, "#B07AA1", "o", "--"),
    ("DADE", 4, "#F28E2B", "D", "--"),
    ("DDCres", 5, "#9C755F", "P", "--"),
    ("FlipFly", 18, "#C44E52", "s", "-"),
    ("FlipFly-D", 20, "#2F2F2F", "^", "-"),
]

EF_RE = re.compile(
    r"^ef(\d+):\s+search time =\s+([\d.]+)\s+\[ms\]\s+recall = ([\d.]+)"
)


def parse(path):
    """Returns (list of {ef, t_ms, recall} dicts, rotation_total_ms)."""
    rows = []
    rot_total = 0.0
    with open(path) as f:
        for line in f:
            m = EF_RE.match(line.strip())
            if m:
                rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
            if line.startswith("rotation_time:"):
                rot_total = float(line.split()[1])

    grouped = defaultdict(list)
    for ef, t_ms, recall in rows:
        grouped[ef].append((t_ms, recall))

    out = []
    for ef in sorted(grouped):
        times = sorted(x[0] for x in grouped[ef])
        recalls = sorted(x[1] for x in grouped[ef])
        out.append(
            {
                "ef": ef,
                "t_ms": times[len(times) // 2],
                "recall": recalls[len(recalls) // 2],
            }
        )
    return out, rot_total


def plot(log_dir, datasets, k_values, query_count, ops, out_path):
    fig, axes = plt.subplots(
        len(k_values),
        len(datasets),
        figsize=(5.0 * len(datasets), 2.8 * len(k_values)),
        squeeze=False,
    )

    legend_handles = {}
    total_found = 0

    for row_idx, k_value in enumerate(k_values):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx][col_idx]
            found_for_dataset = 0
            for label, op, color, marker, linestyle in PALETTE:
                if op not in ops:
                    continue

                log = os.path.join(
                    log_dir, f"hnsw_{dataset}_K{k_value}_operator{op}_search_log.txt"
                )
                if not os.path.exists(log):
                    print(f"skip {dataset} K={k_value} op={op}: {log} not found", file=sys.stderr)
                    continue

                rows, rotation_ms = parse(log)
                if not rows:
                    print(
                        f"skip {dataset} K={k_value} op={op}: no ef rows parsed",
                        file=sys.stderr,
                    )
                    continue

                recall = [r["recall"] for r in rows]
                qps = [
                    1000.0 / (r["t_ms"] / query_count + rotation_ms / query_count)
                    for r in rows
                ]
                (line,) = ax.plot(
                    recall,
                    qps,
                    marker=marker,
                    color=color,
                    linestyle=linestyle,
                    lw=1.8,
                    ms=6.5,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label=label,
                )
                legend_handles.setdefault(label, line)
                found_for_dataset += 1
                total_found += 1

            subplot_idx = row_idx * len(datasets) + col_idx
            ax.set_title(
                f"({chr(ord('a') + subplot_idx)}) "
                f"{DATASET_TITLES.get(dataset, dataset)} with K={k_value}"
            )
            ax.set_xlabel(f"Recall@{k_value}")
            ax.set_ylabel("QPS")
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda v, _: ("%.3f" % v).rstrip("0").rstrip("."))
            )
            if dataset in DATASET_XLIM:
                ax.set_xlim(*DATASET_XLIM[dataset])
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            if found_for_dataset == 0:
                ax.text(
                    0.5,
                    0.5,
                    "no logs",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=14,
                )

    if total_found == 0:
        print("ERROR: no log files found.", file=sys.stderr)
        sys.exit(1)

    handles = [legend_handles[label] for label, _, _, _, _ in PALETTE if label in legend_handles]
    labels = [handle.get_label() for handle in handles]
    # Push the legend hard up to the top of the figure and let the
    # plot area extend up almost to the legend bottom: this closes
    # the gap between the legend strip and the first row of subplot
    # titles.
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(handles), 8),
        frameon=False,
        handlelength=3.2,
        handletextpad=0.7,
        columnspacing=2.2,
        bbox_to_anchor=(0.5, 0.995),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=0.5, h_pad=0.8)
    # hspace was 0.34 when each row was 4.2 in tall. After shrinking
    # rows to 2.8 in the absolute gap dropped 1/3 and the next row's
    # title (e.g. "(e) Trevi with K=40") started colliding with the
    # previous row's "Recall@K" xlabel. Bumping hspace restores the
    # same absolute gap.
    fig.subplots_adjust(wspace=0.22, hspace=0.55)
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
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "query_performance"),
    )
    ap.add_argument(
        "--K",
        default=K,
        help="K value or comma-separated K values in log filenames and Recall@K",
    )
    ap.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="comma-separated dataset names, plotted left-to-right",
    )
    ap.add_argument("--M", type=int, default=100, help="number of queries in each log")
    ap.add_argument("--ops", default="0,1,2,3,4,5,18,20", help="comma-separated operator IDs")
    ap.add_argument("--out", default=None, help="output image path")
    args = ap.parse_args()

    datasets = parse_str_list(args.datasets)
    ops = parse_int_list(args.ops)
    k_values = parse_int_list(args.K)
    out = args.out
    if out is None:
        k_suffix = "_".join(str(k) for k in k_values)
        out = os.path.join(args.log_dir, f"qps_right_K{k_suffix}.pdf")

    plot(args.log_dir, datasets, k_values, args.M, ops, out)
