"""
Bar chart of offline preprocessing (projection-matrix construction
and database rotation) time per method, per dataset, on a
log y-axis.

This figure fills the "Preproc.\\ (s)" column of
Table~\\ref{tab:system-costs} with empirical evidence and gives the
audience a one-glance view of the offline-cost story: SRHT-based
methods (FWHT-based) finish in tens of seconds; dense-projection
baselines (ADSampling, DADE, DDC) take minutes to hours on the
high-dimensional corpora.

The per-method log files live under
    data/<method>/script/log/<dataset>_<step>.txt
inside the dco-benchmark repository. The file naming and the
timing-keyword used by each preprocessing script differ slightly
across methods, so the parser is per-method.

    method      | step(s)                         | keyword
    ------------+---------------------------------+---------------
    adsampling  | <ds>_randomized.txt             | Elapsed time: X ms
    dade        | <ds>_pca_0.txt                  | Elapsed time: X ms
                | <ds>_getEpsilon_0.txt           | Elapsed time: X ms
                | (sum of the two = total preproc)
    ddc         | <ds>_ddc_pca_generate.txt       | Elapsed time: X ms
                | (note: sampling / groundtruth steps are not timed
                |  in the log, so this is a lower bound)
    srht        | <ds>_srht.txt                   | Elapsed: X ms
    srht_pca    | <ds>_srht_pca.txt               | done in X.Ys

Methods whose log either doesn't exist or did not run to
completion (e.g. SRHT's OpenAI3072 / arxiv-for-fanns-large logs
were KeyboardInterrupted) are rendered as missing bars.

Usage:
  python3 src/batch/plot_preprocessing.py

Output (default):
  src/batch/query_performance/preprocessing_time.pdf
"""
import argparse
import os
import re
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

# (label, data-subdir, color)
# Colours match plot_rotation_only.py / plot_qps_right.py.
METHODS = [
    ("ADSampling", "adsampling", "#B07AA1"),
    ("DADE",       "dade",       "#F28E2B"),
    ("DDCres",     "ddc",        "#9C755F"),
    ("SRHT",       "srht",       "#C44E52"),   # ours
    ("SRHT+PCA",   "srht_pca",   "#2F2F2F"),   # ours
]

# Regexes for the three keyword variants found across logs.
RE_ELAPSED_TIME_MS = re.compile(r"Elapsed time:\s+([\d.]+)\s*ms")
RE_ELAPSED_MS      = re.compile(r"^Elapsed:\s+([\d.]+)\s*ms")
RE_DONE_IN_S       = re.compile(r"done in\s+([\d.]+)\s*s")


def _scan(path, regex):
    """Return the LAST regex match (as float) from `path`, or None."""
    if not os.path.exists(path):
        return None
    last = None
    # Explicit utf-8 with `errors="replace"` so a stray non-UTF8
    # byte (sometimes emitted by upstream training scripts) does
    # not crash the parser on Windows, where the default codec is
    # cp936/gbk.
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = regex.search(line)
            if m:
                last = float(m.group(1))
    return last


def preproc_seconds(data_root, method_dir, dataset):
    """Return total preprocessing time in seconds for one
    (method, dataset) pair, or None if the log is missing or
    incomplete."""
    log_dir = os.path.join(data_root, method_dir, "script", "log")

    if method_dir == "adsampling":
        ms = _scan(os.path.join(log_dir, f"{dataset}_randomized.txt"),
                   RE_ELAPSED_TIME_MS)
        return None if ms is None else ms / 1000.0

    if method_dir == "dade":
        # Total preproc is PCA + epsilon computation.
        pca = _scan(os.path.join(log_dir, f"{dataset}_pca_0.txt"),
                    RE_ELAPSED_TIME_MS)
        eps = _scan(os.path.join(log_dir, f"{dataset}_getEpsilon_0.txt"),
                    RE_ELAPSED_TIME_MS)
        if pca is None or eps is None:
            return None
        return (pca + eps) / 1000.0

    if method_dir == "ddc":
        ms = _scan(os.path.join(log_dir, f"{dataset}_ddc_pca_generate.txt"),
                   RE_ELAPSED_TIME_MS)
        return None if ms is None else ms / 1000.0

    if method_dir == "srht":
        ms = _scan(os.path.join(log_dir, f"{dataset}_srht.txt"),
                   RE_ELAPSED_MS)
        return None if ms is None else ms / 1000.0

    if method_dir == "srht_pca":
        secs = _scan(os.path.join(log_dir, f"{dataset}_srht_pca.txt"),
                     RE_DONE_IN_S)
        return secs  # already seconds

    return None


def plot(data_root, datasets, out_path, ncols=2):
    nrows = (len(datasets) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 4.2 * nrows),
        squeeze=False,
    )

    found_total = 0

    for ds_idx, dataset in enumerate(datasets):
        row_idx, col_idx = ds_idx // ncols, ds_idx % ncols
        ax = axes[row_idx][col_idx]
        labels, values, colors = [], [], []

        for label, method_dir, color in METHODS:
            v = preproc_seconds(data_root, method_dir, dataset)
            if v is None or v <= 0:
                print(
                    f"skip {dataset} {label}: log missing or incomplete",
                    file=sys.stderr,
                )
                continue
            labels.append(label)
            values.append(v)
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
                f"{v:.1f}" if v >= 10 else f"{v:.2f}",
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
        ax.set_ylabel("preprocessing time (s)")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.yaxis.set_minor_locator(
            LogLocator(base=10, subs=tuple(range(2, 10)), numticks=60)
        )
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if values:
            vmin = max(min(values), 1e-3)
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

    # Hide unused axes when len(datasets) does not fill the grid.
    for empty_idx in range(len(datasets), nrows * ncols):
        axes[empty_idx // ncols][empty_idx % ncols].set_visible(False)

    if found_total == 0:
        print("ERROR: no log files yielded a preprocessing time.", file=sys.stderr)
        sys.exit(1)

    fig.tight_layout(w_pad=0.5, h_pad=1.2)
    fig.subplots_adjust(wspace=0.28, hspace=0.55)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"saved {out_path}")


def parse_str_list(value):
    return [x.strip() for x in value.split(",") if x.strip()]


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    default_data_root = os.path.join(repo_root, "data")
    default_out = os.path.join(here, "query_performance", "preprocessing_time.pdf")

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_root",
        default=default_data_root,
        help="directory containing per-method subfolders "
             "(adsampling, dade, ddc, srht, srht_pca)",
    )
    ap.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="comma-separated dataset names (panel order)",
    )
    ap.add_argument(
        "--ncols",
        type=int,
        default=2,
        help="number of subplot columns (defaults to 2 -> 2x2 grid)",
    )
    ap.add_argument("--out", default=default_out, help="output PDF path")
    args = ap.parse_args()

    datasets = parse_str_list(args.datasets)
    plot(args.data_root, datasets, args.out, ncols=args.ncols)
