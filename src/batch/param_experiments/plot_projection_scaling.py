"""§5.2.1 fig:projection-scaling

Plot the projection-scaling figure from timings produced by
`cpp/bench_projection` (the C++ microbenchmark). The C++ binary
reuses the FWHT implementation from `DCO/L2/SRHTADSampling.hpp` and
the dense matvec path from `DCO/L2/ADSampling.hpp`, so the numbers
plotted here reflect the actual code that runs at search time.

Recommended workflow:
    cd cpp && bash build.sh
    ./bench_projection           > ../output/projection_timings.csv
    cd ..
    python3 plot_projection_scaling.py \\
        --csv output/projection_timings.csv

A Python-only `--bench-python` fallback is provided for sanity
checks; it uses numpy's BLAS for matvec and a vectorised numpy
FWHT, so absolute timings (and the FWHT/matvec ratio) differ from
the C++ run. Do NOT use the Python fallback for paper numbers.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLOR_FWHT   = "#C44E52"   # brick red, ours (SRHT-AD / SRHT+PCA)
COLOR_MATVEC = "#B07AA1"   # mauve, ADSampling-family


# --------------------------------------------------------------------------- #
# CSV (preferred) and python-fallback timing
# --------------------------------------------------------------------------- #
def load_csv(path):
    dims, fwht, mv = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dims.append(int(row["dim"]))
            fwht.append(float(row["fwht_ms"]) / 1000.0)
            mv.append(float(row["matvec_ms"]) / 1000.0)
    return dims, fwht, mv


def python_fallback_time(dims, repeats=200, warmup=20):
    """Python-only timing of FWHT vs matvec. Provided for local sanity
    checks; not for paper numbers (see module docstring)."""
    from common import fwht_inplace  # local import to keep CSV path standalone

    fwht_t, mv_t = [], []
    rng = np.random.default_rng(0)
    for d in dims:
        rad = rng.choice([-1.0, 1.0], size=d).astype(np.float32)
        # FWHT
        x = rng.standard_normal(d).astype(np.float32)
        for _ in range(warmup):
            y = (x * rad).reshape(1, -1); fwht_inplace(y)
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            y = (x * rad).reshape(1, -1); fwht_inplace(y)
            ts.append(time.perf_counter() - t0)
        fwht_t.append(float(np.median(ts)))
        # matvec
        H = rng.standard_normal((d, d)).astype(np.float32)
        for _ in range(warmup):
            y = H @ x
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            y = H @ x
            ts.append(time.perf_counter() - t0)
        mv_t.append(float(np.median(ts)))
    return list(dims), fwht_t, mv_t


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def write_latex_table(dims, fwht_times, matvec_times, out_path):
    """Emit a LaTeX tabular reporting the same numbers as the figure.

    Two-decimal-place precision is uneven across magnitudes here
    (FWHT < 0.1 ms, matvec up to tens of ms), so columns are
    formatted independently: FWHT keeps four decimals (so the
    smallest values stay legible) while matvec uses three.
    """
    lines = []
    lines.append(r"\begin{tabular}{@{}rrrr@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"$\tilde{d}$ & FWHT (ms) & matvec (ms) & ratio \\"
    )
    lines.append(r"\midrule")
    for d, tf, tm in zip(dims, fwht_times, matvec_times):
        tf_ms = tf * 1000.0
        tm_ms = tm * 1000.0
        ratio = tm / tf if tf > 0 else float("inf")
        lines.append(
            f"{d} & {tf_ms:.4f} & {tm_ms:.3f} & "
            f"{ratio:.0f}$\\times$ \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"saved {out_path}")


def plot(dims, fwht_times, matvec_times, out_path):
    # Bump font sizes globally so the figure stays readable when LaTeX
    # scales it down to \columnwidth.
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })
    fig, ax = plt.subplots(figsize=(5.6, 3.6))

    # Each method gets paired curves: a thin dotted reference line at
    # the theoretical slope (anchored at the smallest measured d) and
    # a thick markered empirical line. Both share a colour. Side-by-
    # side this makes the "empirical follows theory" claim immediate.
    c1 = fwht_times[0] / (dims[0] * np.log2(dims[0]))
    theor_fwht = [c1 * d * np.log2(d) for d in dims]
    c2 = matvec_times[0] / (dims[0] ** 2)
    theor_matvec = [c2 * d ** 2 for d in dims]

    # Empirical FWHT
    ax.loglog(
        dims, fwht_times,
        "o-", color=COLOR_FWHT, lw=2.2, ms=7,
        markeredgecolor="white", markeredgewidth=0.8,
        label="FWHT (empirical)",
        zorder=3,
    )
    # Theoretical FWHT slope
    ax.loglog(
        dims, theor_fwht,
        ":", color=COLOR_FWHT, lw=1.6, alpha=0.75,
        label=r"$\Theta(\tilde{d}\log\tilde{d})$",
        zorder=2,
    )
    # Empirical matvec
    ax.loglog(
        dims, matvec_times,
        "s-", color=COLOR_MATVEC, lw=2.2, ms=7,
        markeredgecolor="white", markeredgewidth=0.8,
        label="matvec (empirical)",
        zorder=3,
    )
    # Theoretical matvec slope
    ax.loglog(
        dims, theor_matvec,
        ":", color=COLOR_MATVEC, lw=1.6, alpha=0.75,
        label=r"$\Theta(\tilde{d}^2)$",
        zorder=2,
    )

    ax.set_xlabel(r"padded dimension $\tilde{d}$")
    ax.set_ylabel("projection time (s)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="both", alpha=0.18, lw=0.7)

    # Two-row legend ABOVE the plot (ncol=2). Pairing each method's
    # empirical / theoretical line on the same row makes the two
    # complexity classes line up vertically in the legend.
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=11,
        handlelength=2.0,
        handletextpad=0.5,
        columnspacing=2.0,
        borderpad=0.2,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"saved {out_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=None,
        help="CSV file produced by cpp/bench_projection "
             "(columns: dim, fwht_ms, matvec_ms). If omitted, defaults "
             "to output/projection_timings.csv next to this script.",
    )
    ap.add_argument(
        "--bench-python",
        action="store_true",
        help="(local sanity check only) bypass the CSV and time FWHT "
             "vs matvec in pure Python. NOT for paper numbers.",
    )
    ap.add_argument("--dims", default="1024,2048,4096,8192,16384",
                    help="dims for the Python fallback")
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument(
        "--out",
        default=os.path.join(here, "output", "projection_scaling.pdf"),
    )
    ap.add_argument(
        "--out-table",
        default=os.path.join(here, "output", "projection_scaling_table.tex"),
        help="output path for the LaTeX tabular of the timings",
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.bench_python:
        dims = [int(d) for d in args.dims.split(",") if d.strip()]
        dims, fwht_times, matvec_times = python_fallback_time(
            dims, repeats=args.repeats, warmup=args.warmup,
        )
        print("# Python fallback timings (NOT for paper):")
    else:
        csv_path = args.csv or os.path.join(here, "output", "projection_timings.csv")
        if not os.path.exists(csv_path):
            print(f"ERROR: {csv_path} not found.", file=sys.stderr)
            print("Run the C++ benchmark first:", file=sys.stderr)
            print("    cd cpp && bash build.sh && "
                  "./bench_projection > ../output/projection_timings.csv",
                  file=sys.stderr)
            print("Or use --bench-python for a (less accurate) local sanity check.",
                  file=sys.stderr)
            sys.exit(1)
        dims, fwht_times, matvec_times = load_csv(csv_path)
        print(f"# loaded {csv_path}")

    print(f"{'d':>10} {'FWHT (ms)':>14} {'matvec (ms)':>14} {'ratio':>10}")
    for d, tf, tm in zip(dims, fwht_times, matvec_times):
        print(f"{d:>10} {tf*1000:>14.4f} {tm*1000:>14.4f} {tm/tf:>10.1f}x")

    plot(dims, fwht_times, matvec_times, args.out)
    write_latex_table(dims, fwht_times, matvec_times, args.out_table)


if __name__ == "__main__":
    main()
