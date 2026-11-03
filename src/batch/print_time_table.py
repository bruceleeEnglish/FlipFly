"""
Print a LaTeX table summarising per-query timing **at a fixed
Recall@K**, pivoted so that datasets sit in column-group headers.

For each (dataset, K, method):
  - Parse the ef sweep rows from the log; linearly interpolate the
    per-ef search time at the target recall.
  - per-query distance = interp_search_time / M
        (search time on each ef row is the sum across M queries)
  - per-query rotation = rotation_total / M
        (the C++ benchmark rotates each query once; rotation_total
         in the log is therefore the sum over M rotations)
  - per-query total    = per-query distance + per-query rotation
  - Rot.\\,%           = 100 * rotation / total
This matches the convention used in plot_qps_right.py and
plot_time_breakdown.py.

Methods that never reach the target recall are rendered as "--".

Usage:
  python3 src/batch/print_time_table.py --K 100 --recall 0.95

Output:
  - stdout                                            (the tabular)
  - src/batch/query_performance/time_table_K<K>_R<R>.tex
"""
import argparse
import os
import re
import sys
from collections import defaultdict


DATASETS = [
    "trevi",
    "arxiv-for-fanns-large",
    "XUltra",
    "OpenAI3072",
]

DATASET_TITLES = {
    "trevi": r"\textsc{Trevi}",
    "arxiv-for-fanns-large": r"\textsc{Arxiv}",
    "XUltra": r"\textsc{XUltra}",
    "OpenAI3072": r"\textsc{OpenAI3072}",
}

METHODS = [
    ("FDScanning",  0, r"\textsc{FDScanning}"),
    ("PDScanning",  1, r"\textsc{PDScanning}"),
    ("PDScanning+", 2, r"\textsc{PDScanning+}"),
    ("ADSampling",  3, r"\textsc{ADSampling}"),
    ("DADE",        4, r"\textsc{DADE}"),
    ("DDCres",      5, r"\textsc{DDCres}"),
    ("SRHT",       18, r"\sad"),
    ("SRHT+PCA",   20, r"\sda"),
]

METRIC_COUNT = 4  # Total, Distance, Rotation, Rot %

EF_RE = re.compile(
    r"^ef(\d+):\s+search time =\s+([\d.]+)\s+\[ms\]\s+recall = ([\d.]+)"
)


def parse_log(path):
    """Return (sorted [{ef, t_ms, recall}], rotation_total_ms)."""
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
        ts = sorted(x[0] for x in grouped[ef])
        rs = sorted(x[1] for x in grouped[ef])
        out.append(
            {
                "ef": ef,
                "t_ms": ts[len(ts) // 2],
                "recall": rs[len(rs) // 2],
            }
        )
    return out, rot_total


def interp_t_at_recall(rows, target):
    """Linearly interpolate ef-row search time at the given recall.
    Returns None if the method never reaches the target."""
    for i, r in enumerate(rows):
        if r["recall"] >= target:
            if i == 0:
                return r["t_ms"]
            prev = rows[i - 1]
            denom = r["recall"] - prev["recall"]
            if denom <= 0:
                return r["t_ms"]
            alpha = (target - prev["recall"]) / denom
            return prev["t_ms"] + alpha * (r["t_ms"] - prev["t_ms"])
    return None


def fmt(v, digits):
    return r"--" if v is None else f"{v:.{digits}f}"


def build_data(log_dir, datasets, k, query_count, target_recall):
    """Returns dict[(dataset, method_label)] ->
    (total_q, dist_q, rot_q, rot_pct) or None."""
    data = {}
    for dataset in datasets:
        for label, op, _ in METHODS:
            log = os.path.join(
                log_dir, f"hnsw_{dataset}_K{k}_operator{op}_search_log.txt"
            )
            if not os.path.exists(log):
                print(f"missing: {log}", file=sys.stderr)
                data[(dataset, label)] = None
                continue
            rows, rot_total = parse_log(log)
            if not rows:
                data[(dataset, label)] = None
                continue
            t_ms = interp_t_at_recall(rows, target_recall)
            if t_ms is None:
                print(
                    f"skip {dataset} K={k} op={op}: "
                    f"never reached recall {target_recall}",
                    file=sys.stderr,
                )
                data[(dataset, label)] = None
                continue

            dist_q = t_ms / query_count
            # FDScanning (op 0) and PDScanning (op 1) have no rotation
            # step at all in the C++ benchmark (`if (op >= 2)` guard in
            # hnsw_search.cpp); rotation_total stays at 0. Use None to
            # render as "--" rather than "0", which would falsely
            # suggest "we measured rotation and it was zero".
            if rot_total <= 0:
                rot_q = None
                rot_pct = None
                total_q = dist_q
            else:
                rot_q = rot_total / query_count
                total_q = dist_q + rot_q
                rot_pct = 100.0 * rot_q / total_q if total_q > 0 else None
            data[(dataset, label)] = (total_q, dist_q, rot_q, rot_pct)
    return data


def _block_lines(data, datasets):
    """Render the header + data rows for ONE block of `datasets` (no
    \\toprule / \\bottomrule / tabular wrapper). Used both as the
    standalone single-block path and as a section inside a multi-block
    table."""
    out = []

    top = [""]
    for ds in datasets:
        top.append(
            r"\multicolumn{" + str(METRIC_COUNT) + r"}{c}{"
            + DATASET_TITLES.get(ds, ds) + r"}"
        )
    out.append(" & ".join(top) + r" \\")

    cmids = []
    for i in range(len(datasets)):
        start = 2 + i * METRIC_COUNT
        end = start + METRIC_COUNT - 1
        cmids.append(r"\cmidrule(lr){" + f"{start}-{end}" + "}")
    out.append(" ".join(cmids))

    # Metric-name row. The percent column re-uses "Rot" because the
    # unit row "(\\%)" already signals percent; writing "Rot \\%"
    # + "(\\%)" duplicates the sign.
    sub = ["Method"]
    for _ in datasets:
        sub.extend(["Total", "Dist", "Rot", "Rot"])
    out.append(" & ".join(sub) + r" \\")

    units = [""]
    for _ in datasets:
        units.extend([r"(ms)", r"(ms)", r"(ms)", r"(\%)"])
    out.append(" & ".join(units) + r" \\")
    out.append(r"\midrule")

    for label, op, tex_label in METHODS:
        cells = [tex_label]
        for ds in datasets:
            entry = data.get((ds, label))
            if entry is None:
                cells.extend([r"--"] * METRIC_COUNT)
            else:
                total_q, dist_q, rot_q, rot_pct = entry
                cells.append(fmt(total_q, 3))
                cells.append(fmt(dist_q,  3))
                cells.append(fmt(rot_q,   3))
                cells.append(fmt(rot_pct, 3))
        out.append(" & ".join(cells) + r" \\")

    return out


def render_tabular(data, dataset_blocks):
    """Render one tabular containing every `dataset_blocks` entry as
    a horizontal section. Blocks share the same column count (so all
    blocks must have the same length); they are separated by \\midrule
    and each gets its own dataset-name + metric-name + unit header
    rows. This avoids two visually independent tabulars."""
    nd_per_block = len(dataset_blocks[0])
    col_spec = "@{}l" + "rrrr" * nd_per_block + "@{}"

    out = [r"\begin{tabular}{" + col_spec + "}", r"\toprule"]
    for i, block in enumerate(dataset_blocks):
        if i > 0:
            out.append(r"\midrule")
        out.extend(_block_lines(data, block))
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log_dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "query_performance"
        ),
    )
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument(
        "--recall",
        type=float,
        default=0.95,
        help="target Recall@K at which per-query times are reported",
    )
    ap.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="comma-separated dataset names (column groups)",
    )
    ap.add_argument("--M", type=int, default=100, help="number of queries per log")
    ap.add_argument(
        "--datasets-per-block",
        type=int,
        default=4,
        help="number of datasets per tabular block. Use 4 for a wide "
        "single tabular (goes inside table*), or 2 to split datasets "
        "into two stacked tabulars (fits inside a single-column table).",
    )
    ap.add_argument("--out", default=None, help="output .tex path")
    args = ap.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    data = build_data(args.log_dir, datasets, args.K, args.M, args.recall)

    # Split datasets into equal-sized blocks. With
    # `datasets_per_block=4` (default) the output is a single wide
    # tabular (use this inside table*); with `datasets_per_block=2`
    # the output is ONE tabular with the four datasets split into a
    # top half (2 datasets) and a bottom half (2 datasets) separated
    # by \midrule + a repeated header (fits a single-column `table`).
    n = args.datasets_per_block
    blocks = [datasets[i:i + n] for i in range(0, len(datasets), n)]
    if any(len(b) != n for b in blocks):
        print(
            f"warning: {len(datasets)} datasets does not divide evenly "
            f"by datasets-per-block={n}; the last block has only "
            f"{len(blocks[-1])} datasets.",
            file=sys.stderr,
        )
    tex = render_tabular(data, blocks)

    out_path = args.out
    if out_path is None:
        r_suffix = f"R{int(round(args.recall * 100)):02d}"
        out_path = os.path.join(
            args.log_dir, f"time_table_K{args.K}_{r_suffix}.tex"
        )
    with open(out_path, "w") as f:
        f.write(tex + "\n")

    print(tex)
    print(f"\nsaved {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
