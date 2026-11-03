# new_add: QPS-recall plotter for the new SRHT methods comparison
"""
Reads search logs in src/batch/query_performance/ and produces a 2-panel
QPS-recall plot (search-only + end-to-end).

Run on the server:
  cd /data/dco-benchmark
  python3 src/batch/plot_qps.py
  -> writes  src/batch/query_performance/qps_trevi.png

If you don't have matplotlib:
  pip3 install --user matplotlib
"""
import os
import re
import sys
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EF_RE = re.compile(r"^ef(\d+):\s+search time =\s+([\d.]+)\s+\[ms\]\s+recall = ([\d.]+)")


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
    from collections import defaultdict
    g = defaultdict(list)
    for ef, t, r in rows:
        g[ef].append((t, r))
    out = []
    for ef in sorted(g):
        ts = sorted(x[0] for x in g[ef])
        rs = sorted(x[1] for x in g[ef])
        out.append({"ef": ef,
                    "t_ms":   ts[len(ts)//2],
                    "recall": rs[len(rs)//2]})
    return out, rot_total


def plot(log_dir, dataset, M, ops, out_path):
    # method label, log filename suffix, color, marker
    palette = [
        ("FDScanning",          0,  "#7f7f7f", "x"),
        ("PDScanning",          1,  "#bcbd22", "+"),
        ("PDScanning+",         2,  "#9467bd", "v"),
        ("ADSampling",          3,  "#d62728", "o"),
        ("DADE",                4,  "#ff7f0e", "D"),
        ("DDCres",              5,  "#8c564b", "P"),
        ("SRHT (op=18, ours)",  18, "#1f77b4", "s"),
        ("SRHT+PCA (op=20, ours)", 20, "#2ca02c", "^"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8))
    found = []
    for label, op, color, marker in palette:
        if op not in ops:
            continue
        log = os.path.join(log_dir,
                           f"hnsw_{dataset}_K100_operator{op}_search_log.txt")
        if not os.path.exists(log):
            print(f"  skip op={op}: {log} not found", file=sys.stderr)
            continue
        rows, rt = parse(log)
        if not rows:
            print(f"  skip op={op}: no ef rows parsed", file=sys.stderr)
            continue
        found.append((label, op, rows, rt))
        rec = [r["recall"] for r in rows]
        qps_s = [M * 1000.0 / r["t_ms"] for r in rows]
        qps_e = [1000.0 / (r["t_ms"]/M + rt/M) for r in rows]
        ax1.plot(rec, qps_s, marker=marker, color=color, lw=2, ms=7, label=label)
        ax2.plot(rec, qps_e, marker=marker, color=color, lw=2, ms=7, label=label)

    if not found:
        print("ERROR: no log files found.", file=sys.stderr); sys.exit(1)

    for ax, t in [(ax1, "Search-only QPS  (rotation excluded — benchmark log convention)"),
                  (ax2, "End-to-end QPS  (rotation included — production semantics)")]:
        ax.set_xlabel("Recall@100", fontsize=12)
        ax.set_ylabel("QPS (queries / sec)", fontsize=12)
        ax.set_title(t, fontsize=11)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=9.5)

    fig.suptitle(f"{dataset} — DCO methods on dco-benchmark-new", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"\nsaved {out_path}\n")

    # Print compact end-to-end QPS table at common workpoints
    work_efs = [100, 300, 500, 700, 1000, 1500]
    print("## End-to-end QPS at common workpoints (rotation included)\n")
    header = "| method                  | " + " | ".join(f"ef={e:>4}" for e in work_efs) + " |"
    sep    = "|-" + "-" * 24 + "|" + "|".join(["-" * 11] * len(work_efs)) + "|"
    print(header); print(sep)
    for label, op, rows, rt in found:
        by_ef = {r["ef"]: r for r in rows}
        cells = []
        for ef in work_efs:
            r = by_ef.get(ef)
            if r:
                qps_e = 1000.0 / (r["t_ms"]/M + rt/M)
                cells.append(f"{qps_e:>4.0f} r={r['recall']:.3f}")
            else:
                cells.append("    -      ")
        print(f"| {label:<24} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    print("""
============================================================
 plot_qps.py — QPS-recall plot for SRHT/baseline methods
============================================================
 用法 (在 src/batch/ 下运行):
   python plot_qps.py                              # 默认 OpenAI3072, ops 0-5,18,20
   python plot_qps.py --dataset XUltra             # 切换数据集
   python plot_qps.py --dataset XUltra --ops 3,18,20   # 只画指定 op
   python plot_qps.py --M 1000                     # 调整 query 数 (XUltra/OpenAI3072 都是 100)

 输出: query_performance/qps_<dataset>.png
 日志路径: query_performance/hnsw_<dataset>_K100_operator<op>_search_log.txt
============================================================
""")
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default=None,
                    help="defaults to ../query_performance relative to script")
    ap.add_argument("--dataset", default="OpenAI3072")
    ap.add_argument("--M", type=int, default=100, help="number of queries (XUltra=100, OpenAI3072=100)")
    ap.add_argument("--ops", default="0,1,2,3,4,5,18,20",
                    help="comma-separated op IDs to plot")
    ap.add_argument("--out", default=None, help="output png path")
    args = ap.parse_args()

    if args.log_dir is None:
        args.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "query_performance")
    if args.out is None:
        args.out = os.path.join(args.log_dir, f"qps_{args.dataset}.png")
    ops = [int(x) for x in args.ops.split(",") if x.strip()]
    plot(args.log_dir, args.dataset, args.M, ops, args.out)
