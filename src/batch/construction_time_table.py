r"""
Construction-time ablation table generator.

Reads HNSW construction logs from
    <repo>/Construction/script/log/
and produces (1) a console markdown table for sanity-checking and
(2) a LaTeX table body ready to drop into the paper.

Paper naming convention:
    op0  on prefix S  -> SRHT projection, exact L2 during build      (\sad-Exact row)
    op18 on prefix S  -> SRHT projection, \sad accelerated DCO       (\sad row)
    op0  on prefix T  -> SRHT-PCA projection, exact L2 during build  (\sda-Exact row)
    op20 on prefix T  -> SRHT-PCA projection, \sda accelerated DCO   (\sda row)

The reduction column reports (1 - t_accel / t_exact) * 100%.
The speedup column reports t_exact / t_accel.
"""

from __future__ import annotations

import re
from pathlib import Path

# ---- paths ----------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = (
    SCRIPT_DIR.parent.parent / "Construction" / "script" / "log"
).resolve()

# ---- dataset table (code in filename -> label used in LaTeX) -------------

DATASETS = [
    ("trevi",                 r"\textsc{Trevi}"),
    ("OpenAI3072",            r"\textsc{OpenAI3072}"),
    ("arxiv-for-fanns-large", r"\textsc{Arxiv-fanns}"),
    ("XUltra",                r"\textsc{XUltra}"),
]

# (rotation prefix, exact-op, accelerated-op, paper macro for accelerated)
GROUPS = [
    ("S", 0, 18, r"\sad"),
    ("T", 0, 20, r"\sda"),
]

BUILD_RE = re.compile(r"build index:\s*([0-9.eE+\-]+)\s*\[ms\]")


def read_build_ms(path: Path):
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = BUILD_RE.search(text)
    return float(match.group(1)) if match else None


def fmt_seconds(ms, latex=True):
    if ms is None:
        return "--"
    s = f"{round(ms / 1000.0):,}"
    if not latex:
        return s
    if "," in s:
        return "$" + s.replace(",", "{,}") + "$"
    return s


def fmt_reduction(exact_ms, accel_ms, latex=True):
    if exact_ms is None or accel_ms is None or exact_ms == 0:
        return "--"
    pct = (1.0 - accel_ms / exact_ms) * 100.0
    return f"{pct:.1f}\\%" if latex else f"{pct:.1f}%"


def fmt_speedup(exact_ms, accel_ms, latex=True):
    if exact_ms is None or accel_ms is None or accel_ms == 0:
        return "--"
    ratio = exact_ms / accel_ms
    return f"{ratio:.2f}$\\times$" if latex else f"{ratio:.2f}x"


def collect():
    rows = []
    for ds_code, ds_label in DATASETS:
        entry = {"code": ds_code, "label": ds_label, "groups": {}}
        for prefix, op_exact, op_accel, macro in GROUPS:
            stem = f"hnsw_{prefix}{ds_code}_M16_efConstruction500"
            f_exact = LOG_DIR / f"{stem}_operator{op_exact}_log.txt"
            f_accel = LOG_DIR / f"{stem}_operator{op_accel}_log.txt"
            entry["groups"][prefix] = {
                "macro":     macro,
                "exact_ms":  read_build_ms(f_exact),
                "accel_ms":  read_build_ms(f_accel),
                "exact_path": f_exact,
                "accel_path": f_accel,
            }
        rows.append(entry)
    return rows


def report_missing(rows):
    missing = []
    for r in rows:
        for g in r["groups"].values():
            if g["exact_ms"] is None:
                missing.append(g["exact_path"].name)
            if g["accel_ms"] is None:
                missing.append(g["accel_path"].name)
    if missing:
        print("# WARNING: 'build index' line not found in:")
        for name in missing:
            print(f"#   - {name}")
        print()


def print_markdown(rows):
    header = (
        "| Dataset | \\sad-Exact (s) | \\sad (s) | Reduction | Speedup "
        "| \\sda-Exact (s) | \\sda (s) | Reduction | Speedup |"
    )
    sep = "|" + "---|" * 9
    print("# Markdown (sanity check)")
    print(header)
    print(sep)
    for r in rows:
        s = r["groups"]["S"]
        t = r["groups"]["T"]
        cells = [
            r["code"],
            fmt_seconds(s["exact_ms"], latex=False),
            fmt_seconds(s["accel_ms"], latex=False),
            fmt_reduction(s["exact_ms"], s["accel_ms"], latex=False),
            fmt_speedup(s["exact_ms"], s["accel_ms"], latex=False),
            fmt_seconds(t["exact_ms"], latex=False),
            fmt_seconds(t["accel_ms"], latex=False),
            fmt_reduction(t["exact_ms"], t["accel_ms"], latex=False),
            fmt_speedup(t["exact_ms"], t["accel_ms"], latex=False),
        ]
        print("| " + " | ".join(cells) + " |")
    print()


def print_latex(rows):
    print("# LaTeX (paste into experiments.tex)")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Index construction time (seconds).}")
    print(r"\label{tab:construction-ablation}")
    print(r"\footnotesize")
    print(r"\setlength{\tabcolsep}{3pt}")
    print(r"\begin{tabular}{@{}lrrcrrc@{}}")
    print(r"\toprule")
    print(
        r" & \multicolumn{3}{c}{\sad} "
        r"& \multicolumn{3}{c}{\sda} \\"
    )
    print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    print(
        r"Dataset & Exact (s) & Accel.\ (s) & Speedup "
        r"& Exact (s) & Accel.\ (s) & Speedup \\"
    )
    print(r"\midrule")
    for r in rows:
        s = r["groups"]["S"]
        t = r["groups"]["T"]
        cells = [
            r["label"],
            fmt_seconds(s["exact_ms"]),
            fmt_seconds(s["accel_ms"]),
            fmt_speedup(s["exact_ms"], s["accel_ms"]),
            fmt_seconds(t["exact_ms"]),
            fmt_seconds(t["accel_ms"]),
            fmt_speedup(t["exact_ms"], t["accel_ms"]),
        ]
        print(" & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


def print_summary(rows):
    print("# Aggregate")
    s_reductions = []
    t_reductions = []
    s_speedups = []
    t_speedups = []
    for r in rows:
        s = r["groups"]["S"]
        t = r["groups"]["T"]
        if s["exact_ms"] and s["accel_ms"]:
            s_reductions.append((1 - s["accel_ms"] / s["exact_ms"]) * 100)
            s_speedups.append(s["exact_ms"] / s["accel_ms"])
        if t["exact_ms"] and t["accel_ms"]:
            t_reductions.append((1 - t["accel_ms"] / t["exact_ms"]) * 100)
            t_speedups.append(t["exact_ms"] / t["accel_ms"])

    def avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(
        f"\\sad : reduction range = {min(s_reductions):.1f}% -- "
        f"{max(s_reductions):.1f}%, mean = {avg(s_reductions):.1f}%; "
        f"speedup range = {min(s_speedups):.2f}x -- "
        f"{max(s_speedups):.2f}x, mean = {avg(s_speedups):.2f}x"
    )
    print(
        f"\\sda : reduction range = {min(t_reductions):.1f}% -- "
        f"{max(t_reductions):.1f}%, mean = {avg(t_reductions):.1f}%; "
        f"speedup range = {min(t_speedups):.2f}x -- "
        f"{max(t_speedups):.2f}x, mean = {avg(t_speedups):.2f}x"
    )
    print()


def main():
    print(f"# Log dir: {LOG_DIR}")
    print()
    rows = collect()
    report_missing(rows)
    print_markdown(rows)
    print_latex(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
