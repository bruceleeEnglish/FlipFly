r"""
Parse dataset_sizes.log (in this directory) and emit two tables:

    Table A: HNSW index size, 4 datasets x 6 methods
             Uses the op=0, M=16, efC=500 build for every method.
             Different DCO operators produce identical graph topologies,
             so the index file size is independent of operator; op=0 is
             chosen because every method has it built.

    Table B: Projection-matrix memory, 4 datasets x 6 methods
             For each method we report the size of its canonical
             projection-representation file:
                FDScanning  -> (none)                       => 0
                ADSampling  -> {ds}/adsampling/O.fvecs
                DADE        -> {ds}/dade/P.fvecs
                DDCres      -> {ds}/ddc/pca_matrix.fvecs
                \sad        -> {ds}/srht/S.fvecs
                \sda        -> {ds}/srht_pca/S.fvecs

Output goes to stdout: markdown (sanity) + LaTeX (paste-ready).
"""

from __future__ import annotations

import re
from pathlib import Path

# ---- paths ---------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FILE = SCRIPT_DIR / "dataset_sizes.log"
TEX_OUT = SCRIPT_DIR / "size_tables.tex"

# ---- experiment matrix ---------------------------------------------------

DATASETS = [
    ("trevi",                 r"\textsc{Trevi}"),
    ("OpenAI3072",            r"\textsc{OpenAI3072}"),
    ("arxiv-for-fanns-large", r"\textsc{Arxiv-fanns}"),
    ("XUltra",                r"\textsc{XUltra}"),
]

# (label, HNSW-filename prefix, projection-file path template)
METHODS = [
    (r"\textsc{FDScanning}", "",  None),
    (r"\textsc{ADSampling}", "O", "{ds}/adsampling/O.fvecs"),
    (r"\textsc{DADE}",       "P", "{ds}/dade/P.fvecs"),
    (r"\textsc{DDCres}",     "L", "{ds}/ddc/pca_matrix.fvecs"),
    (r"\sad",                "S", "{ds}/srht/S.fvecs"),
    (r"\sda",                "T", "{ds}/srht_pca/S.fvecs"),
]

# ---- log parsing ---------------------------------------------------------

UNIT_BYTES = {
    "B":  1,
    "KB": 1024,
    "MB": 1024 ** 2,
    "GB": 1024 ** 3,
    "TB": 1024 ** 4,
}

SIZE_LINE = re.compile(r"^\s*([0-9.]+)\s+(B|KB|MB|GB|TB)\s+(\S.+?)\s*$")


def parse_log(log_path: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        m = SIZE_LINE.match(line)
        if not m:
            continue
        path = m.group(3).strip()
        if path.endswith("/"):          # directory aggregate, skip
            continue
        sizes[path] = round(float(m.group(1)) * UNIT_BYTES[m.group(2)])
    return sizes


# ---- lookups -------------------------------------------------------------

def index_path(prefix: str, ds: str) -> str:
    return (
        f"{ds}/hnsw_{prefix}{ds}_M16_efConstruction500_operator0.index"
    )


def collect_index_sizes(sizes):
    out = {}
    for mi, (_, prefix, _) in enumerate(METHODS):
        for di, (ds, _) in enumerate(DATASETS):
            out[(mi, di)] = sizes.get(index_path(prefix, ds))
    return out


def collect_projection_sizes(sizes):
    out = {}
    for mi, (_, _, tpl) in enumerate(METHODS):
        for di, (ds, _) in enumerate(DATASETS):
            if tpl is None:
                out[(mi, di)] = 0
            else:
                out[(mi, di)] = sizes.get(tpl.format(ds=ds))
    return out


# ---- formatting ----------------------------------------------------------

def fmt_gb(n):
    if n is None:
        return "--"
    return f"{n / 1024 ** 3:.2f}"


def fmt_mb(n):
    if n is None:
        return "--"
    if n == 0:
        return "0"
    return f"{n / 1024 ** 2:.2f}"


def strip_latex(label):
    return (
        label.replace(r"\textsc{", "")
             .replace("}", "")
             .replace("\\", "")
    )


# ---- table emitters ------------------------------------------------------

def md_table(values, fmt):
    header = "| Method | " + " | ".join(c for c, _ in DATASETS) + " |"
    sep = "|" + "---|" * (len(DATASETS) + 1)
    lines = [header, sep]
    for mi, (label, _, _) in enumerate(METHODS):
        row = [strip_latex(label)]
        for di in range(len(DATASETS)):
            row.append(fmt(values.get((mi, di))))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def latex_index_table(values):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{HNSW index size (GB).}",
        r"\label{tab:hnsw-index-size}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Method & " + " & ".join(lab for _, lab in DATASETS) + r" \\",
        r"\midrule",
    ]
    for mi, (label, _, _) in enumerate(METHODS):
        row = [label]
        for di in range(len(DATASETS)):
            row.append(fmt_gb(values.get((mi, di))))
        lines.append(" & ".join(row) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def latex_projection_table(values):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Projection-matrix memory (MB).}",
        r"\label{tab:projection-memory}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Method & " + " & ".join(lab for _, lab in DATASETS) + r" \\",
        r"\midrule",
    ]
    for mi, (label, _, _) in enumerate(METHODS):
        row = [label]
        for di in range(len(DATASETS)):
            row.append(fmt_mb(values.get((mi, di))))
        lines.append(" & ".join(row) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---- main ----------------------------------------------------------------

def main():
    if not LOG_FILE.exists():
        raise SystemExit(f"Log not found: {LOG_FILE}")

    sizes = parse_log(LOG_FILE)
    print(f"# parsed {len(sizes)} files from {LOG_FILE.name}\n")

    idx = collect_index_sizes(sizes)
    proj = collect_projection_sizes(sizes)

    missing = [k for k, v in {**idx, **proj}.items() if v is None]
    if missing:
        print("# WARNING: missing entries")
        for mi, di in missing:
            print(
                f"#   method = {strip_latex(METHODS[mi][0])}, "
                f"dataset = {DATASETS[di][0]}"
            )
        print()

    print("## Table A: HNSW index size (GB) [markdown]")
    print(md_table(idx, fmt_gb))
    print()

    print("## Table B: Projection-matrix memory (MB) [markdown]")
    print(md_table(proj, fmt_mb))
    print()

    tex = "\n\n".join([
        "% Auto-generated by generate_size_tables.py from dataset_sizes.log.",
        "% Do not edit by hand; re-run the script to refresh.",
        "% Table A: HNSW index size",
        latex_index_table(idx),
        "% Table B: Projection-matrix memory",
        latex_projection_table(proj),
    ]) + "\n"
    TEX_OUT.write_text(tex, encoding="utf-8")
    print(f"# wrote LaTeX tables to {TEX_OUT.name}")


if __name__ == "__main__":
    main()
