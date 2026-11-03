"""
Walk the dataset root on the experiment server and dump a sized
inventory to <script_dir>/dataset_sizes.log.

Sections in the log:
    1. Directory tree (depth up to MAX_DEPTH) with cumulative size per dir
    2. Per-dataset total size
    3. HNSW index files (*.index)
    4. Candidate projection-matrix files (likely small, non-base, non-query)
    5. All files sorted by size (largest first)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------- config --------------------------------------------------------

DATASET_ROOT = Path("/data/dataset")
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FILE = SCRIPT_DIR / "dataset_sizes.log"

MAX_DEPTH = 4               # how deep the directory tree section goes
TOP_N_LARGEST = 50          # how many largest files to list at the end

# files that are clearly raw data / queries / ground truth, not projections
RAW_DATA_PATTERNS = ("_base.", "_query.", "_groundtruth.", "_train.")

# ---------- helpers -------------------------------------------------------

def fmt_size(n_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(n_bytes)
    for u in units:
        if size < 1024 or u == units[-1]:
            return f"{size:7.2f} {u}"
        size /= 1024
    return f"{n_bytes} B"


def safe_stat_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def collect_files(root: Path):
    """Return list of (path, size) for every regular file under root."""
    entries = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.is_symlink():
                continue
            entries.append((p, safe_stat_size(p)))
    return entries


def dir_total(root: Path, files: list[tuple[Path, int]]) -> dict[Path, int]:
    """Total bytes under each directory (root-relative)."""
    totals: dict[Path, int] = {}
    for p, s in files:
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        # walk up the relative parents, accumulate
        cur = rel.parent
        while True:
            totals[cur] = totals.get(cur, 0) + s
            if cur == Path("."):
                break
            cur = cur.parent
    return totals


# ---------- log sections --------------------------------------------------

def section_tree(out, root: Path, dir_totals: dict[Path, int]):
    out.write("=" * 72 + "\n")
    out.write(f"1. Directory tree (depth <= {MAX_DEPTH}) with cumulative size\n")
    out.write("=" * 72 + "\n")
    out.write(f"{fmt_size(dir_totals.get(Path('.'), 0))}  {root}\n")

    # collect every directory we know about, relative to root
    rel_dirs = sorted(d for d in dir_totals if d != Path("."))
    for rd in rel_dirs:
        depth = len(rd.parts)
        if depth > MAX_DEPTH:
            continue
        indent = "  " * depth
        out.write(f"{indent}{fmt_size(dir_totals[rd])}  {rd.as_posix()}/\n")
    out.write("\n")


def section_per_dataset(out, root: Path, dir_totals: dict[Path, int]):
    out.write("=" * 72 + "\n")
    out.write("2. Per-dataset total (top-level subdirectories of root)\n")
    out.write("=" * 72 + "\n")
    top = sorted(
        (d for d in dir_totals if len(d.parts) == 1),
        key=lambda d: -dir_totals[d],
    )
    for d in top:
        out.write(f"{fmt_size(dir_totals[d])}  {d.as_posix()}\n")
    out.write("\n")


def section_indices(out, root: Path, files: list[tuple[Path, int]]):
    out.write("=" * 72 + "\n")
    out.write("3. HNSW index files (*.index)\n")
    out.write("=" * 72 + "\n")
    idx = [(p, s) for p, s in files if p.suffix == ".index"]
    idx.sort(key=lambda ps: ps[0].as_posix())
    for p, s in idx:
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            rel = p.as_posix()
        out.write(f"{fmt_size(s)}  {rel}\n")
    out.write(f"({len(idx)} index files total)\n\n")


def section_projection_candidates(
    out, root: Path, files: list[tuple[Path, int]]
):
    out.write("=" * 72 + "\n")
    out.write("4. Candidate projection-matrix / metadata files\n")
    out.write("   (everything that is NOT raw base/query/groundtruth/train\n")
    out.write("    and NOT an .index file)\n")
    out.write("=" * 72 + "\n")
    out.write("   These are the files most likely to be projection\n")
    out.write("   matrices, sign vectors, permutations, PCA components,\n")
    out.write("   precomputed norms, etc. Inspect by hand to map each\n")
    out.write("   to a method (ADSampling / DADE / DDCres / SAD / SDA).\n\n")

    def is_raw(p: Path) -> bool:
        name = p.name
        return any(pat in name for pat in RAW_DATA_PATTERNS)

    cand = [
        (p, s)
        for p, s in files
        if p.suffix != ".index" and not is_raw(p)
    ]
    cand.sort(key=lambda ps: ps[0].as_posix())
    for p, s in cand:
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            rel = p.as_posix()
        out.write(f"{fmt_size(s)}  {rel}\n")
    out.write(f"({len(cand)} candidate files total)\n\n")


def section_largest(out, root: Path, files: list[tuple[Path, int]]):
    out.write("=" * 72 + "\n")
    out.write(f"5. Largest {TOP_N_LARGEST} files overall\n")
    out.write("=" * 72 + "\n")
    sorted_by_size = sorted(files, key=lambda ps: -ps[1])[:TOP_N_LARGEST]
    for p, s in sorted_by_size:
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            rel = p.as_posix()
        out.write(f"{fmt_size(s)}  {rel}\n")
    out.write("\n")


# ---------- main ----------------------------------------------------------

def main():
    if not DATASET_ROOT.is_dir():
        print(
            f"ERROR: dataset root not found: {DATASET_ROOT}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Scanning {DATASET_ROOT} ...", file=sys.stderr)
    files = collect_files(DATASET_ROOT)
    print(f"  found {len(files)} files", file=sys.stderr)

    totals = dir_total(DATASET_ROOT, files)

    with LOG_FILE.open("w", encoding="utf-8") as out:
        out.write(f"# Dataset inventory\n")
        out.write(f"# Root : {DATASET_ROOT}\n")
        out.write(f"# Files: {len(files)}\n")
        out.write(f"# Total: {fmt_size(totals.get(Path('.'), 0))}\n\n")

        section_tree(out, DATASET_ROOT, totals)
        section_per_dataset(out, DATASET_ROOT, totals)
        section_indices(out, DATASET_ROOT, files)
        section_projection_candidates(out, DATASET_ROOT, files)
        section_largest(out, DATASET_ROOT, files)

    print(f"Wrote {LOG_FILE}", file=sys.stderr)


if __name__ == "__main__":
    main()
