"""Extract X / Y / Z ratios for the paper at recall@100 = 0.95.

X = QPS(SRHT-AD)    / QPS(ADSampling) at same recall
Y = QPS(SRHT-DA)    / QPS(SRHT-AD)    at same recall
Z = QPS(SRHT-DA)    / QPS(DDCres)     at same recall

For each dataset, parses the per-ef aggregate "search time" lines and
linearly interpolates the time-vs-recall curve at recall = 0.95. QPS
ratios collapse to (time_other / time_target) since the query count is
the same across methods.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "query_performance")

DATASETS = ["trevi", "arxiv-for-fanns-large", "XUltra", "OpenAI3072"]
BASELINES = {
    0: "FDScanning",
    1: "PDScanning",
    2: "PDScanning+",
    3: "ADSampling",
    4: "DADE",
    5: "DDCres",
}
OURS = {18: "SRHT-AD", 20: "SRHT-DA"}
ALL_OPS = {**BASELINES, **OURS}
K_LIST = [20, 60, 100]
TARGET_RECALL = 0.95

EF_RE = re.compile(r"^ef(\d+):\s+search time =\s+([\d.]+)\s+\[ms\]\s+recall = ([\d.]+)")


QUERY_COUNT = 100  # 100 queries per dataset, per dco-benchmark convention


def parse_log(path):
    """Returns sorted list of (recall, per_query_ms_end_to_end) pairs.

    Matches plot_qps_right.py's accounting:
      - per-ef `search time` is the total ms across QUERY_COUNT queries at that ef;
        across repeats we take the median (line 104 of plot_qps_right.py).
      - `rotation_time:` is the total rotation ms across QUERY_COUNT queries
        for the whole log.
      - per-query end-to-end ms = (search_total + rotation_total) / QUERY_COUNT.
    """
    if not os.path.exists(path):
        return None
    bucket = {}
    rot = 0.0
    with open(path) as f:
        for line in f:
            m = EF_RE.match(line.strip())
            if m:
                ef = int(m.group(1))
                t = float(m.group(2))
                r = float(m.group(3))
                bucket.setdefault(ef, []).append((t, r))
            elif line.startswith("rotation_time:"):
                rot = float(line.split()[1])
    pts = []
    for ef, vs in bucket.items():
        ts = sorted(v[0] for v in vs)
        rs = sorted(v[1] for v in vs)
        t_med = ts[len(ts) // 2]
        r_med = rs[len(rs) // 2]
        per_query_ms = (t_med + rot) / QUERY_COUNT
        pts.append((r_med, per_query_ms))
    pts.sort()
    return pts


def interp_time(pts, target_recall):
    """Linear interpolation: given pts sorted by recall, return time at target."""
    if not pts:
        return None
    if target_recall <= pts[0][0]:
        return pts[0][1]
    if target_recall >= pts[-1][0]:
        return pts[-1][1]
    for (r1, t1), (r2, t2) in zip(pts[:-1], pts[1:]):
        if r1 <= target_recall <= r2:
            if r2 == r1:
                return (t1 + t2) / 2
            return t1 + (t2 - t1) * (target_recall - r1) / (r2 - r1)
    return None


import math


def gmean(xs):
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def collect_times(K):
    """Returns {dataset: {op_name: per_query_ms}} at TARGET_RECALL."""
    out = {}
    for ds in DATASETS:
        times = {}
        for opid, name in ALL_OPS.items():
            path = os.path.join(LOG_DIR,
                                f"hnsw_{ds}_K{K}_operator{opid}_search_log.txt")
            pts = parse_log(path)
            times[name] = interp_time(pts, TARGET_RECALL) if pts else None
        out[ds] = times
    return out


def report_K(K):
    print(f"\n{'=' * 100}")
    print(f"K = {K},  target recall@{K} = {TARGET_RECALL}  (per-query ms)")
    print(f"{'=' * 100}")

    times_all = collect_times(K)

    # 1) Per-(K, dataset): does each of our two methods dominate ALL 6 baselines?
    print(f"\n[Domination check: do SRHT-AD and SRHT-DA beat EVERY baseline?]")
    header = f"{'Dataset':<25} " + " ".join(f"{n[:7]:>8}" for n in ALL_OPS.values())
    print(header)
    print("-" * len(header))
    failures = []
    for ds, times in times_all.items():
        cells = []
        for n in ALL_OPS.values():
            t = times.get(n)
            cells.append(f"{t:>8.2f}" if t is not None else f"{'--':>8}")
        print(f"{ds:<25} " + " ".join(cells))
        t_sad = times.get("SRHT-AD")
        t_sda = times.get("SRHT-DA")
        for bname in BASELINES.values():
            tb = times.get(bname)
            if tb is None:
                continue
            if t_sad is not None and t_sad >= tb:
                failures.append((K, ds, "SRHT-AD", bname, t_sad, tb))
            if t_sda is not None and t_sda >= tb:
                failures.append((K, ds, "SRHT-DA", bname, t_sda, tb))

    if failures:
        print(f"\n*** {len(failures)} failures (our method NOT strictly faster than baseline) ***")
        for K_, ds, ours, base, t_o, t_b in failures:
            slowdown = t_o / t_b
            print(f"    K={K_} {ds:<22} {ours} ({t_o:.2f} ms) vs {base} ({t_b:.2f} ms) -> {slowdown:.2f}x slower")
    else:
        print(f"\n  OK: at K={K}, SRHT-AD and SRHT-DA strictly beat all 6 baselines on all datasets.")

    # 2) Headline ratios X / Y / Z
    Xs, Ys, Zs = [], [], []
    print(f"\n[Headline ratios]")
    print(f"{'Dataset':<25} {'X (sad/AD)':>12} {'Y (sda/sad)':>13} {'Z (sda/DDC)':>13}")
    for ds, times in times_all.items():
        t_ad = times.get("ADSampling")
        t_ddc = times.get("DDCres")
        t_sad = times.get("SRHT-AD")
        t_sda = times.get("SRHT-DA")
        X = (t_ad / t_sad) if t_ad and t_sad else None
        Y = (t_sad / t_sda) if t_sad and t_sda else None
        Z = (t_ddc / t_sda) if t_ddc and t_sda else None
        for arr, v in ((Xs, X), (Ys, Y), (Zs, Z)):
            if v is not None and v > 0:
                arr.append(v)

        def fr(v):
            return f"{v:>11.2f}x" if v is not None else f"{'--':>12}"

        print(f"{ds:<25} {fr(X):>12} {fr(Y):>13} {fr(Z):>13}")
    print(f"  geo mean across {len(Xs)} datasets: "
          f"X={gmean(Xs):.3f}x  Y={gmean(Ys):.3f}x  Z={gmean(Zs):.3f}x")
    return failures, (Xs, Ys, Zs)


def main():
    print("All time values below are PER-QUERY (ms), end-to-end (search + rotation).")
    all_failures = []
    all_X, all_Y, all_Z = [], [], []
    for K in K_LIST:
        fails, (Xs, Ys, Zs) = report_K(K)
        all_failures.extend(fails)
        all_X.extend(Xs); all_Y.extend(Ys); all_Z.extend(Zs)

    print(f"\n{'#' * 100}")
    print(f"OVERALL (across K in {K_LIST}, all {len(DATASETS)} datasets at recall={TARGET_RECALL}):")
    print(f"{'#' * 100}")
    print(f"  Total domination failures (ours NOT faster than every baseline): {len(all_failures)}")
    print(f"  X (SRHT-AD vs ADSampling) geo mean across {len(all_X)} (K, ds) pairs: {gmean(all_X):.3f}x")
    print(f"  Y (SRHT-DA vs SRHT-AD)     geo mean across {len(all_Y)} (K, ds) pairs: {gmean(all_Y):.3f}x")
    print(f"  Z (SRHT-DA vs DDCres)      geo mean across {len(all_Z)} (K, ds) pairs: {gmean(all_Z):.3f}x")
    print(f"  X max: {max(all_X):.3f}x,  Z max: {max(all_Z):.3f}x")


if __name__ == "__main__":
    main()
