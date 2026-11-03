# [runs locally] Parse the dco-benchmark query logs (per-query latencies) and report, at
#   recall@100 = 0.95 (linear interpolation between adjacent ef values), the mean and std of the
#   latency, QPS and a bootstrap 95% CI, plus the geometric means of SRHT-AD/ADSampling,
#   SRHT-DA/DDCres and SRHT-DA/SRHT-AD (SRHT-AD = FlipFly, SRHT-DA = FlipFly-D).
# Logs: src/batch/query_performance (shipped in this repository).
# Usage: python parse_logs.py > qps_table.txt   (pass another log folder as argv[1] if needed)
# The output fills Table VI (tab:qps95) and the Findings paragraph of Sec. VI-C.
import os, re, glob, math, random, sys

LOGDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "batch", "query_performance")
OPS = {0: "FDScanning", 1: "PDScanning", 2: "PDScanning+", 3: "ADSampling",
       4: "DADE", 5: "DDCres", 18: "SRHT-AD", 20: "SRHT-DA"}
TARGET = 0.95
random.seed(0)

pat = re.compile(r"hnsw_(.+)_K(\d+)_operator(\d+)_search_log\.txt")
q_re = re.compile(r"^query(\d+): search time = ([0-9.]+) \[ms\]")
ef_re = re.compile(r"^ef(\d+): search time = ([0-9.]+) \[ms\]\s+recall = ([0-9.]+)")
rot_re = re.compile(r"^rotation_time: ([0-9.]+) ms")

def parse(path):
    blocks = {}
    cur = []
    rot = 0.0
    with open(path) as f:
        for line in f:
            m = q_re.match(line)
            if m:
                cur.append(float(m.group(2))); continue
            m = ef_re.match(line)
            if m:
                ef = int(m.group(1)); rec = float(m.group(3))
                blocks[ef] = (rec, cur); cur = []; continue
            m = rot_re.match(line)
            if m:
                rot = float(m.group(1))
    return blocks, rot

results = {}  # (dataset, K, op) -> dict
for path in sorted(glob.glob(os.path.join(LOGDIR, "*.txt"))):
    m = pat.match(os.path.basename(path))
    if not m: continue
    ds, K, op = m.group(1), int(m.group(2)), int(m.group(3))
    if op not in OPS: continue
    blocks, rot = parse(path)
    if not blocks: continue
    nq = max(len(v[1]) for v in blocks.values())
    rot_pq = rot / nq if nq else 0.0
    ok = [(ef, rec, t) for ef, (rec, t) in blocks.items() if rec >= TARGET and len(t) == nq]
    if ok:
        ef, rec, t = min(ok, key=lambda x: x[0])
    else:
        ef, (rec, t) = max(blocks.items(), key=lambda x: x[1][0])
    # linear interpolation at recall == TARGET between bracketing ef points
    pts = sorted((ef_, rec_, t_) for ef_, (rec_, t_) in blocks.items() if len(t_) == nq)
    lat_i = None
    for j in range(1, len(pts)):
        e0, r0, t0 = pts[j-1]; e1, r1, t1 = pts[j]
        if r0 < TARGET <= r1 and r1 > r0:
            w = (TARGET - r0) / (r1 - r0)
            lat_i = [(1-w)*a + w*b + rot_pq for a, b in zip(t0, t1)]
            break
    if lat_i is None and ok:
        lat_i = [x + rot_pq for x in t]   # already >= target at smallest ef
    lat = lat_i if lat_i is not None else [x + rot_pq for x in t]
    mean = sum(lat) / len(lat)
    sd = math.sqrt(sum((x - mean) ** 2 for x in lat) / (len(lat) - 1)) if len(lat) > 1 else 0
    results[(ds, K, op)] = dict(ef=ef, rec=rec, nq=nq, lat=lat, mean=mean, sd=sd,
                                rot=rot_pq, reached=bool(ok))

def boot_ratio(a, b, reps=2000):
    # ratio of mean latency b / a  (speedup of a over b = QPS_a / QPS_b)
    n = len(a); vals = []
    for _ in range(reps):
        ia = [a[random.randrange(n)] for _ in range(n)]
        ib = [b[random.randrange(len(b))] for _ in range(len(b))]
        vals.append((sum(ib) / len(ib)) / (sum(ia) / n))
    vals.sort()
    return vals[int(0.025 * reps)], vals[int(0.5 * reps)], vals[int(0.975 * reps)]

datasets = sorted({k[0] for k in results})
Ks = sorted({k[1] for k in results})
print("=== Per (dataset, K): smallest ef with recall >= %.2f; latency = search + rotation/query [ms]; QPS = 1000/mean ===" % TARGET)
for ds in datasets:
    for K in Ks:
        rows = [(op, results[(ds, K, op)]) for op in OPS if (ds, K, op) in results]
        if not rows: continue
        print("\n[%s  K=%d  n_queries=%d]" % (ds, K, rows[0][1]["nq"]))
        print("%-12s %6s %8s %10s %10s %8s %9s %s" % ("operator", "ef", "recall", "mean_ms", "std_ms", "rot_ms", "QPS", ""))
        for op, r in rows:
            flag = "" if r["reached"] else "  (target not reached; best recall shown)"
            print("%-12s %6d %8.4f %10.3f %10.3f %8.3f %9.1f%s" % (OPS[op], r["ef"], r["rec"], r["mean"], r["sd"], r["rot"], 1000.0 / r["mean"], flag))
        pairs = [(18, 3, "SRHT-AD / ADSampling"), (20, 5, "SRHT-DA / DDCres"), (20, 18, "SRHT-DA / SRHT-AD"), (18, 5, "SRHT-AD / DDCres"), (20, 3, "SRHT-DA / ADSampling")]
        for a, b, name in pairs:
            if (ds, K, a) in results and (ds, K, b) in results:
                ra, rb = results[(ds, K, a)], results[(ds, K, b)]
                lo, med, hi = boot_ratio(ra["lat"], rb["lat"])
                print("   speedup %-22s = %.2fx   bootstrap 95%% CI [%.2f, %.2f]" % (name, rb["mean"] / ra["mean"], lo, hi))

# geometric means over datasets at K=100
print("\n=== Geometric-mean speedups over datasets (K=100) ===")
for a, b, name in [(18, 3, "SRHT-AD / ADSampling"), (20, 5, "SRHT-DA / DDCres"), (20, 18, "SRHT-DA / SRHT-AD")]:
    vals = []
    for ds in datasets:
        if (ds, 100, a) in results and (ds, 100, b) in results:
            vals.append(results[(ds, 100, b)]["mean"] / results[(ds, 100, a)]["mean"])
    if vals:
        g = math.exp(sum(math.log(v) for v in vals) / len(vals))
        print("%-22s geo-mean = %.2fx over %d datasets  (per-dataset: %s)" % (name, g, len(vals), ", ".join("%.2f" % v for v in vals)))
