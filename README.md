# FlipFly

**Partial-distance DCO operators that rotate a query in Θ(d̃ log d̃) time and keep a calibrated, variance-bounded stopping rule.**

Code, raw experiment logs and analysis scripts for the ICDE 2027 submission
*Sub-Quadratic Projection for Calibrated Distance Comparison in High-Dimensional Vector Search*.

Author: Yiqi Li (bruceprofession@whu.edu.cn)

---

## The problem and the answer

Graph and partition indexes for vector search spend most of their query time
on distance-comparison operations (DCOs). Partial-distance DCO operators prune
a candidate from a prefix of its rotated residual. The variance-bounded ones,
ADSampling, DADE and DDCres, rotate every query with a dense d×d matrix. At
3,072 to 12,288 dimensions this rotation alone takes 34 to 93% of query
latency, and all of them end up slower than a scan with no rotation at all.

FlipFly attains Θ(d̃ log d̃) time and Θ(d̃) memory per query rotation by
instantiating the projection with a randomized Hadamard transform (SRHT).
Theorem 1 gives the prefix variance under SRHT in closed form at every depth,
so the distance-aware stopping rule stays calibrated. An offline certificate
identifies the residuals on which a fast Hadamard rotation fails.
FlipFly-D adds a data-derived coordinate ordering, and an offline margin
condition decides per corpus whether to use it.

## Headline results

Setting: HNSW (M = 16, efConstruction = 500), four corpora of 3,072 to 12,288
dimensions, K ∈ {20, 60, 100}, recall@100 = 0.95, one thread, one query at a
time. Every number below is computed from the logs shipped in this repository.

| Result | Value | Paper | Reproduce |
|---|---|---|---|
| End-to-end speedup, FlipFly over ADSampling (geometric mean, K = 100) | **3.0×**, 1.6× to 7.6× per corpus | Abstract, Table VI | `parse_logs.py` |
| End-to-end speedup, FlipFly-D over DDCres | **3.1×**, up to **7.9×** on XUltra | Sec. VI-C | `parse_logs.py` |
| FlipFly-D over the leading data-aware baseline on each corpus | **2.8×** | Abstract | `qps_table.txt` |
| (dataset, K) configurations where both operators beat all six baselines | **12 of 12** | Sec. VI-C, Fig. 7 | `extract_qps_ratios.py` |
| Speedup over the rotation-free PDScanning | **1.41×** each; every dense-rotation operator is slower than PDScanning (best 0.85×) | Sec. VI-C | `qps_table.txt` |
| Rotation share of per-query latency | **≤ 1.06%** for FlipFly and FlipFly-D, 34 to 93% for ADSampling, DADE and DDCres | Sec. VI-D, Fig. 8 | `qps_table.txt` |
| Projection memory | **16 to 128 KB**, against up to 576 MiB (≥ 1,150× smaller) | Table VII | `src/batch/Index_logs/` |
| True neighbours pruned on residuals built to break SRHT | SRHT 62.2%, Haar 0.0%, SRHT with a Θ(d̃) permutation 0.1% | Table IV | `alignment_stress.py` |

In the code and logs FlipFly is `SRHT-AD` (operator 18) and FlipFly-D is
`SRHT-DA` (operator 20). See [Names used in the code](#names-used-in-the-code).

## Verify the headline numbers in one minute

No dataset download is needed. The scripts below read only the logs and
synthetic data in this repository. `parse_logs.py` and
`extract_qps_ratios.py` use the Python standard library, and
`alignment_stress.py` needs `numpy`.

```sh
git clone https://github.com/bruceleeEnglish/FlipFly.git && cd FlipFly

# Table VI and the speedups in the text (about 7 s)
python3 supplementary/qps_table/parse_logs.py > /tmp/qps_table.txt
diff /tmp/qps_table.txt supplementary/qps_table/qps_table.txt   # prints nothing: identical
tail -3 /tmp/qps_table.txt

# Domination check over all 12 (dataset, K) configurations
python3 src/batch/extract_qps_ratios.py | grep -E "Total domination|geo mean across 4"

# Table IV, alignment stress test (about 40 s, rewrites results.txt with identical content)
python3 supplementary/alignment_stress/alignment_stress.py
cat supplementary/alignment_stress/results.txt
```

Expected output:

```
SRHT-AD / ADSampling   geo-mean = 2.99x over 4 datasets  (per-dataset: 1.74, 7.60, 1.61, 3.73)
SRHT-DA / DDCres       geo-mean = 3.05x over 4 datasets  (per-dataset: 1.67, 7.85, 2.89, 2.28)
SRHT-DA / SRHT-AD      geo-mean = 1.00x over 4 datasets  (per-dataset: 1.08, 1.10, 1.30, 0.66)

  geo mean across 4 datasets: X=4.070x  Y=1.024x  Z=4.272x      (K = 20)
  geo mean across 4 datasets: X=4.034x  Y=0.982x  Z=3.866x      (K = 60)
  geo mean across 4 datasets: X=2.987x  Y=0.999x  Z=3.038x      (K = 100)
  Total domination failures (ours NOT faster than every baseline): 0

family       ... pruned SRHT pruned Haar
generic      ...      0.001      0.001
column       ...      0.622      0.000
column+perm  ...      0.001      0.000
```

## Why the stopping rule stays calibrated

| Statement | Where | Checked by |
|---|---|---|
| Theorem 1: the prefix variance under SRHT has a closed form at every depth. Its deviation from the Haar variance is a weighted low-frequency Walsh mass of the squared residual. | Sec. IV-C | Fig. 5, `src/batch/param_experiments/plot_variance_verification.py` |
| Remark 3: an offline certificate with an accept rule. Residuals that pass it keep the distance-aware variance bound. | Sec. IV-C | Table III, `supplementary/certificates/certificates.py` |
| Theorem 3: an index-set form of the bound. Any coordinate ordering fixed independently of the rotation signs keeps it. | Sec. V | Fig. 6, `src/batch/plot_depth_verify.py` |
| Eq. (25): the margin condition. It decides offline, per corpus, whether FlipFly-D's data-derived ordering pays off. | Sec. V-C | lines `MAX_l h/c` in `supplementary/certificates/certificates_out_v2.txt` |

## Costs and scope

These limits are measured and reported in the paper.

- **Padding.** FlipFly pads the dimension to the next power of two. The HNSW
  index grows on OpenAI3072 (11.58 to 15.40 GiB) and XUltra (4.59 to
  6.12 GiB). Trevi and Arxiv-fanns are already 4,096-dimensional. All
  end-to-end numbers above include the padding.
- **FlipFly-D is not uniformly faster than FlipFly.** On Trevi it runs at
  0.66× of FlipFly. The margin condition predicts this sign offline, before
  any query is served, and FlipFly-D still beats all six baselines on Trevi.
- **The variance bound is certified for residuals that pass the certificate.**
  Table IV shows what happens on residuals constructed to fail it, and how a
  Θ(d̃) permutation repairs them.
- **Protocol.** All timings are single-threaded with one query at a time.

## Repository layout

```
FlipFly/
├── README.md
├── CMakeLists.txt  build_all.sh  utils.h  global.h  main.cpp
├── DCO/
│   ├── L2/SRHTADSampling.hpp        FlipFly (operator 18)
│   ├── L2/SRHTPCAADSampling.hpp     FlipFly-D (operator 20)
│   └── ...                          baseline operators from dco-benchmark
├── data/
│   ├── srht/  srht_pca/             offline preprocessing for FlipFly and FlipFly-D
│   └── adsampling/  dade/  ddc/     baseline preprocessing, with logs
├── Construction/                    HNSW and IVF construction, with logs
├── GroundTruth/                     ground truth, with logs
├── src/                             search drivers
│   └── batch/                       run scripts, analysis scripts, all query logs
├── include/                         hnswlib (Apache-2.0) and IVF headers
├── FileIO/  Logger/  GPU/           utilities and GPU code from dco-benchmark (GPU is not used in the paper)
├── supplementary/                   analysis scripts that were run outside the benchmark
│   ├── qps_table/                   parse_logs.py -> qps_table.txt
│   ├── certificates/                certificates.py -> certificates_out.txt, certificates_out_v2.txt
│   ├── alignment_stress/            alignment_stress.py -> results.txt
│   └── paper_figures/               make_fig1a.py, make_fig_rotation.py
└── docs/dco-benchmark_README.md     README of the upstream benchmark
```

All paths in this README are relative to the repository root.

## Names used in the code

The code and the logs predate the final method names of the paper.

| Paper | Names in code, scripts and logs | Operator id | Data prefix |
|---|---|---|---|
| **FlipFly** | `SRHTADSampling`, SRHT-AD, "SRHT" | 18 | `S` |
| **FlipFly-D** | `SRHTPCAADSampling`, SRHT-DA, "SRHT+PCA" | 20 | `T` |
| (inner-product variant, not in the paper) | `SRHTADSamplingIP` | 19 | `S` |
| FDScanning | `FDScanning` | 0 | none |
| PDScanning | `PDScanning` | 1 | none |
| PDScanning+ | `PDScanning+` | 2 | `P` |
| ADSampling | `ADSampling` | 3 | `O` |
| DADE | `DADE` | 4 | `P` |
| DDCres | `DDCres` | 5 | `L` |

Log files are named by operator id, so `..._operator18_...` is FlipFly and
`..._operator20_...` is FlipFly-D. The data prefix is prepended to the base
file and index names. For example, `Strevi_base.fvecs` is the Trevi base set
rotated for FlipFly.

## Where each result comes from

| Paper item | Code | Logs or data it reads | Output |
|---|---|---|---|
| Fig. 1(a), latency breakdown | `src/batch/print_time_table.py`, then `supplementary/paper_figures/make_fig1a.py` | `src/batch/query_performance/hnsw_<ds>_K100_operator{3,18}_search_log.txt` | `src/batch/query_performance/time_table_K100_R95.tex`, `intro_overview_a.pdf` |
| Fig. 4, projection time | `src/batch/param_experiments/cpp/bench_projection.cpp`, `plot_projection_scaling.py` (driver `run_A.sh`) | synthetic vectors | `src/batch/param_experiments/output/projection_timings.csv`, `projection_scaling.pdf` |
| Fig. 5, variance against Haar | `src/batch/param_experiments/plot_variance_verification.py` (driver `run_all.sh`) | base and query vectors | `src/batch/param_experiments/output/variance_verification.pdf`, `walsh_flatness.tex` |
| Fig. 6, scan-depth survival | `src/batch/plot_depth_verify.py` | depth histograms at the end of `src/batch/query_performance/hnsw_<ds>_K{20,60,100}_operator{3,18,20}_search_log.txt` | `depth_verify_K20_60_100.pdf` |
| Fig. 7, QPS against recall | `src/batch/plot_qps_right.py` | `src/batch/query_performance/hnsw_<ds>_K{20,60,100}_operator{0,1,2,3,4,5,18,20}_search_log.txt` | `qps_right_K20_60_100.pdf` |
| Fig. 8, rotation time | `supplementary/paper_figures/make_fig_rotation.py` (values from the `rot_ms` column of `qps_table.txt`); `src/batch/plot_rotation_only.py` reads the same `rotation_time` lines directly from the logs | `src/batch/query_performance/hnsw_<ds>_K100_operator{3,4,5,18,20}_search_log.txt` | `rotation_only_K100.pdf` |
| Table III, variance factor | `supplementary/certificates/certificates.py` | training residuals and `srht/S.fvecs` of each dataset | `supplementary/certificates/certificates_out_v2.txt`, blocks `exact Var/(2\|\|r\|\|^4/l)` |
| Table IV, alignment stress test | `supplementary/alignment_stress/alignment_stress.py` (numpy only, synthetic residuals) | none | `supplementary/alignment_stress/results.txt` |
| Table V, depth headroom | `src/batch/param_experiments/plot_headroom.py` | `S<ds>_base.fvecs` | `src/batch/param_experiments/output/proxy_agreement.tex` |
| Sec. V-C, margin condition `max h/c` | `supplementary/certificates/certificates.py` | as for Table III | `certificates_out_v2.txt`, lines `MAX_l h/c` |
| Secs. IV-C and VI-B2, spread parameters `kappa(r)` and `kappa_inf(r)`; Sec. VI-B2, flatness excess `theta` | `supplementary/certificates/certificates.py` | as for Table III | `certificates_out_v2.txt`, lines `kappa(r)`, `kappa_inf(r)` and the `flatness theta` blocks |
| Table VI, QPS and latency at recall@100 = 0.95; speedups in the text | `supplementary/qps_table/parse_logs.py`; `src/batch/extract_qps_ratios.py` recomputes the main ratios by the same interpolation, without the bootstrap | `src/batch/query_performance/hnsw_<ds>_K100_operator*_search_log.txt` | `supplementary/qps_table/qps_table.txt` (the 2.8× and 1.41× ratios are computed from its per-dataset latencies) |
| Sec. VI-D, rotation share | `src/batch/print_time_table.py` | as for Table VI | `time_table_K100_R95.tex` |
| Table VII, projection memory and index size | `src/batch/Index_logs/generate_size_tables.py` (file inventory from `scan_dataset_sizes.py`) | `src/batch/Index_logs/dataset_sizes.log` | `src/batch/Index_logs/size_tables.tex` |
| Table VIII, index construction time | `src/batch/construction_time_table.py` | `Construction/script/log/hnsw_{S,T}<ds>_M16_efConstruction500_operator{0,18,20}_log.txt` | printed to stdout |

`<ds>` is one of `trevi`, `OpenAI3072`, `arxiv-for-fanns-large` and `XUltra`.

## Full reproduction

Each step names the script that was run and the folder its log went to. The
run scripts are kept as they were after their last run. The dataset, operator
and `K` arrays at the top of each script were edited by hand between runs, so
set them before running a script again.

1. **Build.** Run `build_all.sh` from the repository root. It needs Eigen
   3.4.0 in `third_party/eigen`:
   `git clone --branch 3.4.0 --depth 1 https://gitlab.com/libeigen/eigen.git third_party/eigen`.
   Binaries go to `build/`.
2. **Ground truth.** `GroundTruth/script/generate_ground_truth.sh` writes
   `gt_256.ivecs`. Logs: `GroundTruth/script/log/<ds>_gt_256_log.txt`.
3. **Offline preprocessing,** one pass per method.

   | Method | Script | Logs |
   |---|---|---|
   | FlipFly (`S`) | `data/srht/script/batch.sh` (calls `srht.sh` and `randomized.py`; `pad_query.py` pads only the query file) | `data/srht/script/log/<ds>_srht.txt` |
   | FlipFly-D (`T`) | `data/srht_pca/script/batch.sh` (calls `srht_pca.sh` and `randomized.py`) | `data/srht_pca/script/log/<ds>_srht_pca.txt` |
   | ADSampling (`O`) | `data/adsampling/script/batch.sh` | `data/adsampling/script/log/<ds>_randomized.txt` |
   | DADE (`P`) | `data/dade/script/batch.sh` | `data/dade/script/log/<ds>_pca_0.txt`, `<ds>_getEpsilon_0.txt` |
   | DDCres (`L`) | `data/ddc/script/batch_res.sh` | `data/ddc/script/log/<ds>_ddc_pca_generate.txt`, `<ds>_sampling.txt`, `<ds>_train_groundtruth.txt` |

   `src/batch/plot_preprocessing.py` summarizes the preprocessing times in
   these logs.
4. **HNSW construction.** `Construction/script/hnsw_batch.sh` with `M = 16`,
   `efConstruction = 500` and one `rotation` prefix per run. The indexes used
   for querying are built with operator 0 (exact L2). For Table VIII the same
   script was run with `operator = 18` on prefix `S` and `operator = 20` on
   prefix `T`. Logs:
   `Construction/script/log/hnsw_<prefix><ds>_M<M>_efConstruction<efC>_operator<op>_log.txt`.
   The `M8` and `efConstruction300` logs come from
   `Construction/script/hnsw_batch_parameter_ours.sh`, a parameter study that
   is not reported in the paper.
5. **Queries.** `src/batch/hnsw_search.sh` runs `build/src/hnsw_search` for
   every dataset, operator and `K`. Logs:
   `src/batch/query_performance/hnsw_<ds>_K<K>_operator<op>_search_log.txt`.
   The paper reports `K = 20, 60, 100`. The `K40` and `K50` logs are extra
   runs.
6. **Theory checks.** `src/batch/param_experiments/` holds the stand-alone
   scripts for Figs. 4 and 5 and Table V, described in its own `README.md`.
   Their outputs are in `src/batch/param_experiments/output/`.
7. **Analysis and figures.** The scripts in the table above.

## Log formats

- **Query logs** start with `K`, the file paths and the operator. The `ef`
  sweep is `100, K, 100, 200, ..., 1500`, and the first pass at `ef = 100`
  warms the caches. For each `ef` the log has one line
  `queryN: search time = t [ms]` per query and then
  `efE: search time = T [ms]  recall = r per query`, where `T` is summed over
  the queries. Every query is rotated once before the sweep, and
  `rotation_time: R ms` is the sum over all queries (zero for operators
  without a rotation). The log ends with a histogram of termination depths,
  one line `"depth" : count` per checkpoint.
- **Construction logs** end with `build index: <ms> [ms]`.
- **Preprocessing logs** report the elapsed time of each pass in ms.

## Data layout

Every script reads the datasets from `/data/dataset/<ds>/`. This placeholder
replaced the path on the experiment server. Change it in the run scripts, or
pass `--data-root` to the scripts in `src/batch/param_experiments/`. The
datasets and the HNSW indexes take about 1 TB and are not included.

| Corpus | Vectors | Dim | Source |
|---|---|---|---|
| Trevi | 100,000 | 4,096 | download links in `docs/dco-benchmark_README.md` |
| OpenAI3072 | 1,000,000 | 3,072 | OpenAI `text-embedding-3-large` embeddings, cited in the paper |
| Arxiv-fanns | 2,735,364 | 4,096 | https://huggingface.co/datasets/SPCL/arxiv-for-fanns-large |
| XUltra | 100,100 | 12,288 | download links in `docs/dco-benchmark_README.md` |

Each dataset folder holds the original base and query files, `gt_256.ivecs`,
the rotated base files `<prefix><ds>_base.fvecs`, the padded query
`Squery.fvecs` used by FlipFly and FlipFly-D, the projection files
`srht/S.fvecs` (FlipFly signs), `srht_pca/S.fvecs` (FlipFly-D signs and
ordering), `adsampling/O.fvecs`, `dade/P.fvecs` and `ddc/pca_matrix.fvecs`,
and the indexes `hnsw_<prefix><ds>_M16_efConstruction500_operator<op>.index`.

## Changes to the upstream benchmark

This repository extends the open-source DCO benchmark
[`dco-benchmark`](https://github.com/zzlin237/dco-benchmark)
(Zheng et al., *Distance Comparison Operations Are Not Silver Bullets in
Vector Similarity Search*, ICDE 2026). Everything not marked `new_add` comes
from that repository unchanged. Search for `new_add` to find every change.

- **New files.** `DCO/L2/SRHTADSampling.hpp` (FlipFly),
  `DCO/L2/SRHTPCAADSampling.hpp` (FlipFly-D), `DCO/IP/SRHTADSamplingIP.hpp`,
  `data/srht/`, `data/srht_pca/`, `src/batch/param_experiments/`, the
  analysis scripts in `src/batch/` (`plot_*.py`, `*_table.py`,
  `extract_qps_ratios.py`, `Index_logs/`),
  `Construction/script/hnsw_batch_parameter_ours.sh`, `build_all.sh` and
  everything in `supplementary/`.
- **Modified files.** `CMakeLists.txt` and `utils.h` register operators
  18 to 20. `src/hnsw_search.cpp` and `src/hnsw_search_no_simd.cpp` route
  operator 19 to the inner-product search. The run and preprocessing scripts
  carry our dataset lists, paths and operator ids.

The hnswlib headers in `include/hnswlib/` are under the Apache License 2.0,
included as `include/hnswlib/LICENSE`.

## Environment

All timings were measured single-threaded, one query at a time, on a
dual-socket Intel Xeon Gold 5218R server with 128 GB DRAM. The C++ code is
C++17 and builds with CMake and Eigen 3.4.0. The Python scripts need
Python 3.8 or later with `numpy` and `matplotlib`, and `plot_headroom.py`
also needs `scipy`.

## Not included

- `build/`, the upstream git history and `third_party/eigen` (download it as
  in step 1 of [Full reproduction](#full-reproduction)).
- The upstream paper PDF and the compiled `bench_projection` binary.
- `nohup` outputs of aborted runs and the logs of a dataset that the paper
  does not use.
- The datasets and the HNSW indexes.

## Machine-readable summary

```yaml
paper: "Sub-Quadratic Projection for Calibrated Distance Comparison in High-Dimensional Vector Search"
venue: "ICDE 2027 (submission)"
repository: "https://github.com/bruceleeEnglish/FlipFly"
operators:
  FlipFly:   {code: "DCO/L2/SRHTADSampling.hpp",    operator_id: 18, log_name: "SRHT-AD", rotation: "Theta(d log d) time, Theta(d) memory"}
  FlipFly-D: {code: "DCO/L2/SRHTPCAADSampling.hpp", operator_id: 20, log_name: "SRHT-DA", rotation: "Theta(d log d) time, Theta(d) memory"}
baselines: [FDScanning, PDScanning, PDScanning+, ADSampling, DADE, DDCres]
setting:
  index: "HNSW, M=16, efConstruction=500"
  corpora: {Trevi: 4096, OpenAI3072: 3072, Arxiv-fanns: 4096, XUltra: 12288}
  K: [20, 60, 100]
  target: "recall@100 = 0.95"
  protocol: "single thread, one query at a time"
claims:
  - id: speedup_flipfly_vs_adsampling
    value: "2.99x geometric mean at K=100; 1.61x to 7.60x per corpus"
    evidence: "supplementary/qps_table/qps_table.txt"
    reproduce: "python3 supplementary/qps_table/parse_logs.py"
  - id: speedup_flipflyd_vs_ddcres
    value: "3.05x geometric mean at K=100; 7.85x on XUltra"
    evidence: "supplementary/qps_table/qps_table.txt"
    reproduce: "python3 supplementary/qps_table/parse_logs.py"
  - id: dominates_all_baselines
    value: "both operators faster than all six baselines in 12 of 12 (dataset, K) configurations"
    evidence: "src/batch/query_performance/*_search_log.txt"
    reproduce: "python3 src/batch/extract_qps_ratios.py"
  - id: faster_than_rotation_free_scan
    value: "1.41x geometric mean each over PDScanning; every dense-rotation operator is slower than PDScanning"
    evidence: "supplementary/qps_table/qps_table.txt"
  - id: projection_memory
    value: "16 to 128 KB versus up to 576 MiB"
    evidence: "src/batch/Index_logs/size_tables.tex"
  - id: alignment_stress
    value: "true neighbours pruned: SRHT 0.622, Haar 0.000, SRHT with permutation 0.001"
    evidence: "supplementary/alignment_stress/results.txt"
    reproduce: "python3 supplementary/alignment_stress/alignment_stress.py"
theory:
  - "Theorem 1 (Sec. IV-C): closed-form prefix variance under SRHT at every depth"
  - "Remark 3 (Sec. IV-C): offline certificate; residuals that pass keep the variance bound"
  - "Theorem 3 (Sec. V): index-set form; any ordering fixed independently of the signs keeps the bound"
  - "Eq. (25) (Sec. V-C): offline margin condition choosing FlipFly or FlipFly-D per corpus"
known_costs:
  - "padding to the next power of two grows the HNSW index on OpenAI3072 and XUltra by about 33%"
  - "FlipFly-D runs at 0.66x of FlipFly on Trevi, as the margin condition predicts"
```
