# Parameter / theory-verification experiments

Three standalone Python scripts that produce the empirical figures
for §5.2 of the paper. None of them require the C++ benchmark, the
HNSW index, or the query path. They are designed to be copied to
the server (which has the datasets at
`/data/dataset/`) and run end-to-end with a
single command each.

| Script                            | Figure / table          | Section | Needs                                  |
|-----------------------------------|-------------------------|---------|----------------------------------------|
| `cpp/bench_projection` + `plot_projection_scaling.py` | `fig:projection-scaling` | §5.2.1  | nothing (synthetic, C++ binary times the same code path used by the search benchmark) |
| `plot_variance_verification.py`   | `fig:variance-verification` + `tab:walsh-flatness` | §5.2.2 | original base + query fvecs            |
| `plot_headroom.py`                | `fig:headroom` + `tab:headroom-table` + `tab:proxy-agreement` | §5.2.3 | SRHT-rotated base fvecs (`S<ds>_base.fvecs`) |

## Dependencies

```
python >= 3.8
numpy
matplotlib
scipy   # only for plot_headroom (kendalltau)
```

## Default data layout

Scripts assume the server layout (`/data/dataset/<dataset>/...`)
with files:

| Dataset                  | Base (B / C input)                  | Query (B input only)        | SRHT-rotated (C input)         |
|--------------------------|--------------------------------------|-----------------------------|--------------------------------|
| trevi                    | `trevi_base.fvecs`                   | `query.fvecs`               | `Strevi_base.fvecs`            |
| arxiv-for-fanns-large    | `arxiv-for-fanns-large_base.fvecs`   | `query.fvecs`               | `Sarxiv-for-fanns-large_base.fvecs` |
| XUltra                   | `msmacro_base-001.bin`               | `query.bin`                 | `SXUltra_base.fvecs`           |
| OpenAI3072               | `openai3072_base.bin`                | `query.bin`                 | `SOpenAI3072_base.fvecs`       |

`.bin` files are treated as raw float32 with no length prefix
(dimension supplied via the per-dataset config in `common.py`).

If a dataset's file path is different on your machine, override via
the `--base-path-*` / `--query-path-*` flags or edit `DATASET_FILES`
in `common.py`.

## Output

All scripts write a single PDF to `output/` next to the scripts (or
wherever you point `--out`). PDFs use the same DB-paper colour
palette as the rest of the paper (mauve for ADSampling-family,
brick red for SRHT, near-black for SRHT+PCA).

## Quick start

```bash
# A: C++ microbenchmark, then Python plot.
#    The C++ binary reuses the FWHT / matvec implementations from
#    DCO/L2/SRHTADSampling.hpp and DCO/L2/ADSampling.hpp so the timings
#    reflect the same code path used by the search benchmark.
cd cpp && bash build.sh && cd ..
cpp/bench_projection           > output/projection_timings.csv
python3 plot_projection_scaling.py --csv output/projection_timings.csv

# B: a few minutes, reads M_q query/base pairs per dataset
python3 plot_variance_verification.py \
    --data-root /data/dataset

# C: a few minutes, streams S<ds>_base.fvecs to compute coordinate
#    statistics
python3 plot_headroom.py \
    --data-root /data/dataset
```

Subsample size (M_q, chunk_size, etc.) can be tuned via flags; see
`--help` on each script.

For local sanity checks on A without a C++ build, use
`python3 plot_projection_scaling.py --bench-python`. The pure-numpy
timing under-states the FWHT advantage because numpy's matvec is
backed by BLAS while the numpy FWHT is not vectorised to the same
degree; do not use this fallback for paper numbers.

## What each script does

### A. projection-scaling

Times an FWHT-based projection vs a dense Haar-style matrix-vector
multiply on synthetic random float32 vectors of dimension
$\tilde{d} \in \{1024, 2048, 4096, 8192, 16384\}$. Plots both
curves on a log-log axis and overlays the theoretical
$\Theta(\tilde{d}\log\tilde{d})$ and $\Theta(\tilde{d}^2)$ slopes
for reference.

### B. variance-verification

For each dataset and each $\ell/\tilde{d} \in \{1/8, 1/4, 1/2\}$:

1. Sample $M_q$ pairs $(\mathbf{x}, \mathbf{q})$ from base and
   query files.
2. Pad both to the next power of two so SRHT applies.
3. For each pair, run `R` independent SRHT draws and compute the
   partial-distance estimator
   $\widehat{\Delta}_\ell = \tfrac{\tilde{d}}{\ell}
   \sum_{i=1}^\ell ((\mathbf{S}(\mathbf{x}-\mathbf{q}))_i)^2$.
4. Empirical variance over the `R` draws is plotted against the
   Haar reference variance
   $\mathrm{Var}^{\text{Haar}}[\widehat{\Delta}_\ell] =
    2 \|\mathbf{x}-\mathbf{q}\|^4 (\tilde{d}-\ell) / (\ell(\tilde{d}+2))$.
5. The ratio empirical/Haar averaged over pairs is the empirical
   Walsh-flatness parameter $\eta_\ell$.

Output: 3-row × 4-col scatter figure + Walsh-flatness LaTeX table.

### C. headroom

For each dataset, streams the SRHT-rotated base file
`S<ds>_base.fvecs` to compute, per coordinate $i \in [\tilde{d}]$:

* $\mu_i = \mathbb{E}[(\mathbf{S}\mathbf{x})_i]$
* $\sigma_i^2 = \mathbb{E}[(\mathbf{S}\mathbf{x})_i^2]$
* $\nu_i^2 = \sigma_i^2 - \mu_i^2$

Then:

* Kendall's $\tau$ between the proxy permutation
  $\pi = \mathrm{argsort}\, \sigma_i^2$ desc and the optimal
  $\pi^\star = \mathrm{argsort}\, \nu_i^2$ desc.
* Polynomial-decay exponent $\alpha$ fitted on the sorted
  $\nu_{(i)}^2$ spectrum.
* Predicted headroom from Corollary 3 (closed form
  $\ell_{\mathrm{eff}}^{\pi^\star}/\ell \approx
  (\ell/\tilde{d})^{\alpha/(1-\alpha)}$) vs observed headroom
  (computed by integrating the sorted variance spectrum).
