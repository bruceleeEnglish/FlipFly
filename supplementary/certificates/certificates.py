# [needs the datasets] Flatness certificate and margin condition (offline; no query is re-run).
# Inputs: dataset/<ds>/{<ds>_train.fvecs, srht/S.fvecs, srht_pca/S.fvecs, Squery.fvecs}
# Output (stdout; tee it into a file):
#   - h(l) and max_l h(l)/c(l) for the second-moment ordering pi and the variance ordering pi*
#     -> margin condition, Sec. V-C of the paper
#   - median / q0.90 / q0.99 / max of the flatness excess theta(r,l) -> Sec. VI-B2
#   - exact variance ratio Var/(2||r||^4/l) at l = 32..D/4 -> Table III (variance factor)
#   - kappa(r) and kappa_inf(r) of the same residuals -> Secs. IV-C and VI-B2
# certificates_out.txt is the first run (first two items only); certificates_out_v2.txt is the
# full run with all three items for all four datasets.
import numpy as np, os, sys, time

ROOT = "/data/dataset"
DS = {  # dataset dir -> (train file, query padded file)
    "trevi": "trevi_train.fvecs",
    "OpenAI3072": "OpenAI3072_train.fvecs",
    "arxiv-for-fanns-large": "arxiv-for-fanns-large_train.fvecs",
    "XUltra": "XUltra_train.fvecs",
}
EPS0 = 2.1
DELTA = 32
rng = np.random.default_rng(0)

def read_fvecs(path, max_rows=None):
    fv = np.fromfile(path, dtype=np.float32)
    dim = fv.view(np.int32)[0]
    fv = fv.reshape(-1, 1 + dim)[:, 1:]
    if max_rows: fv = fv[:max_rows]
    return np.ascontiguousarray(fv)

def fwht(X):
    X = np.ascontiguousarray(X, dtype=np.float32).copy()
    D = X.shape[-1]; h = 1
    while h < D:
        v = X.reshape(-1, D // (2 * h), 2, h)
        a = v[..., 0, :].copy()
        np.add(a, v[..., 1, :], out=v[..., 0, :])
        np.subtract(a, v[..., 1, :], out=v[..., 1, :])
        h <<= 1
    return X

def pad(X, D):
    if X.shape[1] == D: return X
    return np.pad(X, ((0, 0), (0, D - X.shape[1]))).astype(np.float32)

for ds, train_name in DS.items():
    t0 = time.time()
    d = os.path.join(ROOT, ds)
    rad = read_fvecs(os.path.join(d, "srht", "S.fvecs"))[0]
    D = rad.size
    S2 = read_fvecs(os.path.join(d, "srht_pca", "S.fvecs"))
    perm = S2[1].astype(np.int64)
    assert np.allclose(S2[0], rad), "rad mismatch between srht and srht_pca"
    Q = read_fvecs(os.path.join(d, "Squery.fvecs"))            # padded raw queries
    Xt = pad(read_fvecs(os.path.join(d, train_name)), D)        # raw train, padded
    Yt = fwht(Xt * rad[None, :]) / np.sqrt(D)                   # rotated train
    mu = Yt.mean(0).astype(np.float64); s2 = (Yt.astype(np.float64) ** 2).mean(0)
    nu2 = np.maximum(s2 - mu ** 2, 0)
    pistar = np.argsort(-nu2)
    ells = np.arange(DELTA, D + 1, DELTA)
    c = (1 + EPS0 / np.sqrt(ells)) ** 2
    tot = nu2.sum()
    def h(p):
        cs = np.cumsum(nu2[p]); return D * cs[ells - 1] / (ells * tot)
    hp, hs = h(perm), h(pistar)
    r_p, r_s = hp / c, hs / c
    ip, is_ = r_p.argmax(), r_s.argmax()
    print(f"\n[{ds}]  D={D}  n_train={Xt.shape[0]}  n_query={Q.shape[0]}")
    print(f"  headroom h at D/8: pi(sigma^2)={hp[D//8//DELTA-1]:.3f}  pi*(nu^2)={hs[D//8//DELTA-1]:.3f}   at D/4: {hp[D//4//DELTA-1]:.3f} / {hs[D//4//DELTA-1]:.3f}")
    print(f"  h(32): pi={hp[0]:.3f}  pi*={hs[0]:.3f}   c(32)={c[0]:.3f}")
    print(f"  MAX_l h/c  pi(sigma^2): {r_p.max():.3f} at l={ells[ip]}   |  pi*(nu^2): {r_s.max():.3f} at l={ells[is_]}")
    print(f"  fraction of checkpoints with h/c>=1: pi={np.mean(r_p >= 1):.3f}  pi*={np.mean(r_s >= 1):.3f}")
    # flatness certificate: residuals = train vector - query
    nq = Q.shape[0]; reps = 4
    idx = rng.integers(0, Xt.shape[0], size=(nq, reps))
    R = (Xt[idx.ravel()] - np.repeat(Q, reps, axis=0)).astype(np.float32)   # 400 residuals
    W = fwht(R ** 2)                                                        # w = H r^2 (unnormalized)
    r4 = (R.astype(np.float64) ** 4).sum(1)
    W2 = W.astype(np.float64) ** 2
    print("  flatness theta(r,l) = (sum_{m=1}^{L-1} w_m^2 / (l*||r||_4^4) - 1)_+  over %d residuals" % R.shape[0])
    print("   %6s %5s %8s %8s %8s %8s" % ("l", "L", "median", "q0.90", "q0.99", "max"))
    for l in [32, 64, 96, 128, 160, 256, 512, D // 8, D // 4]:
        L = 1 << int(np.ceil(np.log2(l)))
        th = np.maximum(W2[:, 1:L].sum(1) / (l * r4) - 1, 0)
        print("   %6d %5d %8.3f %8.3f %8.3f %8.3f" % (l, L, np.median(th), np.quantile(th, 0.9), np.quantile(th, 0.99), th.max()))
    # exact variance ratio to the Haar leading term 2||r||^4/l at power-of-two depths (Theorem identity)
    r2 = (R.astype(np.float64) ** 2).sum(1); kappa = D * r4 / r2 ** 2
    print("  kappa(r): median=%.2f q0.99=%.2f max=%.2f" % (np.median(kappa), np.quantile(kappa, 0.99), kappa.max()))
    kinf = D * (R.astype(np.float64) ** 2).max(1) / r2
    print("  kappa_inf(r): median=%.2f q0.99=%.2f max=%.2f" % (np.median(kinf), np.quantile(kinf, 0.99), kinf.max()))
    print("  exact Var/(2||r||^4/l) at power-of-two depths (1.0 = Haar leading term):")
    print("   %6s %8s %8s %8s %8s" % ("l", "median", "q0.90", "q0.99", "max"))
    for l in [32, 64, 128, 256, 512, 1024, D // 8, D // 4]:
        ratio = 1 - l * r4 / r2 ** 2 + W2[:, 1:l].sum(1) / r2 ** 2
        print("   %6d %8.3f %8.3f %8.3f %8.3f" % (l, np.median(ratio), np.quantile(ratio, 0.9), np.quantile(ratio, 0.99), ratio.max()))
    # variance ratio to Haar at D/8 via exact identity (power of two), for cross-check with Table walsh-flatness
    l = D // 8; r2 = (R.astype(np.float64) ** 2).sum(1)
    var = 2 * r2 ** 2 / l - 2 * r4 + 2 / l * W2[:, 1:l].sum(1)
    haar = 2 * r2 ** 2 * (D - l) / (l * (D + 2))
    print(f"  exact-identity variance ratio to Haar at l=D/8: median={np.median(var/haar):.3f}  (paper Table: eta_{{D/8}})")
    print(f"  elapsed {time.time()-t0:.1f}s")
