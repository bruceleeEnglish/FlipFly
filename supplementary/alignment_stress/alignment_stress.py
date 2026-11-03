"""Stress test: residuals whose energy aligns with one residue class modulo a power of two.

The experiment answers one question. Is the calibration of ADSampling safe under an arbitrary
substitution of the Haar rotation by the structured SRHT rotation? The paper's variance theorem says
that the deviation from Haar is governed by the low-frequency Walsh mass of the squared residual, and
the residue-class identity says that this mass is large exactly when the energy of the residual
concentrates on indices that share a residue modulo a power of two. This script builds such residuals,
measures the resulting variance and the resulting rate of wrongly pruned neighbours, and checks that the
offline flatness certificate flags the same residuals.

Everything below is self-contained. It needs numpy only, runs in under a minute on one core, and uses no
data from the benchmark. Self-checks run first and abort on failure.

Outputs: results.txt (table reproduced in the paper) written next to this file.
"""
import io
import os

import numpy as np

D = 4096                 # padded dimension, matches the OpenAI3072 and Trevi settings of the paper
L_IMG = 64               # side of the 64 x 64 Trevi patch, so one image column is one residue class mod 64
DELTA = 32               # checkpoint stride of the paper
EPS0 = 2.1               # calibration constant of ADSampling, unchanged
DEPTHS = (32, 64, 128, 512)
N_RESIDUAL = 120         # residual draws per family
N_SIGN = 240             # sign draws per residual
CLOSE = 0.9              # a true neighbour sits at squared distance 0.9 * tau


def fwht(a):
    """In-place style fast Walsh-Hadamard transform along the last axis (Sylvester order)."""
    a = a.astype(np.float64, copy=True)
    h = 1
    while h < a.shape[-1]:
        a = a.reshape(*a.shape[:-1], -1, 2 * h)
        x = a[..., :h].copy()
        y = a[..., h:].copy()
        a[..., :h] = x + y
        a[..., h:] = x - y
        a = a.reshape(*a.shape[:-2], -1)
        h *= 2
    return a


def hadamard(n):
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def c_of(l):
    return (1.0 + EPS0 / np.sqrt(l)) ** 2


def theory_variance(r, l):
    """Theorem 'Distance-aware variance of SRHT-AD', exact at every power-of-two depth."""
    w = fwht(r ** 2)
    ell_pow = 1 << (int(np.ceil(np.log2(l))))
    low = float(np.sum(w[1:ell_pow] ** 2))
    return 2 * np.sum(r ** 2) ** 2 / l - 2 * np.sum(r ** 4) + 2 * low / l


def haar_variance(r, l):
    """Exact variance of the Haar prefix estimator, from the Beta law of the partial energy fraction."""
    n2 = np.sum(r ** 2) ** 2
    return 2 * n2 / l * (D - l) / (D + 2)


def flatness_excess(r, l):
    """theta(r, l) of the offline certificate, zero when the prefix is as flat as a spread residual."""
    w = fwht(r ** 2)
    ell_pow = 1 << (int(np.ceil(np.log2(l))))
    low = float(np.sum(w[1:ell_pow] ** 2))
    return max(low / (l * np.sum(r ** 4)) - 1.0, 0.0)


def self_check():
    rng = np.random.default_rng(1)
    v = rng.normal(size=64)
    assert np.allclose(fwht(v), hadamard(64) @ v), "FWHT does not match the Hadamard matrix"
    r = rng.normal(size=D)
    xi = rng.choice([-1.0, 1.0], size=(20000, D))
    z = fwht(xi * r) / np.sqrt(D)
    acc = np.cumsum(z ** 2, axis=1)
    for l in (32, 128):
        emp = float(np.var(acc[:, l - 1] * (D / l)))
        thy = theory_variance(r, l)
        rel = abs(emp - thy) / thy
        assert rel < 0.05, "measured variance %.4g differs from Theorem 1 by %.1f%%" % (emp, 100 * rel)
        print("  check: depth %3d, measured variance %.4g, Theorem 1 %.4g, relative gap %.1f%%" % (l, emp, thy, 100 * rel))
    assert abs(float(np.mean(acc[:, 511] * (D / 512))) - np.sum(r ** 2)) / np.sum(r ** 2) < 0.01, "estimator is biased"
    print("  check: unbiasedness and the Hadamard transform hold")


def family(kind, rng):
    """One residual of unit norm. 'generic' is isotropic. 'column' puts the energy on one image column,
    that is on one residue class modulo 64. 'column+perm' applies the random permutation of the paper."""
    r = np.zeros(D)
    if kind == "generic":
        r = rng.normal(size=D)
    else:
        col = rng.integers(L_IMG)
        idx = np.arange(col, D, L_IMG)
        r[idx] = rng.normal(size=idx.size)
        if kind == "column+perm":
            r = r[rng.permutation(D)]
    return r / np.linalg.norm(r)


def run_family(kind, rng):
    var_ratio = {l: [] for l in DEPTHS}
    theta = {l: [] for l in DEPTHS}
    pruned_srht, pruned_haar = 0, 0
    total = 0
    for _ in range(N_RESIDUAL):
        r = family(kind, rng)
        tau = float(np.sum(r ** 2)) / CLOSE                   # the candidate is a true neighbour
        xi = rng.choice([-1.0, 1.0], size=(N_SIGN, D))
        acc = np.cumsum(fwht(xi * r) ** 2 / D, axis=1)
        for l in DEPTHS:
            var_ratio[l].append(theory_variance(r, l) / haar_variance(r, l))
            theta[l].append(flatness_excess(r, l))
        lost = np.zeros(N_SIGN, dtype=bool)
        for l in range(DELTA, D + 1, DELTA):
            lost |= acc[:, l - 1] >= tau * (l / D) * c_of(l)
        pruned_srht += int(lost.sum())
        # Haar reference: a Haar rotation sends the residual to a uniform direction, so the vector of
        # partial energy fractions has the law of cumsum(g^2)/||g||^2 for a standard Gaussian g.
        g = rng.normal(size=(N_SIGN, D))
        acc_h = np.cumsum(g ** 2, axis=1) / np.sum(g ** 2, axis=1, keepdims=True) * float(np.sum(r ** 2))
        lost_h = np.zeros(N_SIGN, dtype=bool)
        for l in range(DELTA, D + 1, DELTA):
            lost_h |= acc_h[:, l - 1] >= tau * (l / D) * c_of(l)
        pruned_haar += int(lost_h.sum())
        total += N_SIGN
    return ({l: float(np.mean(var_ratio[l])) for l in DEPTHS},
            {l: float(np.median(theta[l])) for l in DEPTHS},
            pruned_srht / total, pruned_haar / total)


def main():
    print("self-checks")
    self_check()
    rng = np.random.default_rng(20260916)
    rows = []
    for kind in ("generic", "column", "column+perm"):
        vr, th, fp, fph = run_family(kind, rng)
        rows.append((kind, vr, th, fp, fph))
        print("%-12s var/Haar %s  theta(32) %6.2f  wrongly pruned: SRHT %.3f, Haar %.3f"
              % (kind, " ".join("%7.2f" % vr[l] for l in DEPTHS), th[32], fp, fph))
    out = ["setting: d = %d, Delta = %d, eps0 = %.1f, %d residuals x %d sign draws per family" % (D, DELTA, EPS0, N_RESIDUAL, N_SIGN),
           "a candidate is a true neighbour at squared distance %.2f * tau" % CLOSE, "",
           "%-12s %s %10s %10s %10s" % ("family", " ".join("%10s" % ("var/Haar l=%d" % l) for l in DEPTHS),
                                        "theta l=32", "pruned SRHT", "pruned Haar")]
    for kind, vr, th, fp, fph in rows:
        out.append("%-12s %s %10.2f %10.3f %10.3f"
                   % (kind, " ".join("%10.2f" % vr[l] for l in DEPTHS), th[32], fp, fph))
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.txt")
    io.open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
