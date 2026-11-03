# new_add: SRHT + PCA-permutation preprocessing (not in upstream dco-benchmark)
"""
Preprocess data for SRHT-PCA-ADSampling (operator 20).

Pipeline:
  1. Random Rademacher diagonal D_rad
  2. SRHT rotation: y = (1/sqrt(D)) * H * D_rad * x
  3. Compute per-dim variance over all rotated vectors
  4. Sort dims by variance descending -> permutation pi
  5. Permute columns: store y[:, pi] as the data file.

Outputs (mirrors the SRHT layout in data/srht):
  <dataset_dir>/srht_pca/S.fvecs       2 rows of length D:
                                        row 0 = +/-1 Rademacher diagonal
                                        row 1 = pi as float (cast to int32)
  <dataset_dir>/T<dataset>_base.fvecs  data after SRHT rotation + permutation
"""
import os, sys, struct, time, argparse
import numpy as np

source = '/data/dataset'


def read_vector_file(path):
    _, ext = os.path.splitext(path)
    if ext == ".fvecs":
        return read_fvecs(path)
    return read_fbin(path)


def read_fvecs(path, c_contiguous=True):
    fv = np.fromfile(path, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    fv = fv.reshape(-1, 1 + dim)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


def read_fbin(path, start_idx=0, chunk_size=None):
    with open(path, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32,
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)


def to_fvecs(path, data):
    print(f"Writing {path}")
    with open(path, "wb") as fp:
        for row in data:
            fp.write(struct.pack("I", len(row)))
            fp.write(np.asarray(row, dtype=np.float32).tobytes())


def fwht_inplace(X):
    """In-place FWHT along last axis. MUTATES X (must be float32 C-contiguous)."""
    assert X.dtype == np.float32 and X.flags['C_CONTIGUOUS']
    D = X.shape[-1]
    h = 1
    while h < D:
        v = X.reshape(-1, D // (2 * h), 2, h)
        a = v[..., 0, :].copy()                          # only 0.5x X temp
        np.add(a, v[..., 1, :], out=v[..., 0, :])
        np.subtract(a, v[..., 1, :], out=v[..., 1, :])
        h <<= 1
    return X


def srht(X, rad):
    """In-place: X = (1/sqrt(D)) * H * D_rad * X. MUTATES X."""
    D = X.shape[-1]
    np.multiply(X, rad[None, :], out=X)
    fwht_inplace(X)
    X *= np.float32(1.0 / np.sqrt(D))
    return X


def chunked_meansq(Y, chunk_rows=200000):
    """Compute (Y**2).mean(axis=0) without ever allocating a full Y**2 temp."""
    N, D = Y.shape
    sumsq = np.zeros(D, dtype=np.float64)
    for i in range(0, N, chunk_rows):
        block = Y[i:i + chunk_rows]
        sumsq += np.einsum('ij,ij->j', block, block)   # sum of squares per col
    return (sumsq / N).astype(np.float32)


def next_pow2(n):
    p = 1
    while p < n: p <<= 1
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SRHT + PCA-permutation preprocessing")
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-n", "--name",    required=True)
    parser.add_argument("-q", "--query", default="")
    args = parser.parse_args()

    np.random.seed(0)
    path = os.path.join(source, args.dataset)
    data_path = os.path.join(path, args.name)
    if not os.path.exists(data_path):
        raise SystemExit(f"missing: {data_path}")

    print(f"Reading {data_path}")
    X = read_vector_file(data_path)
    N, D_in = X.shape
    D = next_pow2(D_in)
    if D != D_in:
        print(f"Padding D={D_in} -> D={D}")
        X = np.pad(X, ((0, 0), (0, D - D_in)), mode="constant").astype(np.float32)

    rad = np.random.choice(np.array([-1.0, 1.0], dtype=np.float32), size=D)
    print(f"Applying SRHT (D={D}, N={N})")
    t0 = time.time()
    X = np.ascontiguousarray(X, dtype=np.float32)
    Y = srht(X, rad)        # in-place: Y is just X renamed
    print(f"  done in {time.time()-t0:.1f}s")

    print("Computing per-dim variance + permutation by sigma_i^2 desc...")
    var = chunked_meansq(Y)                          # avoids 1x Y**2 temp
    perm = np.argsort(var)[::-1].astype(np.int32)
    print(f"  post-SRHT var max/min = {var.max()/max(var.min(),1e-30):.2f}")

    Y_perm = Y[:, perm]
    del Y                                            # free 1x X before write

    out_dir = os.path.join(path, "srht_pca")
    os.makedirs(out_dir, exist_ok=True)
    arr = np.stack([rad.astype(np.float32), perm.astype(np.float32)], axis=0)
    to_fvecs(os.path.join(out_dir, "S.fvecs"), arr)
    to_fvecs(os.path.join(path, f"T{args.dataset}_base.fvecs"), Y_perm)
    del Y_perm                                        # free before optional query padding

    # new_add: optionally pad query to D (no rotation — runtime applies SRHT+perm)
    if args.query:
        qpath = os.path.join(path, args.query)
        if not os.path.exists(qpath):
            raise SystemExit(f"missing query: {qpath}")
        print(f"Reading query from {qpath}")
        Q = read_vector_file(qpath)
        Mq, Dq_in = Q.shape
        if Dq_in != D_in:
            raise SystemExit(f"query D={Dq_in} != base D_in={D_in}")
        if D != Dq_in:
            print(f"Padding query D_in={Dq_in} -> D={D}")
            Q = np.pad(Q, ((0, 0), (0, D - Dq_in)), mode="constant").astype(np.float32)
        squery_path = os.path.join(path, "Squery.fvecs")
        to_fvecs(squery_path, Q)
        print(f"  padded query : {squery_path}")

    print("done.")
