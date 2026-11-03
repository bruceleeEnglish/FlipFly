# new_add: SRHT preprocessing (not in upstream dco-benchmark)
"""
SRHT preprocessing for the dco-benchmark.

Generates the +/-1 Rademacher diagonal D_rad and applies SRHT to all data:
    Y[i] = (1 / sqrt(D)) * H_D * D_rad * X[i]

Outputs (mirroring the ADSampling layout in ../adsampling/):
    <dataset_dir>/srht/S.fvecs           one row of length D = the diagonal
    <dataset_dir>/S<dataset>_base.fvecs  rotated data, same count and dim as input

Usage:
    python randomized.py -d <dataset_subdir> -n <basename.fvecs>

Constraint:
    D must be a power of 2. We pad with zeros to the next power of 2 if it isn't.
"""
import os
import time
import struct
import argparse
import numpy as np

source = '/data/dataset'


# ---------- file IO (same helpers as adsampling/randomized.py) ----------

def read_vector_file(filename):
    _, ext = os.path.splitext(filename)
    if ext == '.fvecs':
        return read_fvecs(filename)
    else:
        return read_fbin(filename)


def read_fvecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


def read_fbin(filename, start_idx=0, chunk_size=None):
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32,
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)


def to_fvecs(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        for y in data:
            y = np.asarray(y, dtype=np.float32)
            d = struct.pack('I', len(y))
            fp.write(d)
            fp.write(y.tobytes())


# ---------- SRHT specific ----------

def next_power_of_two(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def fwht_inplace(X):
    """
    Batched in-place Fast Walsh-Hadamard Transform along last axis.
    Returns unnormalized H_D * x for each row. MUTATES X.
    Memory: only one 0.5*X temporary per stage (the saved 'a' slice);
    butterfly writes use np.add/subtract with out= to avoid extra allocations.
    """
    assert X.dtype == np.float32 and X.flags['C_CONTIGUOUS']
    D = X.shape[-1]
    h = 1
    while h < D:
        v = X.reshape(-1, D // (2 * h), 2, h)
        a = v[..., 0, :].copy()                       # 0.5x X (only allocation here)
        np.add(a, v[..., 1, :], out=v[..., 0, :])     # slot0 = a + b
        np.subtract(a, v[..., 1, :], out=v[..., 1, :])# slot1 = a - b
        h <<= 1
    return X


def apply_srht(X, rad):
    """
    In-place: X = (1/sqrt(D)) * H * D_rad * X    applied row-wise. MUTATES X.
    """
    D = X.shape[-1]
    np.multiply(X, rad[None, :], out=X)         # X *= D_rad row-wise (no new alloc)
    fwht_inplace(X)                              # H * (...) in place
    X *= np.float32(1.0 / np.sqrt(D))            # normalize in place
    return X


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SRHT-ADSampling preprocessing')
    parser.add_argument('-d', '--dataset', help='dataset subdir', default='')
    parser.add_argument('-n', '--name', help='base file name', default='')
    parser.add_argument('-q', '--query', help='optional query file to pad', default='')
    args = vars(parser.parse_args())
    dataset = args['dataset']
    name = args['name']
    query = args['query']

    if not dataset:
        raise ValueError("Dataset cannot be empty.")
    if not name:
        raise ValueError("Name cannot be empty.")

    np.random.seed(0)
    start_time = time.time()

    path = os.path.join(source, dataset)
    data_path = os.path.join(path, name)
    if not os.path.exists(data_path):
        raise ValueError(f'{dataset} does not exist at {data_path}.')

    print(f"Reading {dataset} from {data_path}.")
    X = read_vector_file(data_path)
    N, D_in = X.shape

    D = next_power_of_two(D_in)
    if D != D_in:
        print(f"Padding from D_in={D_in} to next power of 2 D={D}")
        X = np.pad(X, ((0, 0), (0, D - D_in)), mode='constant').astype(np.float32)

    # Rademacher diagonal
    rad = np.random.choice(np.array([-1.0, 1.0], dtype=np.float32), size=D)

    print(f"Applying SRHT with D={D} on {N} vectors.")
    # apply_srht mutates X in-place; XS is the same array as X (just renamed).
    X = np.ascontiguousarray(X, dtype=np.float32)
    XS = apply_srht(X, rad)

    # output dirs/paths matching ADSampling convention
    S_dir = os.path.join(path, 'srht')
    os.makedirs(S_dir, exist_ok=True)
    S_path = os.path.join(S_dir, 'S.fvecs')
    transformed_path = os.path.join(path, f'S{dataset}_base.fvecs')

    # store the diagonal as a single fvecs row of length D
    to_fvecs(S_path, rad[None, :])
    to_fvecs(transformed_path, XS)


    # new_add: optionally pad query to D (no SRHT rotation — runtime applies it)
    if query:
        query_path = os.path.join(path, query)
        if not os.path.exists(query_path):
            raise ValueError(f'query {query_path} does not exist.')
        print(f"Reading query from {query_path}.")
        Q = read_vector_file(query_path)
        Mq, Dq_in = Q.shape
        if Dq_in != D_in:
            raise ValueError(f'query D={Dq_in} != base D_in={D_in}')
        if D != Dq_in:
            print(f"Padding query from D_in={Dq_in} to D={D}")
            Q = np.pad(Q, ((0, 0), (0, D - Dq_in)), mode='constant').astype(np.float32)
        squery_path = os.path.join(path, 'Squery.fvecs')
        to_fvecs(squery_path, Q)
        print(f"  padded query : {squery_path}")

    elapsed_ms = (time.time() - start_time) * 1000
    print(f"Elapsed: {elapsed_ms:.2f} ms")
    print(f"  diagonal     : {S_path}")
    print(f"  rotated data : {transformed_path}")
