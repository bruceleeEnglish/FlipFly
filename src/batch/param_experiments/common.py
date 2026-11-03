"""Shared helpers for the §5.2 verification experiments.

* fvecs / raw-bin readers (streaming + bulk variants)
* in-place Fast Walsh--Hadamard Transform on float32 arrays
* SRHT projection wrapper
* dataset path config
"""
import os

import numpy as np


# --------------------------------------------------------------------------- #
# Dataset paths on the server. Override via CLI in each script if needed.
# Each dataset entry: (base_filename, query_filename, base_format, base_dim,
#                      srht_rotated_filename)
# base_format ∈ {"fvecs", "bin"}. For "bin" we need to know the dimension
# in advance.
# --------------------------------------------------------------------------- #
DATASETS = ["trevi", "arxiv-for-fanns-large", "XUltra", "OpenAI3072"]

DATASET_TITLES = {
    "trevi": "Trevi",
    "arxiv-for-fanns-large": "Arxiv",
    "XUltra": "XUltra",
    "OpenAI3072": "OpenAI3072",
}

DATASET_FILES = {
    # dataset -> (base_filename, query_filename, base_format, base_dim, srht_rotated_filename)
    "trevi": (
        "trevi_base.fvecs", "query.fvecs", "fvecs", 4096,
        "Strevi_base.fvecs",
    ),
    "arxiv-for-fanns-large": (
        "arxiv-for-fanns-large_base.fvecs", "query.fvecs", "fvecs", 4096,
        "Sarxiv-for-fanns-large_base.fvecs",
    ),
    "XUltra": (
        "msmacro_base-001.bin", "query.bin", "bin", 12288,
        "SXUltra_base.fvecs",
    ),
    "OpenAI3072": (
        "openai3072_base.bin", "query.bin", "bin", 3072,
        "SOpenAI3072_base.fvecs",
    ),
}


# --------------------------------------------------------------------------- #
# fvecs I/O
# --------------------------------------------------------------------------- #
def fvecs_dim(path):
    """Return the dimension stored in the first 4 bytes of an fvecs file."""
    with open(path, "rb") as f:
        return int(np.frombuffer(f.read(4), dtype=np.int32)[0])


def read_fvecs(path, max_n=None):
    """Read an entire .fvecs file as a contiguous (N, D) float32 array."""
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"empty fvecs file {path}")
    D = int(raw[0])
    record_size = D + 1
    N_total = raw.size // record_size
    if max_n is not None and max_n < N_total:
        N_total = max_n
    arr = raw[:N_total * record_size].reshape(N_total, record_size)
    return arr[:, 1:].view(np.float32).copy()


def stream_fvecs(path, chunk_size=10000, max_n=None):
    """Yield successive (chunk, D) float32 arrays, total <= max_n rows."""
    D = fvecs_dim(path)
    record_bytes = (D + 1) * 4
    yielded = 0
    with open(path, "rb") as f:
        while True:
            buf = f.read(record_bytes * chunk_size)
            if not buf:
                break
            n = len(buf) // record_bytes
            arr = np.frombuffer(buf, dtype=np.int32).reshape(n, D + 1)
            data = arr[:, 1:].view(np.float32)
            if max_n is not None and yielded + n > max_n:
                data = data[: max_n - yielded]
            yield data.copy()
            yielded += data.shape[0]
            if max_n is not None and yielded >= max_n:
                break


def read_bin(path, dim, max_n=None):
    """Read a raw float32 .bin file with no length prefix; dim is fixed."""
    raw = np.fromfile(path, dtype=np.float32)
    N_total = raw.size // dim
    if max_n is not None and max_n < N_total:
        N_total = max_n
    return raw[: N_total * dim].reshape(N_total, dim).copy()


def stream_bin(path, dim, chunk_size=10000, max_n=None):
    record_bytes = dim * 4
    yielded = 0
    with open(path, "rb") as f:
        while True:
            buf = f.read(record_bytes * chunk_size)
            if not buf:
                break
            n = len(buf) // record_bytes
            data = np.frombuffer(buf, dtype=np.float32).reshape(n, dim)
            if max_n is not None and yielded + n > max_n:
                data = data[: max_n - yielded]
            yield data.copy()
            yielded += data.shape[0]
            if max_n is not None and yielded >= max_n:
                break


def read_dataset_base(data_root, dataset, max_n=None):
    """Auto-dispatch to read_fvecs / read_bin based on DATASET_FILES."""
    base_name, _, fmt, dim, _ = DATASET_FILES[dataset]
    path = os.path.join(data_root, dataset, base_name)
    if fmt == "fvecs":
        return read_fvecs(path, max_n=max_n)
    return read_bin(path, dim, max_n=max_n)


def read_dataset_query(data_root, dataset, max_n=None):
    _, q_name, fmt, dim, _ = DATASET_FILES[dataset]
    path = os.path.join(data_root, dataset, q_name)
    if fmt == "fvecs":
        return read_fvecs(path, max_n=max_n)
    return read_bin(path, dim, max_n=max_n)


def stream_srht_base(data_root, dataset, chunk_size=10000, max_n=None):
    """Stream the pre-rotated S<ds>_base.fvecs file."""
    _, _, _, _, srht_name = DATASET_FILES[dataset]
    path = os.path.join(data_root, dataset, srht_name)
    yield from stream_fvecs(path, chunk_size=chunk_size, max_n=max_n)


# --------------------------------------------------------------------------- #
# FWHT and SRHT
# --------------------------------------------------------------------------- #
def next_power_of_two(n):
    p = 1
    while p < n:
        p *= 2
    return p


def pad_to_pow2(X):
    """Right-zero-pad along last axis to the next power of 2.
    Returns the padded float32 array; copy is forced so caller can mutate."""
    if X.ndim == 1:
        X = X.reshape(1, -1)
    D_in = X.shape[1]
    D = next_power_of_two(D_in)
    if D == D_in:
        return np.ascontiguousarray(X, dtype=np.float32).copy()
    Z = np.zeros((X.shape[0], D), dtype=np.float32)
    Z[:, :D_in] = X.astype(np.float32, copy=False)
    return Z


def fwht_inplace(X):
    """In-place fast Walsh--Hadamard transform along the last axis.

    Operates on a float32 2-D array of shape (N, D) where D is a power
    of 2. Returns the *same array* (mutated). For convenience the
    function also accepts a 1-D array.

    Each level reshapes (N, D) -> (N, D/(2h), 2, h) and applies one
    butterfly via two numpy slice assignments, avoiding the per-block
    Python loop of a naive implementation. For D up to 65536 this is
    fast enough that the wall-clock cost scales like O(D log D) as
    expected (instead of being Python-loop dominated).

    The transform is unnormalised:
        FWHT(x)[i] = sum_j H_{ij} x_j
    where H is the Walsh--Hadamard matrix with H[0,0] = +1.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    N, D = X.shape
    if D & (D - 1) != 0:
        raise ValueError(f"FWHT requires power-of-2 dimension, got D={D}")
    h = 1
    while h < D:
        # Reshape into (N, D/(2h), 2, h) so we can butterfly an entire
        # level in one vectorised op.
        Y = X.reshape(N, D // (2 * h), 2, h)
        a = Y[:, :, 0, :].copy()
        b = Y[:, :, 1, :].copy()
        Y[:, :, 0, :] = a + b
        Y[:, :, 1, :] = a - b
        h *= 2
    return X


def srht_project(X, rad):
    """Apply SRHT y = (1/sqrt(D)) * H_D * D_rad * x to every row of X.

    X    : (N, D) float32, D = power of 2
    rad  : (D,)   ±1 float32
    Returns a fresh (N, D) float32 array.
    """
    D = X.shape[-1]
    Y = X * rad
    Y = fwht_inplace(Y)
    Y *= np.float32(1.0 / np.sqrt(D))
    return Y
