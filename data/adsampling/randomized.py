import os
import time
import numpy as np
import struct
import argparse

source = '/data/dataset'


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
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('f', x)
                fp.write(a)


def Orthogonal(D):
    G = np.random.randn(D, D).astype('float32')
    Q, _ = np.linalg.qr(G)
    return Q


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='adsampling randomized')
    parser.add_argument('-d', '--dataset', help='dataset', default='')
    parser.add_argument('-n', '--name', help='dataset name', default='')
    args = vars(parser.parse_args())
    dataset = args['dataset']
    name = args['name']

    if not dataset:
        raise ValueError("Dataset cannot be empty.")
    if not name:
        raise ValueError("Name cannot be empty.")

    start_time = time.time()

    np.random.seed(0)

    # path
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, name)

    # check if dataset exists
    if not os.path.exists(data_path):
        raise ValueError(f'{dataset} does not exist.')

    # read data vectors
    print(f"Reading {dataset} from {data_path}.")
    X = read_vector_file(data_path)
    D = X.shape[1]

    # generate random orthogonal matrix, store it and apply it
    print(f"Randomizing {dataset} of dimensionality {D}.")
    P = Orthogonal(D)
    XP = np.dot(X, P)

    O_path = os.path.join(path, 'adsampling')
    projection_path = os.path.join(O_path, 'O.fvecs')
    transformed_path = os.path.join(path, f'O{dataset}_base.fvecs')

    os.makedirs(O_path, exist_ok=True)
    to_fvecs(projection_path, P)
    to_fvecs(transformed_path, XP)

    # elapsed time
    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000
    print(f"Elapsed time: {elapsed_time_ms:.2f} ms")
