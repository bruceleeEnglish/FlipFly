import os
import numpy as np
import struct
import copy
import time
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


def PCA(x):
    x = copy.deepcopy(np.asmatrix(x))
    dim, num = x.shape[0], x.shape[1]
    print(f"PCA - {dim} x {num}")
    x -= x.mean(axis=1)
    lmd, w = np.linalg.eig(x * x.T)
    return np.array(w.astype('float32')), np.array(lmd.astype('float32'))[np.newaxis, :]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dade pca')
    parser.add_argument('-d', '--dataset', help='dataset', default='')
    parser.add_argument('-n', '--name', help='dataset name', default='')
    parser.add_argument('-s', '--size', help='dataset size', default='all')
    parser.add_argument('-l', '--normalize', help='if normalize', default='0')
    args = vars(parser.parse_args())
    dataset = args['dataset']
    name = args['name']
    size = args['size']
    normalize = int(args['normalize'])

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

    if normalize:
        print("Normalizing data.")
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

    # generate pca orthogonal matrix & eigenvalues, store it and apply it
    print(f"Randomizing {dataset} of dimensionality {D}.")
    X_train = X
    if size != 'all':
        path = os.path.join(path, 'train_' + size)
        os.makedirs(path, exist_ok=True)
        X_train = X[:int(size), :]
    P, LMD = PCA(X_train.T)
    XP = np.dot(X, P)

    suffix = ''
    if normalize:
        suffix = '_normalized'

    P_path = os.path.join(path, 'dade')
    projection_path = os.path.join(P_path, f'P{suffix}.fvecs')
    lambda_path = os.path.join(P_path, f'LMD{suffix}.fvecs')
    transformed_path = os.path.join(path, f'P{dataset}{suffix}_base.fvecs')

    os.makedirs(P_path, exist_ok=True)
    to_fvecs(projection_path, P)
    to_fvecs(lambda_path, LMD)
    to_fvecs(transformed_path, XP)

    # elapsed time
    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000
    print(f"Elapsed time: {elapsed_time_ms:.2f} ms")
