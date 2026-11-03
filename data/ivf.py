import numpy as np
import faiss
import struct
import os
import argparse
import time

source = '/home/zzlin/dataset'
dataset = 'YouTube'
# the number of clusters
K = 4096


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


if __name__ == '__main__':
    print(f"Clustering - {dataset}")
    parser = argparse.ArgumentParser(description='ivf build')
    parser.add_argument('-d', '--dataset', help='dataset', default='')
    parser.add_argument('-n', '--name', help='dataset name', default='')
    parser.add_argument('-k', '--num', help='cluster number', default=4096)
    args = vars(parser.parse_args())
    dataset = args['dataset']
    name = args['name']
    K = args['num']
    # path
    start_time = time.time()
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{name}')
    centroids_path = os.path.join(path, f'{dataset}_centroid_{K}.fvecs')

    # read data vectors
    X = read_vector_file(data_path)
    D = X.shape[1]

    # cluster data vectors
    index = faiss.index_factory(D, f"IVF{K},Flat")
    index.verbose = True
    index.train(X)
    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    to_fvecs(centroids_path, centroids)

    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000
    print(f"Elapsed time: {elapsed_time_ms:.2f} ms")
