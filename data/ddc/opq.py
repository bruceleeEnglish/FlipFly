import numpy as np
import faiss
import struct
import os
from utils import read_vector_file, fvecs_write
import argparse
import time

source = '/data/dataset'
M = 64
nbits = 8


def save_centroid(filename, data):
    print(f"Writing centroid file - {filename}")
    M, k, d = data.shape
    print(f"M: {M}")
    print(f"k: {k}")
    print(f"d: {d}")
    with open(filename, 'wb') as fp:
        item = struct.pack('I', M)
        fp.write(item)
        item = struct.pack('I', k)
        fp.write(item)
        item = struct.pack('I', d)
        fp.write(item)
        for x in data:
            for y in x:
                for z in y:
                    a = struct.pack('f', z)
                    fp.write(a)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='opq projection')
    parser.add_argument('-d', '--dataset', help='dataset', default='')
    parser.add_argument('-n', '--name', help='dataset name', default='')
    parser.add_argument('-b', '--bits', help='cluster bits', default=8)
    parser.add_argument('-l', '--normalize', help='cluster bits', default=0)
    args = vars(parser.parse_args())
    dataset = args['dataset']
    name = args['name']
    nbits = int(args['bits'])
    normalize = int(args['normalize'])
    if not dataset:
        raise ValueError("Dataset cannot be empty.")
    if not name:
        raise ValueError("Name cannot be empty.")
    print(f"OPQ - {dataset}")
    print(f"OPQ - {name}")

    start_time = time.time()
    # path
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f'{name}')
    # read data vectors
    X_base = read_vector_file(data_path)
    if normalize == 1:
        X_base = X_base / np.linalg.norm(X_base, axis=1, keepdims=True)
        print(f"OPQ - Normalize")
    d = X_base.shape[1]
    M = 64
    # if dataset == "gist":
    #     M = int(d / 8)
    # elif dataset == "_tiny80M":
    #     M = int(d / 3)
    # else:
    #     M = int(d / 4)
    if d > 8192:
        M = int(d / 64)
    elif d > 2048:
        M = int(d / 32)
    elif d > 512:
        M = int(d / 8)
    else:
        M = int(d / 4)
    d2 = ((d + M - 1) // M) * M
    opq = faiss.OPQMatrix(d, M, d2)
    opq.verbose = True
    N, D = X_base.shape
    print(f"OPQ - N: {N}")
    print(f"OPQ - D: {D}")
    opq.train(X_base)
    Matrix_A = faiss.vector_float_to_array(opq.A)
    Matrix_A = Matrix_A.reshape(d2, d2)
    # save the transpose matrix
    P_path = os.path.join(path, 'ddc')
    matrix_save_path = os.path.join(P_path, 'opq_matrix.fvecs')
    if normalize == 1:
        matrix_save_path = os.path.join(P_path, 'opq_matrix_normalize.fvecs')
    fvecs_write(matrix_save_path, Matrix_A.T)
    X_base = opq.apply(X_base)
    pq = faiss.ProductQuantizer(d2, M, nbits)
    pq.verbose = True
    pq.train(X_base)
    centroids = faiss.vector_float_to_array(pq.centroids)
    centroids = centroids.reshape(pq.M, pq.ksub, pq.dsub)
    print(f"pq.M: {pq.M}")
    print(f"pq.ksub: {pq.ksub}")
    print(f"pq.dsub: {pq.dsub}")
    centroid_save_path = os.path.join(P_path, 'codebook.centroid')
    if normalize == 1:
        centroid_save_path = os.path.join(P_path, 'codebook_normalize.centroid')
    save_centroid(centroid_save_path, centroids)
    save_base_path = os.path.join(path, f'Q{dataset}_base.fvecs')
    if normalize == 1:
        save_base_path = os.path.join(path, f'Q{dataset}_base_normalize.fvecs')
    fvecs_write(save_base_path, X_base)

    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000
    print(f"Elapsed time: {elapsed_time_ms:.2f} ms")
    print(M)
    print(pq.M)
