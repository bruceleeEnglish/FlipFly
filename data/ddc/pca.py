import numpy as np
from utils import read_vector_file, fvecs_write, ivecs_read
import os
import argparse
import time
import struct

source = '/data/dataset'

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='pca projection')
    parser.add_argument('-d', '--dataset', help='dataset', default='')
    parser.add_argument('-n', '--name', help='dataset name', default='')
    parser.add_argument('-l', '--learn_ground_truth', help='learn_ground_truth', default='')
    args = vars(parser.parse_args())
    dataset = args['dataset']
    name = args['name']
    learn_ground_truth = args['learn_ground_truth']
    if not dataset:
        raise ValueError("Dataset cannot be empty.")
    if not name:
        raise ValueError("Name cannot be empty.")
    if not learn_ground_truth:
        raise ValueError("Learn_ground_truth cannot be empty.")
    print(f"PCA - {dataset}")
    print(f"PCA - {name}")
    print(f"PCA - {learn_ground_truth}")

    start_time = time.time()
    # path
    path = os.path.join(source, dataset)
    base_path = os.path.join(path, f'{name}')
    ground_path = os.path.join(path, f'{learn_ground_truth}')
    # read data vectors
    base = read_vector_file(base_path)
    N, D = base.shape
    pca_dim = D
    print(f"PCA - {N} vectors, {D} dimensions")

    # projection
    # 1. Center the data by subtracting the mean
    mean = np.mean(base, axis=0)
    base -= mean

    # ------------------ NumPy PCA Implementation Start ------------------
    # Determine the subset of data for fitting PCA based on dataset size
    fit_data = base
    if D > 1024 and N > 1000000:
        fit_data = base[:1000000]
        print(f"PCA - fitting on first {1000000} vectors")
    elif N >= 10000000:
        fit_data = base[:10000000]
        print(f"PCA - fitting on first {10000000} vectors")
    else:
        print(f"PCA - fitting on all {N} vectors")

    # 2. Calculate the covariance matrix.
    # 'rowvar=False' signifies that columns are variables (dimensions) and rows are observations.
    print("PCA - Calculating covariance matrix...")
    cov_matrix = np.cov(fit_data, rowvar=False)

    # 3. Compute eigenvalues and eigenvectors of the covariance matrix.
    print("PCA - Performing eigendecomposition...")
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 4. Sort eigenvectors in descending order of their corresponding eigenvalues.
    # np.linalg.eigh returns eigenvalues in ascending order, so we reverse the sort.
    sorted_indices = np.argsort(eigenvalues)[::-1]
    projection_matrix = eigenvectors[:, sorted_indices]
    # ------------------- NumPy PCA Implementation End -------------------

    # Transform the centered data using the new projection matrix
    base = np.dot(base, projection_matrix.astype(np.float32))
    print(f"PCA - finished")

    P_path = os.path.join(path, 'ddc')

    matrix_save_path = os.path.join(P_path, 'pca_matrix.fvecs')
    save_base_path = os.path.join(path, f'L{dataset}_base.fvecs')

    variance = np.var(base, axis=0)
    save_matrix = np.vstack((mean, mean, variance, projection_matrix))
    os.makedirs(P_path, exist_ok=True)
    fvecs_write(matrix_save_path, save_matrix)
    fvecs_write(save_base_path, base)

    for K in [20, 40, 60, 100]:
        matrix_save_path = os.path.join(P_path, f'pca_matrix_{K}.fvecs')
        ground = ivecs_read(ground_path)
        ground = ground[:int(1e4), :K]
        ground = ground.flatten()
        X_sample = base[ground]
        sample_mean = np.mean(X_sample, axis=0)
        X_sample -= sample_mean
        variance = np.var(X_sample, axis=0)
        save_matrix = np.vstack((mean, sample_mean, variance, projection_matrix))
        fvecs_write(matrix_save_path, save_matrix)

    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000
    print(f"Elapsed time: {elapsed_time_ms:.2f} ms")