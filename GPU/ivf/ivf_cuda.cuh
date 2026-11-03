#pragma once
#ifndef IVF_CUDA_CUH
#define IVF_CUDA_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cstring>
#include <math.h>
#include <queue>
#include <vector>
#include <limits>
#include <algorithm>
#include <fstream>
#include <string>
#include <numeric>
#include "../../Logger/Logger.h"
#include "../../FileIO/File_IO.h"

#define INF INFINITY

const size_t delta_d = 32;
const float epsilon0 = 2.1;
const float sigma = 8.0;

// CUDA check
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                     \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

/////////////////// Kernel ///////////////////

__global__ void MatrixMulKernel(const float * __restrict__ d_matrix, float *d_query, float * __restrict__ d_result, size_t m) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < m) {
        float sum = 0.0f;
        for (size_t k = 0; k < m; ++k) {
            sum += d_query[k] * d_matrix[k * m + idx];
        }
        d_result[idx] = sum;
    }
}

__global__ void DistMapKernel(float *d_codebook, float *d_query, float * __restrict__ d_result, size_t sub_vector,
                              size_t sub_cluster_count, size_t sub_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sub_vector) {
        float *query = d_query + idx * sub_dim;
        float *matrix = d_codebook + idx * sub_cluster_count * sub_dim;
        float *result = d_result + idx * sub_cluster_count;
        for (size_t k = 0; k < sub_cluster_count; ++k) {
            float sum_sq_diff = 0.0f;
            for (size_t i = 0; i < sub_vector; ++i) {
                float diff = matrix[i] - query[i];
                sum_sq_diff += diff * diff;
            }
            result[k] = sum_sq_diff;
            matrix += sub_dim;
        }
    }
}

__global__ void DistMap4ExtKernel(float *d_codebook, float *d_query, float * __restrict__ d_result, size_t sub_vector,
                                  size_t sub_cluster_count, size_t sub_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sub_vector) {
        float *query = d_query + idx * sub_dim;
        float *matrix = d_codebook + idx * sub_cluster_count * sub_dim;
        float *result = d_result + idx * sub_cluster_count;
        float4 *matrix_vec = reinterpret_cast<float4 *>(matrix);
        float4 *query_vec = reinterpret_cast<float4 *>(query);
        size_t new_m = sub_dim / 4;
        for (size_t k = 0; k < sub_cluster_count; ++k) {
            float sum_sq_diff = 0.0f;

            for (size_t i = 0; i < new_m; ++i) {
                float4 m_val = matrix_vec[i];
                float4 q_val = query_vec[i];

                float diff1 = m_val.x - q_val.x;
                float diff2 = m_val.y - q_val.y;
                float diff3 = m_val.z - q_val.z;
                float diff4 = m_val.w - q_val.w;

                sum_sq_diff += diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4;
            }
            result[k] = sum_sq_diff;
            matrix_vec += new_m;
        }
    }
}

/**
 * FDScanningKernel
 * @param d_matrix : n * m dataset
 * @param d_query : m * 1 query
 * @param d_result : n * 1 result
 * @param n : number of rows
 * @param m : number of columns
 */
__global__ void FDScanningKernel(const float * __restrict__ d_matrix, const float * __restrict__ d_query, float * __restrict__ d_result, size_t n, size_t m) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float sum_sq_diff = 0.0f;
        for (size_t i = 0; i < m; ++i) {
            float diff = d_matrix[idx * m + i] - d_query[i];
            sum_sq_diff += diff * diff;
        }
        d_result[idx] = sum_sq_diff;
    }
}

__global__ void FDScanning4ExtKernel(const float * __restrict__ d_matrix, const float * __restrict__ d_query, float * __restrict__ d_result, size_t n, size_t m) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float sum_sq_diff = 0.0f;
        const float4 *matrix_vec = reinterpret_cast<const float4 *>(&d_matrix[idx * m]);
        const float4 *query_vec = reinterpret_cast<const float4 *>(d_query);

        for (size_t i = 0; i < m / 4; ++i) {
            float4 m_val = matrix_vec[i];
            float4 q_val = query_vec[i];

            float diff1 = m_val.x - q_val.x;
            float diff2 = m_val.y - q_val.y;
            float diff3 = m_val.z - q_val.z;
            float diff4 = m_val.w - q_val.w;

            sum_sq_diff += diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4;
        }
        d_result[idx] = sum_sq_diff;
    }
}

/**
 * PcanningKernel
 * @param d_matrix : n * m dataset
 * @param d_query : m * 1 query
 * @param d_result : n * 1 result
 * @param n : number of rows
 * @param m : number of columns
 * @param threshold : threshold
 */
__global__ void
PDScanningKernel(const float * __restrict__ d_matrix, const float * __restrict__ d_query, float * __restrict__ d_result, size_t n, size_t m, float threshold) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float sum_sq_diff = 0.0f;
        size_t i = 0;
        while (i < m) {
            size_t check = (delta_d < m - i) ? delta_d : m - i;
            for (size_t j = 0; j < check; ++j) {
                float diff = d_matrix[idx * m + i + j] - d_query[i + j];
                sum_sq_diff += diff * diff;
            }
            if (sum_sq_diff >= threshold) {
                sum_sq_diff = INF;
                break;
            }
            i += check;
        }
        d_result[idx] = sum_sq_diff;
    }
}

__global__ void
PDScanning4ExtKernel(const float * __restrict__ d_matrix, const float * __restrict__ d_query, float * __restrict__ d_result, size_t n, size_t m,
                     float threshold) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float sum_sq_diff = 0.0f;
        const float4 *matrix_vec = reinterpret_cast<const float4 *>(&d_matrix[idx * m]);
        const float4 *query_vec = reinterpret_cast<const float4 *>(d_query);
        size_t i = 0;
        while (i < m / 4) {
            size_t check = (delta_d / 4 < m / 4 - i) ? delta_d / 4 : m / 4 - i;
            for (size_t j = 0; j < check; ++j) {
                float4 m_val = matrix_vec[i + j];
                float4 q_val = query_vec[i + j];

                float diff1 = m_val.x - q_val.x;
                float diff2 = m_val.y - q_val.y;
                float diff3 = m_val.z - q_val.z;
                float diff4 = m_val.w - q_val.w;

                sum_sq_diff += diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4;
            }
            if (sum_sq_diff >= threshold) {
                sum_sq_diff = INF;
                break;
            }
            i += check;
        }
        d_result[idx] = sum_sq_diff;
    }
}

/**
 * ADSamplingKernel
 * @param d_matrix : n * m dataset
 * @param d_query : m * 1 query
 * @param d_result : n * 1 result
 * @param n : number of rows
 * @param m : number of columns
 * @param threshold : threshold
 */
__global__ void
ADSamplingKernel(const float * __restrict__ d_matrix, const float * __restrict__ d_query, float * __restrict__ d_result, size_t n, size_t m, float threshold) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float sum_sq_diff = 0.0f;
        size_t i = 0;
        while (i < m) {
            size_t check = (delta_d < m - i) ? delta_d : m - i;
            for (size_t j = 0; j < check; ++j) {
                float diff = d_matrix[idx * m + i + j] - d_query[i + j];
                sum_sq_diff += diff * diff;
            }
            i += check;
            if (i == m) break;
            float sqrt_i = std::sqrt(static_cast<float>(i));
            float factor = (1.0f + epsilon0 / sqrt_i);
            if (sum_sq_diff >= threshold * i / m * factor * factor) {
                sum_sq_diff = INF;
                break;
            }
        }
        d_result[idx] = sum_sq_diff;
    }
}

__global__ void ADSampling4ExtKernel(const float * __restrict__ d_matrix, const float * __restrict__ d_query, float * __restrict__ d_result, size_t n, size_t m,
                                     float threshold) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float sum_sq_diff = 0.0f;
        const float4 *matrix_vec = reinterpret_cast<const float4 *>(&d_matrix[idx * m]);
        const float4 *query_vec = reinterpret_cast<const float4 *>(d_query);
        size_t i = 0;
        while (i < m / 4) {
            size_t check = (delta_d / 4 < m / 4 - i) ? delta_d / 4 : m / 4 - i;
            for (size_t j = 0; j < check; ++j) {
                float4 m_val = matrix_vec[i + j];
                float4 q_val = query_vec[i + j];

                float diff1 = m_val.x - q_val.x;
                float diff2 = m_val.y - q_val.y;
                float diff3 = m_val.z - q_val.z;
                float diff4 = m_val.w - q_val.w;

                sum_sq_diff += diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4;
            }
            i += check;
            if (i == m / 4) break;
            float sqrt_i = std::sqrt(static_cast<float>(4 * i));
            float factor = (1.0f + epsilon0 / sqrt_i);
            if (sum_sq_diff >= threshold * 4 * i / m * factor * factor) {
                sum_sq_diff = INF;
                break;
            }
        }
        d_result[idx] = sum_sq_diff;
    }
}

/**
 * DADEKernel
 * @param d_matrix : n * m dataset
 * @param d_query : m * 1 query
 * @param d_result : n * 1 result
 * @param n : number of rows
 * @param m : number of columns
 * @param threshold : threshold
 */
__global__ void DADEKernel(const float * __restrict__ d_matrix, const float * __restrict__ d_query, float * __restrict__ d_result, size_t n, size_t m,
                           float threshold, float *d_cdf_lmd, float *d_epsilon) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float sum_sq_diff = 0.0f;
        size_t i = 0;
        while (i < m) {
            size_t check = (delta_d < m - i) ? delta_d : m - i;
            for (size_t j = 0; j < check; ++j) {
                float diff = d_matrix[idx * m + i + j] - d_query[i + j];
                sum_sq_diff += diff * diff;
            }
            i += check;
            if (i == m) break;
            float factor = 1.0 + d_epsilon[i];
            if (sum_sq_diff >= threshold * d_cdf_lmd[i] / d_cdf_lmd[m] * factor * factor) {
                sum_sq_diff = INF;
                break;
            }
        }
        d_result[idx] = sum_sq_diff;
    }
}

__global__ void DADE4ExtKernel(const float * __restrict__ d_matrix, const float * __restrict__ d_query, float * __restrict__ d_result, size_t n, size_t m,
                               float threshold, float *d_cdf_lmd, float *d_epsilon) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float sum_sq_diff = 0.0f;
        const float4 *matrix_vec = reinterpret_cast<const float4 *>(&d_matrix[idx * m]);
        const float4 *query_vec = reinterpret_cast<const float4 *>(d_query);
        size_t i = 0;
        while (i < m / 4) {
            size_t check = (delta_d / 4 < m / 4 - i) ? delta_d / 4 : m / 4 - i;
            for (size_t j = 0; j < check; ++j) {
                float4 m_val = matrix_vec[i + j];
                float4 q_val = query_vec[i + j];

                float diff1 = m_val.x - q_val.x;
                float diff2 = m_val.y - q_val.y;
                float diff3 = m_val.z - q_val.z;
                float diff4 = m_val.w - q_val.w;

                sum_sq_diff += diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4;
            }
            i += check;
            if (i == m / 4) break;
            float factor = 1.0 + d_epsilon[4 * i];
            if (sum_sq_diff >= threshold * d_cdf_lmd[4 * i] / d_cdf_lmd[m] * factor * factor) {
                sum_sq_diff = INF;
                break;
            }
        }
        d_result[idx] = sum_sq_diff;
    }
}

/**
 * DDCresKernel
 * @param d_matrix : n * m dataset
 * @param d_query : m * 1 query
 * @param d_result : n * 1 result
 * @param n : number of rows
 * @param m : number of columns
 * @param threshold : threshold
 */
__global__ void DDCresKernel(const float * __restrict__ d_matrix, const float * __restrict__ d_query, float * __restrict__ d_result, size_t n, size_t m,
                             float threshold, float res, float *d_base_square, float *d_pre_query, size_t pos) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float sum_sq_diff = res + d_base_square[idx + pos];
        size_t i = 0;
        while (i < m) {
            size_t check = (delta_d < m - i) ? delta_d : m - i;
            for (size_t j = 0; j < check; ++j) {
                sum_sq_diff -= 2 * d_matrix[idx * m + i + j] - d_query[i + j];
            }
            i += check;
            if (sum_sq_diff - d_pre_query[i] >= threshold) {
                sum_sq_diff = INF;
                break;
            }
        }
        d_result[idx] = sum_sq_diff;
    }
}

__global__ void DDCres4ExtKernel(const float * __restrict__ d_matrix, const float * __restrict__ d_query, float * __restrict__ d_result, size_t n, size_t m,
                                 float threshold, float res, float *d_base_square, float *d_pre_query, size_t pos) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float sum_sq_diff = res + d_base_square[idx + pos];
        const float4 *matrix_vec = reinterpret_cast<const float4 *>(&d_matrix[idx * m]);
        const float4 *query_vec = reinterpret_cast<const float4 *>(d_query);
        size_t i = 0;
        while (i < m / 4) {
            size_t check = (delta_d / 4 < m / 4 - i) ? delta_d / 4 : m / 4 - i;
            for (size_t j = 0; j < check; ++j) {
                float4 m_val = matrix_vec[i + j];
                float4 q_val = query_vec[i + j];

                float diff1 = m_val.x * q_val.x;
                float diff2 = m_val.y * q_val.y;
                float diff3 = m_val.z * q_val.z;
                float diff4 = m_val.w * q_val.w;

                sum_sq_diff -= 2 * (diff1 + diff2 + diff3 + diff4);
            }
            i += check;
            if (sum_sq_diff - d_pre_query[4 * i] >= threshold) {
                sum_sq_diff = INF;
                break;
            }
        }
        d_result[idx] = sum_sq_diff;
    }
}

/**
 * DDCpcaKernel
 * @param d_matrix : n * m dataset
 * @param d_query : m * 1 query
 * @param d_result : n * 1 result
 * @param n : number of rows
 * @param m : number of columns
 * @param threshold : threshold
 */
__global__ void
DDCpcaKernel(const float * __restrict__ d_matrix, const float * __restrict__ d_query, float * __restrict__ d_result, size_t n, size_t m,
             float threshold, float *d_W, float *d_B) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        size_t tag_model = 0;
        float sum_sq_diff = 0.0f;
        size_t i = 0;
        while (i < m) {
            size_t check = (delta_d < m - i) ? delta_d : m - i;
            for (size_t j = 0; j < check; ++j) {
                float diff = d_matrix[idx * m + i + j] - d_query[i + j];
                sum_sq_diff += diff * diff;
            }
            i += check;
            if (sum_sq_diff * d_W[tag_model] + d_B[tag_model] >= threshold) {
                sum_sq_diff = INF;
                break;
            }
            tag_model++;
        }
        d_result[idx] = sum_sq_diff;
    }
}

__global__ void DDCpca4ExtKernel(const float * __restrict__ d_matrix, const float * __restrict__ d_query, float * __restrict__ d_result, size_t n, size_t m,
                                 float threshold, float *d_W, float *d_B) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        size_t tag_model = 0;
        float sum_sq_diff = 0.0f;
        const float4 *matrix_vec = reinterpret_cast<const float4 *>(&d_matrix[idx * m]);
        const float4 *query_vec = reinterpret_cast<const float4 *>(d_query);
        size_t i = 0;
        while (i < m / 4) {
            size_t check = (delta_d / 4 < m / 4 - i) ? delta_d / 4 : m / 4 - i;
            for (size_t j = 0; j < check; ++j) {
                float4 m_val = matrix_vec[i + j];
                float4 q_val = query_vec[i + j];

                float diff1 = m_val.x - q_val.x;
                float diff2 = m_val.y - q_val.y;
                float diff3 = m_val.z - q_val.z;
                float diff4 = m_val.w - q_val.w;

                sum_sq_diff += diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4;
            }
            i += check;
            if (sum_sq_diff * d_W[tag_model] + d_B[tag_model] >= threshold) {
                sum_sq_diff = INF;
                break;
            }
            tag_model++;
        }
        d_result[idx] = sum_sq_diff;
    }
}

/**
 * DDCopqKernel
 * @param d_matrix : n * m dataset
 * @param d_query : m * 1 query
 * @param d_result : n * 1 result
 * @param n : number of rows
 * @param m : number of columns
 * @param threshold : threshold
 */
__global__ void DDCopqKernel(const float * __restrict__ d_matrix, const float * __restrict__ d_query, float * __restrict__ d_result, size_t n, size_t m,
                             float threshold, float *d_W, float *d_B, size_t sub_vector, size_t sub_cluster_count,
                             float *d_dist_mp, uint8_t *d_pq_mp, float *d_node_cluster_dist, size_t pos) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t id = idx + pos;
    if (idx < n) {
        float sum_sq_diff = 0.0f;
        float res = 0;
        for (int i = 0; i < sub_vector; i++) {
            res += d_dist_mp[i * sub_cluster_count + d_pq_mp[id * sub_vector + i]];
        }
        if (res * d_W[0] + d_node_cluster_dist[id] * d_W[1] + d_B[0] >= threshold) {
            sum_sq_diff = INF;
        } else {
            for (size_t i = 0; i < m; ++i) {
                float diff = d_matrix[idx * m + i] - d_query[i];
                sum_sq_diff += diff * diff;
            }
        }
        d_result[idx] = sum_sq_diff;
    }
}

__global__ void DDCopq4ExtKernel(const float * __restrict__ d_matrix, const float * __restrict__ d_query, float * __restrict__ d_result, size_t n, size_t m,
                                 float threshold, float *d_W, float *d_B, size_t sub_vector, size_t sub_cluster_count,
                                 float *d_dist_mp, uint8_t *d_pq_mp, float *d_node_cluster_dist, size_t pos) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t id = idx + pos;
    if (idx < n) {
        float sum_sq_diff = 0.0f;
        float res = 0;
        for (int i = 0; i < sub_vector; i++) {
            res += d_dist_mp[i * sub_cluster_count + d_pq_mp[id * sub_vector + i]];
        }
        if (res * d_W[0] + d_node_cluster_dist[id] * d_W[1] + d_B[0] >= threshold) {
            sum_sq_diff = INF;
        } else {
            const float4 *matrix_vec = reinterpret_cast<const float4 *>(&d_matrix[idx * m]);
            const float4 *query_vec = reinterpret_cast<const float4 *>(d_query);

            for (size_t i = 0; i < m / 4; ++i) {
                float4 m_val = matrix_vec[i];
                float4 q_val = query_vec[i];

                float diff1 = m_val.x - q_val.x;
                float diff2 = m_val.y - q_val.y;
                float diff3 = m_val.z - q_val.z;
                float diff4 = m_val.w - q_val.w;

                sum_sq_diff += diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4;
            }
        }
        d_result[idx] = sum_sq_diff;
    }
}


/////////////////// class ///////////////////

class IVF_CUDA {
public:
    //// dataset parameters
    size_t N_;
    size_t D_;
    size_t C_;
    size_t d_;

    //// data
    float *L1_data_;
    float *res_data_;
    float *centroids_;

    size_t *start_;
    size_t *len_;
    size_t *id_;

    //// GPU
    float *h_result_;
    float *d_res_data_;
    float *d_centroids_;
    float *d_query_;
    float * __restrict__ d_result_;


    //// dco
    typedef void (IVF_CUDA::*DcoCUDAPtr)(float *, float *, float *, float *, size_t, size_t, float);

    DcoCUDAPtr cudaPtr_;
    DcoCUDAPtr centroidPtr_;
    DcoCUDAPtr codebookPtr_;
    size_t op_{0};
    float *transform_matrix_;
    float *d_transform_matrix_;
    // DADE
    float significance_{0.1};
    float *cdf_lmd_;
    float *d_cdf_lmd_;
    float *epsilon_;
    float *d_epsilon_;
    // DDCres
    bool learned_{false};
    std::vector<float> mean_;
    std::vector<float> var_;
    std::vector<float> extra_mean_;
    float *pre_query_;
    float *d_pre_query_;
    float *base_square_;
    float *d_base_square_;
    float query_square_{0};
    size_t pos_{0};
    // DDCpca
    size_t model_count_ = 0;
    float *W_, *B_, *b_;
    float *d_W_, *d_B_, *d_b_;
    // DDCopq
    uint8_t *pq_mp_;
    float *node_cluster_dist_;
    size_t sub_dim_{0}, sub_vector_{0}, sub_cluster_count_{0};
    float *pq_book_;
    float *dist_mp_;
    size_t project_dim_;

    uint8_t *d_pq_mp_;
    float *d_node_cluster_dist_;
    float *d_pq_book_;
    float *d_dist_mp_;


    //// CUDA
    void TransformCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t m);

    void FDScanningCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                        float threshold);

    void FDScanning4ExtCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                            float threshold);

    void PDScanningCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                        float threshold);

    void PDScanning4ExtCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                            float threshold);

    void ADSamplingCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                        float threshold);

    void ADSampling4ExtCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                            float threshold);

    void DADECUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                  float threshold);

    void DADE4ExtCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                      float threshold);

    void DDCresCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                    float threshold);

    void DDCres4ExtCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                        float threshold);

    void DDCpcaCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                    float threshold);

    void DDCpca4ExtCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                        float threshold);

    void DDCopqCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                    float threshold);

    void DDCopq4ExtCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                        float threshold);

    void DistMapCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                     float threshold);

    void DistMap4ExtCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                         float threshold);

    //// function
    IVF_CUDA();

    std::priority_queue <std::pair<float, size_t>>
    searchKNN(float *query, size_t k, size_t nprobe, float distK = std::numeric_limits<float>::max());

    std::priority_queue <std::pair<float, size_t>>
    searchKNNIP(float *query, size_t k, size_t nprobe, float distK = std::numeric_limits<float>::max());

    std::priority_queue <std::pair<float, size_t>>
    searchKNNOPQ(float *query, size_t k, size_t nprobe, float distK = std::numeric_limits<float>::max());

    void setQuery(float *query);

    void setPos(size_t pos);

    void get_query_square(const float * q);

    void compute_base_square(bool learned);

    void encoder_origin_data();

    void load(std::string filename);

    void setup_dco(std::string dataset_path, size_t op, std::string type);

    void query_transform(const void *query, size_t n = 1);

    void load_ADSampling(std::string &dataset_path);

    void load_RPDScanning(std::string &dataset_path);

    void load_DADE(std::string &dataset_path);

    void load_DDCres(std::string &dataset_path, std::string type);

    void load_DDCpca(std::string &dataset_path, std::string type);

    void load_DDCopq(std::string &dataset_path, std::string type);

    ~IVF_CUDA();
};

/////////////////// CUDA ///////////////////

void IVF_CUDA::TransformCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t m) {
    size_t threadsPerBlock = 256;
    size_t numBlocks = (m + threadsPerBlock - 1) / threadsPerBlock;
    MatrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_query, d_result, m);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_result, d_result, (size_t) m * sizeof(float), cudaMemcpyDeviceToHost));
}

void IVF_CUDA::FDScanningCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                              float threshold) {
    // Launch kernel
    size_t threadsPerBlock = 256;
    size_t numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    FDScanningKernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_query, d_result, n, m);
    // Synchronous device
    CUDA_CHECK(cudaDeviceSynchronize());
    // Copy result
    CUDA_CHECK(cudaMemcpy(h_result, d_result, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost));
}

void IVF_CUDA::FDScanning4ExtCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                                  float threshold) {
    // Launch kernel
    size_t threadsPerBlock = 256;
    size_t numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    FDScanning4ExtKernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_query, d_result, n, m);
    // Synchronous device
    CUDA_CHECK(cudaDeviceSynchronize());
    // Copy result
    CUDA_CHECK(cudaMemcpy(h_result, d_result, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost));
}

void IVF_CUDA::PDScanningCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                              float threshold) {
    // Launch kernel
    size_t threadsPerBlock = 256;
    size_t numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    PDScanningKernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_query, d_result, n, m, threshold);
    // Synchronous device
    CUDA_CHECK(cudaDeviceSynchronize());
    // Copy result
    CUDA_CHECK(cudaMemcpy(h_result, d_result, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost));
}

void IVF_CUDA::PDScanning4ExtCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                                  float threshold) {
    // Launch kernel
    size_t threadsPerBlock = 256;
    size_t numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    PDScanning4ExtKernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_query, d_result, n, m, threshold);
    // Synchronous device
    CUDA_CHECK(cudaDeviceSynchronize());
    // Copy result
    CUDA_CHECK(cudaMemcpy(h_result, d_result, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost));
}

void IVF_CUDA::ADSamplingCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                              float threshold) {
    // Launch kernel
    size_t threadsPerBlock = 256;
    size_t numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    ADSamplingKernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_query, d_result, n, m, threshold);
    // Synchronous device
    CUDA_CHECK(cudaDeviceSynchronize());
    // Copy result
    CUDA_CHECK(cudaMemcpy(h_result, d_result, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost));
}

void IVF_CUDA::ADSampling4ExtCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                                  float threshold) {
    // Launch kernel
    size_t threadsPerBlock = 256;
    size_t numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    ADSampling4ExtKernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_query, d_result, n, m, threshold);
    // Synchronous device
    CUDA_CHECK(cudaDeviceSynchronize());
    // Copy result
    CUDA_CHECK(cudaMemcpy(h_result, d_result, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost));
}

void IVF_CUDA::DADECUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                        float threshold) {
    // Launch kernel
    size_t threadsPerBlock = 256;
    size_t numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    DADEKernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_query, d_result, n, m, threshold, d_cdf_lmd_, d_epsilon_);
    // Synchronous device
    CUDA_CHECK(cudaDeviceSynchronize());
    // Copy result
    CUDA_CHECK(cudaMemcpy(h_result, d_result, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost));
}

void IVF_CUDA::DADE4ExtCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                            float threshold) {
    // Launch kernel
    size_t threadsPerBlock = 256;
    size_t numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    DADE4ExtKernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_query, d_result, n, m, threshold, d_cdf_lmd_,
                                                   d_epsilon_);
    // Synchronous device
    CUDA_CHECK(cudaDeviceSynchronize());
    // Copy result
    CUDA_CHECK(cudaMemcpy(h_result, d_result, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost));
}

void IVF_CUDA::DDCresCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                          float threshold) {
    // Launch kernel
    size_t threadsPerBlock = 256;
    size_t numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    DDCresKernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_query, d_result, n, m, threshold, query_square_ + 1e-5,
                                                 d_base_square_, d_pre_query_, pos_);
    // Synchronous device
    CUDA_CHECK(cudaDeviceSynchronize());
    // Copy result
    CUDA_CHECK(cudaMemcpy(h_result, d_result, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost));
}

void IVF_CUDA::DDCres4ExtCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                              float threshold) {
    // Launch kernel
    size_t threadsPerBlock = 256;
    size_t numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    DDCres4ExtKernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_query, d_result, n, m, threshold, query_square_ + 1e-5,
                                                     d_base_square_, d_pre_query_, pos_);
    // Synchronous device
    CUDA_CHECK(cudaDeviceSynchronize());
    // Copy result
    CUDA_CHECK(cudaMemcpy(h_result, d_result, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost));
}

void IVF_CUDA::DDCpcaCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                          float threshold) {
    // Launch kernel
    size_t threadsPerBlock = 256;
    size_t numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    DDCpcaKernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_query, d_result, n, m, threshold, d_W_, d_B_);
    // Synchronous device
    CUDA_CHECK(cudaDeviceSynchronize());
    // Copy result
    CUDA_CHECK(cudaMemcpy(h_result, d_result, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost));
}

void IVF_CUDA::DDCpca4ExtCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                              float threshold) {
    // Launch kernel
    size_t threadsPerBlock = 256;
    size_t numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    DDCpca4ExtKernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_query, d_result, n, m, threshold, d_W_, d_B_);
    // Synchronous device
    CUDA_CHECK(cudaDeviceSynchronize());
    // Copy result
    CUDA_CHECK(cudaMemcpy(h_result, d_result, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost));
}

void IVF_CUDA::DDCopqCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                          float threshold) {
    // Launch kernel
    size_t threadsPerBlock = 256;
    size_t numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    DDCopqKernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_query, d_result, n, m, threshold, d_W_, d_B_, sub_vector_,
                                                 sub_cluster_count_, d_dist_mp_, d_pq_mp_, d_node_cluster_dist_, pos_);
    // Synchronous device
    CUDA_CHECK(cudaDeviceSynchronize());
    // Copy result
    CUDA_CHECK(cudaMemcpy(h_result, d_result, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost));
}

void IVF_CUDA::DDCopq4ExtCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                              float threshold) {
    // Launch kernel
    size_t threadsPerBlock = 256;
    size_t numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    DDCopq4ExtKernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_query, d_result, n, m, threshold, d_W_, d_B_,
                                                     sub_vector_,
                                                     sub_cluster_count_, d_dist_mp_, d_pq_mp_, d_node_cluster_dist_,
                                                     pos_);
    // Synchronous device
    CUDA_CHECK(cudaDeviceSynchronize());
    // Copy result
    CUDA_CHECK(cudaMemcpy(h_result, d_result, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost));
}

void IVF_CUDA::DistMapCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                           float threshold) {
    // Launch kernel
    size_t threadsPerBlock = 256;
    size_t numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    DistMapKernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_query, d_result, sub_vector_, sub_cluster_count_,
                                                  sub_dim_);
    // Synchronous device
    CUDA_CHECK(cudaDeviceSynchronize());
}

void IVF_CUDA::DistMap4ExtCUDA(float *d_matrix, float *d_query, float * __restrict__ d_result, float *h_result, size_t n, size_t m,
                               float threshold) {
    // Launch kernel
    size_t threadsPerBlock = 256;
    size_t numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    DistMap4ExtKernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_query, d_result, sub_vector_, sub_cluster_count_,
                                                      sub_dim_);
    // Synchronous device
    CUDA_CHECK(cudaDeviceSynchronize());
}

/////////////////// IVF ///////////////////

IVF_CUDA::IVF_CUDA() {
    N_ = D_ = C_ = d_ = 0;
    start_ = len_ = id_ = NULL;
    L1_data_ = res_data_ = centroids_ = NULL;
}

std::priority_queue <std::pair<float, size_t>>
IVF_CUDA::searchKNN(float *query, size_t k, size_t nprobe, float distK) {
    std::priority_queue <std::pair<float, size_t>> KNNs;
    //// Compute centroid distance
    float *centroid_dist = (float *) malloc((size_t) C_ * sizeof(float));
    (this->*centroidPtr_)(d_centroids_, d_query_, d_result_, centroid_dist, C_, D_, INF);
    std::vector <size_t> indices(C_);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + nprobe, indices.end(),
                      [&](size_t a, size_t b) {
                          return centroid_dist[a] < centroid_dist[b];
                      });
    free(centroid_dist);
    //// Compute cluster distance
    for (size_t i = 0; i < nprobe; ++i) {
        size_t cluster_id = indices[i];
        float *start_pos = d_res_data_ + start_[cluster_id] * D_;
        size_t cluster_size = len_[cluster_id];
        (this->*cudaPtr_)(start_pos, d_query_, d_result_, h_result_, cluster_size, D_, distK);
        for (size_t j = 0; j < cluster_size; ++j) {
            if (KNNs.size() < k) {
                KNNs.emplace(h_result_[j], id_[start_[cluster_id] + j]);
                distK = KNNs.top().first;
            } else {
                if (h_result_[j] < distK) {
                    KNNs.emplace(h_result_[j], id_[start_[cluster_id] + j]);
                    KNNs.pop();
                    distK = KNNs.top().first;
                }
            }
        }
    }
    return KNNs;
}

std::priority_queue <std::pair<float, size_t>>
IVF_CUDA::searchKNNIP(float *query, size_t k, size_t nprobe, float distK) {
    get_query_square(query);
    std::priority_queue <std::pair<float, size_t>> KNNs;
    //// Compute centroid distance
    float *centroid_dist = (float *) malloc((size_t) C_ * sizeof(float));
    (this->*centroidPtr_)(d_centroids_, d_query_, d_result_, centroid_dist, C_, D_, INF);
    std::vector <size_t> indices(C_);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + nprobe, indices.end(),
                      [&](size_t a, size_t b) {
                          return centroid_dist[a] < centroid_dist[b];
                      });
    free(centroid_dist);
    //// Compute cluster distance
    for (size_t i = 0; i < nprobe; ++i) {
        size_t cluster_id = indices[i];
        float *start_pos = d_res_data_ + start_[cluster_id] * D_;
        size_t cluster_size = len_[cluster_id];
        setPos(start_[cluster_id]);
        (this->*cudaPtr_)(start_pos, d_query_, d_result_, h_result_, cluster_size, D_, distK);
        for (size_t j = 0; j < cluster_size; ++j) {
            if (KNNs.size() < k) {
                KNNs.emplace(h_result_[j], id_[start_[cluster_id] + j]);
                distK = KNNs.top().first;
            } else {
                if (h_result_[j] < distK) {
                    KNNs.emplace(h_result_[j], id_[start_[cluster_id] + j]);
                    KNNs.pop();
                    distK = KNNs.top().first;
                }
            }
        }
    }
    return KNNs;
}

std::priority_queue <std::pair<float, size_t>>
IVF_CUDA::searchKNNOPQ(float *query, size_t k, size_t nprobe, float distK) {
    std::priority_queue <std::pair<float, size_t>> KNNs;
    //// Compute centroid distance
    float *centroid_dist = (float *) malloc((size_t) C_ * sizeof(float));
    (this->*centroidPtr_)(d_centroids_, d_query_, d_result_, centroid_dist, C_, D_, INF);
    std::vector <size_t> indices(C_);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + nprobe, indices.end(),
                      [&](size_t a, size_t b) {
                          return centroid_dist[a] < centroid_dist[b];
                      });
    free(centroid_dist);
    //// calc_dist_map
//    for (size_t i = 0; i < sub_vector_; ++i) {
//        (this->*codebookPtr_)(d_pq_book_ + i * sub_cluster_count_ * sub_dim_, d_query_ + i * sub_dim_,
//                              d_dist_mp_ + i * sub_cluster_count_, NULL, sub_cluster_count_, sub_dim_, INF);
//    }
    (this->*codebookPtr_)(d_pq_book_, d_query_, d_dist_mp_, NULL, sub_vector_, sub_cluster_count_ * sub_dim_, INF);
    //// Compute cluster distance
    for (size_t i = 0; i < nprobe; ++i) {
        size_t cluster_id = indices[i];
        float *start_pos = d_res_data_ + start_[cluster_id] * D_;
        size_t cluster_size = len_[cluster_id];
        setPos(start_[cluster_id]);
        (this->*cudaPtr_)(start_pos, d_query_, d_result_, h_result_, cluster_size, D_, distK);
        for (size_t j = 0; j < cluster_size; ++j) {
            if (KNNs.size() < k) {
                KNNs.emplace(h_result_[j], id_[start_[cluster_id] + j]);
                distK = KNNs.top().first;
            } else {
                if (h_result_[j] < distK) {
                    KNNs.emplace(h_result_[j], id_[start_[cluster_id] + j]);
                    KNNs.pop();
                    distK = KNNs.top().first;
                }
            }
        }
    }
    return KNNs;
}

void IVF_CUDA::setPos(size_t pos) {
    this->pos_ = pos;
}

void IVF_CUDA::get_query_square(const float * q) {
    pre_query_ = new float[D_ + 1];
    query_square_ = 0;
    for (int i = 0; i < D_; i++) {
        query_square_ += q[i] * q[i];
        pre_query_[i] = q[i] * q[i] * var_[i];
    }
    pre_query_[D_] = 0;
    for (int i = (int) D_ - 1; i >= 0; i--) {
        pre_query_[i] += pre_query_[i + 1];
    }
    for (int i = 0; i < D_; i++) {
        pre_query_[i] = sqrt(pre_query_[i]);
        pre_query_[i] *= sigma * 2;
    }

    CUDA_CHECK(cudaMalloc((void **) &d_pre_query_, (size_t)(D_ + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_pre_query_, pre_query_, (size_t)(D_ + 1) * sizeof(float), cudaMemcpyHostToDevice));
}

void IVF_CUDA::compute_base_square(bool learned) {
    base_square_ = new float[N_];
    float *extra = extra_mean_.data();
    float *tmp_L1 = L1_data_, *tmp_res = res_data_;
    for (int i = 0; i < N_; i++) {
        float square = 0.0;
        for (int j = 0; j < d_; j++) {
            if (!learned) tmp_L1[j] -= extra[j];
            square += tmp_L1[j] * tmp_L1[j];
        }
        for (int j = 0; j < D_ - d_; j++) {
            if (!learned) tmp_res[j] -= extra[j + d_];
            square += tmp_res[j] * tmp_res[j];
        }
        base_square_[i] = square;
        tmp_L1 += d_;
        tmp_res += (D_ - d_);
    }

    CUDA_CHECK(cudaMalloc((void **) &d_base_square_, (size_t) N_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_base_square_, base_square_, (size_t) N_ * sizeof(float), cudaMemcpyHostToDevice));

    if (!learned) {
        float *tmp_centroid = centroids_;
        for (int i = 0; i < C_; i++) {
            for (int j = 0; j < D_; j++) {
                tmp_centroid[j] -= extra[j];
            }
            tmp_centroid += D_;
        }
        CUDA_CHECK(cudaMemcpy(d_centroids_, centroids_, (size_t) C_ * D_ * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void IVF_CUDA::encoder_origin_data() {
    if (d_ != 0) {
        std::cerr << "Error: Invalid DCO." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    pq_mp_ = new unsigned char[N_ * sub_vector_];
    node_cluster_dist_ = new float[N_];
    double ave_dist = 0.0;
    for (int i = 0; i < N_; i++) {
        float dist_to_centroid = 0.0;
        for (int j = 0; j < sub_vector_; j++) {
            uint8_t belong = 0;
            float dist = L2Distance(res_data_ + i * D_ + j * sub_dim_, &pq_book_[j * sub_cluster_count_ * sub_dim_],
                                    sub_dim_);
            for (int k = 1; k < sub_cluster_count_; k++) {
                float new_dist = L2Distance(res_data_ + i * D_ + j * sub_dim_,
                                            &pq_book_[j * sub_cluster_count_ * sub_dim_ + k * sub_dim_],
                                            sub_dim_);
                if (new_dist < dist) {
                    belong = k;
                    dist = new_dist;
                }
            }
            dist_to_centroid += dist;
            pq_mp_[i * sub_vector_ + j] = belong;
        }
        node_cluster_dist_[i] = dist_to_centroid;
        ave_dist += dist_to_centroid;
        if (i % 50000 == 0) std::cerr << "Encoder progress: " << i << " / " << N_ << std::endl;
    }
    std::cerr << "Encoder ave dist:: " << ave_dist / N_ << std::endl;
    CUDA_CHECK(cudaMalloc((void **) &d_pq_mp_, (size_t) N_ * sub_vector_ * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc((void **) &d_node_cluster_dist_, (size_t) N_ * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_pq_mp_, pq_mp_, (size_t) N_ * sub_vector_ * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(
            cudaMemcpy(d_node_cluster_dist_, node_cluster_dist_, (size_t) N_ * sizeof(float), cudaMemcpyHostToDevice));
}

void IVF_CUDA::setQuery(float *query) {
    //// Copy query
    CUDA_CHECK(cudaMemcpy(d_query_, query, (size_t) D_ * sizeof(float), cudaMemcpyHostToDevice));
}

void IVF_CUDA::load(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::cerr << filename << std::endl;

    if (!input.is_open())
        throw std::runtime_error("Cannot open file");

    input.read((char *) &N_, sizeof(size_t));
    input.read((char *) &D_, sizeof(size_t));
    input.read((char *) &C_, sizeof(size_t));
    input.read((char *) &d_, sizeof(size_t));
    std::cerr << N_ << " " << D_ << " " << C_ << " " << d_ << std::endl;

    L1_data_ = new float[N_ * d_ + 10];
    res_data_ = new float[N_ * (D_ - d_) + 10];
    centroids_ = new float[C_ * D_];

    start_ = new size_t[C_];
    len_ = new size_t[C_];
    id_ = new size_t[N_];

    if (d_ > 0)input.read((char *) L1_data_, N_ * d_ * sizeof(float));
    if (d_ < D_)input.read((char *) res_data_, N_ * (D_ - d_) * sizeof(float));
    input.read((char *) centroids_, C_ * D_ * sizeof(float));

    input.read((char *) start_, C_ * sizeof(size_t));
    input.read((char *) len_, C_ * sizeof(size_t));
    input.read((char *) id_, N_ * sizeof(size_t));

    input.close();

    size_t max_cluster_size = std::max(C_, D_);
    for (size_t i = 0; i < C_; ++i) {
        max_cluster_size = std::max(max_cluster_size, len_[i]);
    }

    h_result_ = (float *) malloc((size_t) max_cluster_size * sizeof(float));

    //// Allocate memory
    CUDA_CHECK(cudaMalloc((void **) &d_res_data_, (size_t) N_ * D_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &d_centroids_, (size_t) C_ * D_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &d_query_, (size_t) D_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &d_result_, (size_t) max_cluster_size * sizeof(float)));

    //// Copy data to device
    CUDA_CHECK(cudaMemcpy(d_res_data_, res_data_, (size_t) N_ * D_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids_, centroids_, (size_t) C_ * D_ * sizeof(float), cudaMemcpyHostToDevice));
}

void IVF_CUDA::setup_dco(std::string dataset_path, size_t op, std::string type) {
    op_ = op;
    if (D_ % 4 == 0) {
        centroidPtr_ = &IVF_CUDA::FDScanning4ExtCUDA;
        codebookPtr_ = &IVF_CUDA::DistMap4ExtCUDA;
    } else {
        centroidPtr_ = &IVF_CUDA::DistMapCUDA;
        codebookPtr_ = &IVF_CUDA::DistMapCUDA;
    }
    switch (op_) {
        case 0:
            std::cout << "Using FDScanning" << std::endl;
            if (D_ % 4 == 0)
                cudaPtr_ = &IVF_CUDA::FDScanning4ExtCUDA;
            else
                cudaPtr_ = &IVF_CUDA::FDScanningCUDA;
            break;
        case 1:
            std::cout << "Using PDScanning" << std::endl;
            if (D_ % 4 == 0)
                cudaPtr_ = &IVF_CUDA::PDScanning4ExtCUDA;
            else
                cudaPtr_ = &IVF_CUDA::PDScanningCUDA;
            break;
        case 2:
            std::cout << "Using RPDScanning" << std::endl;
            if (D_ % 4 == 0)
                cudaPtr_ = &IVF_CUDA::PDScanning4ExtCUDA;
            else
                cudaPtr_ = &IVF_CUDA::PDScanningCUDA;
            load_RPDScanning(dataset_path);
            break;
        case 3:
            std::cout << "Using ADSampling" << std::endl;
            if (D_ % 4 == 0)
                cudaPtr_ = &IVF_CUDA::ADSampling4ExtCUDA;
            else
                cudaPtr_ = &IVF_CUDA::ADSamplingCUDA;
            load_ADSampling(dataset_path);
            break;
        case 4:
            std::cout << "Using DADE" << std::endl;
            if (D_ % 4 == 0)
                cudaPtr_ = &IVF_CUDA::DADE4ExtCUDA;
            else
                cudaPtr_ = &IVF_CUDA::DADECUDA;
            load_DADE(dataset_path);
            break;
        case 5:
            std::cout << "Using DDCres" << std::endl;
            if (D_ % 4 == 0)
                cudaPtr_ = &IVF_CUDA::DDCres4ExtCUDA;
            else
                cudaPtr_ = &IVF_CUDA::DDCresCUDA;
            load_DDCres(dataset_path, type);
            break;
        case 6:
            std::cout << "Using DDCpca" << std::endl;
            if (D_ % 4 == 0)
                cudaPtr_ = &IVF_CUDA::DDCpca4ExtCUDA;
            else
                cudaPtr_ = &IVF_CUDA::DDCpcaCUDA;
            load_DDCpca(dataset_path, type);
            break;
        case 7:
            std::cout << "Using DDCopq" << std::endl;
            if (D_ % 4 == 0)
                cudaPtr_ = &IVF_CUDA::DDCopq4ExtCUDA;
            else
                cudaPtr_ = &IVF_CUDA::DDCopqCUDA;
            load_DDCopq(dataset_path, type);
        default:
            return;
    }
}

void IVF_CUDA::query_transform(const void *query, size_t n) {
    switch (op_) {
        case 2:
        case 3:
        case 7:
        case 4: {
            float *q = (float *) query;
            for (size_t i = 0; i < n; i++) {
                setQuery(q);
                TransformCUDA(d_transform_matrix_, d_query_, d_result_, q, D_);
            }
            break;
        }
        case 5: {
            float *q = (float *) query;
            float *q2 = new float[D_];
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < D_; j++)
                    q2[j] = q[i * D_ + j] - mean_[j];
                setQuery(q2);
                TransformCUDA(d_transform_matrix_, d_query_, d_result_, q2, D_);
                for (size_t j = 0; j < D_; j++) {
                    if (!learned_)
                        q[i * D_ + j] = q2[j] - extra_mean_[j];
                    else
                        q[i * D_ + j] = q2[j];
                }
            }
            delete[] q2;
            break;
        }
        case 6: {
            float *q = (float *) query;
            float *q2 = new float[D_];
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < D_; j++)
                    q2[j] = q[i * D_ + j] - mean_[j];
                setQuery(q2);
                TransformCUDA(d_transform_matrix_, d_query_, d_result_, q, D_);
            }
            delete[] q2;
            break;
        }
        default:
            return;
    }
}

void IVF_CUDA::load_ADSampling(std::string &dataset_path) {
    size_t last_slash_pos = dataset_path.find_last_of('/');
    std::string directory;
    if (last_slash_pos != std::string::npos) {
        directory = dataset_path.substr(0, last_slash_pos);
    } else {
        std::cerr << "Error: Invalid file path." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::string transformation_path = directory + "/adsampling/O.fvecs";
    std::vector <VectorDataType> Orthogonal;
    ReadVectorData(transformation_path, Orthogonal);
    transform_matrix_ = new float[D_ * D_];
    for (int i = 0; i < D_; i++)
        for (int j = 0; j < D_; j++)
            transform_matrix_[i * D_ + j] = Orthogonal[i].data[j];

    CUDA_CHECK(cudaMalloc((void **) &d_transform_matrix_, (size_t) D_ * D_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_transform_matrix_, transform_matrix_, (size_t) D_ * D_ * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void IVF_CUDA::load_RPDScanning(std::string &dataset_path) {
    size_t last_slash_pos = dataset_path.find_last_of('/');
    std::string directory;
    if (last_slash_pos != std::string::npos) {
        directory = dataset_path.substr(0, last_slash_pos);
    } else {
        std::cerr << "Error: Invalid file path." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::string transformation_path = directory + "/dade/P.fvecs";
    std::vector <VectorDataType> Orthogonal;
    ReadVectorData(transformation_path, Orthogonal);
    transform_matrix_ = new float[D_ * D_];
    for (int i = 0; i < D_; i++)
        for (int j = 0; j < D_; j++)
            transform_matrix_[i * D_ + j] = Orthogonal[i].data[j];

    CUDA_CHECK(cudaMalloc((void **) &d_transform_matrix_, (size_t) D_ * D_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_transform_matrix_, transform_matrix_, (size_t) D_ * D_ * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void IVF_CUDA::load_DADE(std::string &dataset_path) {
    size_t last_slash_pos = dataset_path.find_last_of('/');
    std::string directory;
    if (last_slash_pos != std::string::npos) {
        directory = dataset_path.substr(0, last_slash_pos);
    } else {
        std::cerr << "Error: Invalid file path." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::string transformation_path = directory + "/dade/P.fvecs";
    std::vector <VectorDataType> Orthogonal;
    ReadVectorData(transformation_path, Orthogonal);
    transform_matrix_ = new float[D_ * D_];
    for (int i = 0; i < D_; i++)
        for (int j = 0; j < D_; j++)
            transform_matrix_[i * D_ + j] = Orthogonal[i].data[j];

    CUDA_CHECK(cudaMalloc((void **) &d_transform_matrix_, (size_t) D_ * D_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_transform_matrix_, transform_matrix_, (size_t) D_ * D_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    std::string lmd_path = directory + "/dade/LMD.fvecs";
    std::vector <VectorDataType> LMD;
    ReadVectorData(lmd_path, LMD);
    std::vector<float> lmds_;
    cdf_lmd_ = new float[D_ + 1];
    for (int i = 0; i < D_; ++i)
        lmds_.push_back(LMD[0].data[i]);
    float sum = 0;
    cdf_lmd_[0] = sum;
    for (int i = 0; i < D_; ++i) {
        sum += lmds_[i];
        cdf_lmd_[i + 1] = sum;
    }

    CUDA_CHECK(cudaMalloc((void **) &d_cdf_lmd_, (size_t)(D_ + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_cdf_lmd_, cdf_lmd_, (size_t)(D_ + 1) * sizeof(float),
                          cudaMemcpyHostToDevice));

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << significance_;
    std::string epsilon_path = directory + "/dade/E/" + oss.str() + ".fvecs";
    std::vector <VectorDataType> Epsilon;
    ReadVectorData(epsilon_path, Epsilon);
    epsilon_ = new float[D_ + 1];
    epsilon_[0] = 1e10;
    for (int i = 0; i < D_; ++i)
        epsilon_[i + 1] = Epsilon[0].data[i];

    CUDA_CHECK(cudaMalloc((void **) &d_epsilon_, (size_t)(D_ + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_epsilon_, epsilon_, (size_t)(D_ + 1) * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void IVF_CUDA::load_DDCres(std::string &dataset_path, std::string type) {
    size_t last_slash_pos = dataset_path.find_last_of('/');
    std::string directory;
    if (last_slash_pos != std::string::npos) {
        directory = dataset_path.substr(0, last_slash_pos);
    } else {
        std::cerr << "Error: Invalid file path." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::string transformation_path = directory + "/ddc/pca_matrix_" + type + ".fvecs";
    if (learned_) {
        transformation_path = directory + "/ddc/pca_matrix.fvecs";
    }
    std::vector <VectorDataType> Orthogonal;
    ReadVectorData(transformation_path, Orthogonal);
    mean_ = Orthogonal[0].data;
    extra_mean_ = Orthogonal[1].data;
    var_ = Orthogonal[2].data;
    transform_matrix_ = new float[D_ * D_];
    for (int i = 0; i < D_; i++)
        for (int j = 0; j < D_; j++)
            transform_matrix_[i * D_ + j] = Orthogonal[i + 3].data[j];

    CUDA_CHECK(cudaMalloc((void **) &d_transform_matrix_, (size_t) D_ * D_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_transform_matrix_, transform_matrix_, (size_t) D_ * D_ * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void IVF_CUDA::load_DDCpca(std::string &dataset_path, std::string type) {
    size_t last_slash_pos = dataset_path.find_last_of('/');
    std::string directory;
    if (last_slash_pos != std::string::npos) {
        directory = dataset_path.substr(0, last_slash_pos);
    } else {
        std::cerr << "Error: Invalid file path." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::string transformation_path = directory + "/ddc/pca_matrix.fvecs";
    std::vector <VectorDataType> Orthogonal;
    ReadVectorData(transformation_path, Orthogonal);
    mean_ = Orthogonal[0].data;
    extra_mean_ = Orthogonal[1].data;
    var_ = Orthogonal[2].data;
    transform_matrix_ = new float[D_ * D_];
    for (int i = 0; i < D_; i++)
        for (int j = 0; j < D_; j++)
            transform_matrix_[i * D_ + j] = Orthogonal[i + 3].data[j];

    CUDA_CHECK(cudaMalloc((void **) &d_transform_matrix_, (size_t) D_ * D_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_transform_matrix_, transform_matrix_, (size_t) D_ * D_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    std::string model_name = "linear_" + type + "_pca" + ".txt";
    std::string linear_path = directory + "/ddc/" + model_name;
    std::ifstream fin(linear_path);
    if (!fin.good()) {
        std::cerr << "Error: linear model not found." << std::endl;
        return;
    }
    size_t num;
    fin >> num;
    W_ = new float[num];
    B_ = new float[num];
    b_ = new float[num];
    model_count_ = num;
    for (int i = 0; i < num; i++) {
        fin >> W_[i] >> B_[i] >> b_[i];
    }
    if (D_ % delta_d != 0) {
        W_[num - 1] = 1.0;
        B_[num - 1] = 0.0;
        b_[num - 1] = 0.0;
    }
    std::cerr << " fix dim:: " << D_ << " res dim:: " << D_ % delta_d << " linear count:: "
              << num
              << std::endl;
    fin.close();

    CUDA_CHECK(cudaMalloc((void **) &d_W_, (size_t) num * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &d_B_, (size_t) num * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &d_b_, (size_t) num * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_W_, W_, (size_t) num * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_, B_, (size_t) num * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_, b_, (size_t) num * sizeof(float), cudaMemcpyHostToDevice));
}

void IVF_CUDA::load_DDCopq(std::string &dataset_path, std::string type) {
    size_t last_slash_pos = dataset_path.find_last_of('/');
    std::string directory;
    if (last_slash_pos != std::string::npos) {
        directory = dataset_path.substr(0, last_slash_pos);
    } else {
        std::cerr << "Error: Invalid file path." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::string codebook_path = directory + "/ddc/codebook.centroid";
    std::ifstream in(codebook_path, std::ios::binary);
    in.read((char *) &sub_vector_, sizeof(unsigned));
    in.read((char *) &sub_cluster_count_, sizeof(unsigned));
    in.read((char *) &sub_dim_, sizeof(unsigned));
    std::cerr << "sub vec:: " << sub_vector_ << " sub cluster:: " << sub_cluster_count_ << " sub dim:: "
              << sub_dim_ << std::endl;
    pq_book_ = new float[sub_vector_ * sub_cluster_count_ * sub_dim_];
    dist_mp_ = new float[sub_vector_ * sub_cluster_count_];
    for (int i = 0; i < sub_vector_; i++) {
        for (int j = 0; j < sub_cluster_count_; j++) {
            in.read((char *) &pq_book_[i * sub_cluster_count_ * sub_dim_ + j * sub_dim_], sizeof(float) * sub_dim_);
        }
    }
    CUDA_CHECK(cudaMalloc((void **) &d_pq_book_, (size_t) sub_vector_ * sub_cluster_count_ * sub_dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &d_dist_mp_, (size_t) sub_vector_ * sub_cluster_count_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_pq_book_, pq_book_, (size_t) sub_vector_ * sub_cluster_count_ * sub_dim_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    std::string transformation_path = directory + "/ddc/opq_matrix.fvecs";
    std::vector <VectorDataType> Orthogonal;
    ReadVectorData(transformation_path, Orthogonal);
    size_t origin_dim = Orthogonal.size(), project_dim = Orthogonal[0].Dimension();
    project_dim_ = project_dim;
    if (origin_dim != D_) {
        std::cerr << "Error: Invalid dimension." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    transform_matrix_ = new float[D_ * project_dim_];
    for (int i = 0; i < D_; i++)
        for (int j = 0; j < project_dim_; j++)
            transform_matrix_[i * project_dim_ + j] = Orthogonal[i].data[j];

    CUDA_CHECK(cudaMalloc((void **) &d_transform_matrix_, (size_t) D_ * project_dim_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_transform_matrix_, transform_matrix_, (size_t) D_ * project_dim_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    std::string model_name = "linear_" + type + "_opq" + ".txt";
    std::string linear_path = directory + "/ddc/" + model_name;
    std::ifstream fin(linear_path);
    if (!fin.good()) return;
    size_t num;
    fin >> num;
    model_count_ = num;
    W_ = new float[2];
    B_ = new float[1];
    fin >> W_[0] >> W_[1] >> B_[0];
    fin.close();

    CUDA_CHECK(cudaMalloc((void **) &d_W_, (size_t) 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &d_B_, (size_t) 1 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_W_, W_, (size_t) 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_, B_, (size_t) 1 * sizeof(float), cudaMemcpyHostToDevice));
}

IVF_CUDA::~IVF_CUDA() {
    if (id_ != NULL)delete[] id_;
    if (len_ != NULL)delete[] len_;
    if (start_ != NULL)delete[] start_;
    if (L1_data_ != NULL)delete[] L1_data_;
    if (res_data_ != NULL)delete[] res_data_;
    if (centroids_ != NULL)delete[] centroids_;

    //// Free memory
    if (h_result_ != NULL)free(h_result_);
    if (d_res_data_ != NULL)CUDA_CHECK(cudaFree(d_res_data_));
    if (d_centroids_ != NULL)CUDA_CHECK(cudaFree(d_centroids_));
    if (d_query_ != NULL)CUDA_CHECK(cudaFree(d_query_));
    if (d_result_ != NULL)CUDA_CHECK(cudaFree(d_result_));

    //// RPDScanning
    if (op_ == 2) {
        if (transform_matrix_ != NULL)delete[] transform_matrix_;
        if (d_transform_matrix_ != NULL)CUDA_CHECK(cudaFree(d_transform_matrix_));
    }

    //// ADSampling
    if (op_ == 3) {
        if (transform_matrix_ != NULL)delete[] transform_matrix_;
        if (d_transform_matrix_ != NULL)CUDA_CHECK(cudaFree(d_transform_matrix_));
    }

    //// DADE
    if (op_ == 4) {
        if (transform_matrix_ != NULL)delete[] transform_matrix_;
        if (d_transform_matrix_ != NULL)CUDA_CHECK(cudaFree(d_transform_matrix_));
        if (cdf_lmd_ != NULL)delete[] cdf_lmd_;
        if (d_cdf_lmd_ != NULL)CUDA_CHECK(cudaFree(d_cdf_lmd_));
        if (epsilon_ != NULL)delete[] epsilon_;
        if (d_epsilon_ != NULL)CUDA_CHECK(cudaFree(d_epsilon_));
    }

    //// DDCres
    if (op_ == 5) {
        if (transform_matrix_ != NULL)delete[] transform_matrix_;
        if (d_transform_matrix_ != NULL)CUDA_CHECK(cudaFree(d_transform_matrix_));
        if (base_square_ != NULL)delete[] base_square_;
        if (d_base_square_ != NULL)CUDA_CHECK(cudaFree(d_base_square_));
        if (pre_query_ != NULL)delete[] pre_query_;
        if (d_pre_query_ != NULL)CUDA_CHECK(cudaFree(d_pre_query_));
    }

    //// DDCpca
    if (op_ == 6) {
        if (transform_matrix_ != NULL)delete[] transform_matrix_;
        if (d_transform_matrix_ != NULL)CUDA_CHECK(cudaFree(d_transform_matrix_));
        if (W_ != NULL)delete[] W_;
        if (d_W_ != NULL)CUDA_CHECK(cudaFree(d_W_));
        if (B_ != NULL)delete[] B_;
        if (d_B_ != NULL)CUDA_CHECK(cudaFree(d_B_));
        if (b_ != NULL)delete[] b_;
        if (d_b_ != NULL)CUDA_CHECK(cudaFree(d_b_));
    }

    ///DDCopq
    if (op_ == 7) {
        if (transform_matrix_ != NULL)delete[] transform_matrix_;
        if (d_transform_matrix_ != NULL)CUDA_CHECK(cudaFree(d_transform_matrix_));
        if (W_ != NULL)delete[] W_;
        if (d_W_ != NULL)CUDA_CHECK(cudaFree(d_W_));
        if (B_ != NULL)delete[] B_;
        if (d_B_ != NULL)CUDA_CHECK(cudaFree(d_B_));
        if (pq_book_ != NULL) delete[] pq_book_;
        if (d_pq_book_ != NULL)CUDA_CHECK(cudaFree(d_pq_book_));
        if (dist_mp_ != NULL) delete[] dist_mp_;
        if (d_dist_mp_ != NULL)CUDA_CHECK(cudaFree(d_dist_mp_));
        if (pq_mp_ != NULL) delete[] pq_mp_;
        if (d_pq_mp_ != NULL)CUDA_CHECK(cudaFree(d_pq_mp_));
        if (node_cluster_dist_ != NULL) delete[] node_cluster_dist_;
        if (d_node_cluster_dist_ != NULL)CUDA_CHECK(cudaFree(d_node_cluster_dist_));
    }
}

#endif //IVF_CUDA_CUH