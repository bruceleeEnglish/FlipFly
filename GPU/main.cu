#include <iostream>
#include <cuda_runtime.h>
#include <queue>
#include <chrono>
#include "../FileIO/File_IO.h"

__global__ void L2DistanceMatrix(const float *d_matrix, const float *d_query, float *d_result, int n, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float sum_sq_diff = 0.0f;
        if (m % 4 == 0) {
            const float4* matrix_vec = reinterpret_cast<const float4*>(&d_matrix[idx * m]);
            const float4* query_vec = reinterpret_cast<const float4*>(d_query);

            for (int i = 0; i < m / 4; ++i) {
                float4 m_val = matrix_vec[i];
                float4 q_val = query_vec[i];

                float diff1 = m_val.x - q_val.x;
                float diff2 = m_val.y - q_val.y;
                float diff3 = m_val.z - q_val.z;
                float diff4 = m_val.w - q_val.w;

                sum_sq_diff += diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4;
            }
        } else {
            for (int i = 0; i < m; ++i) {
                float diff = d_matrix[idx * m + i] - d_query[i];
                sum_sq_diff += diff * diff;
            }
        }

        d_result[idx] = sum_sq_diff;
    }
}



int main() {
    int k = 128;

    std::vector <VectorDataType> data_list;
    ReadVectorData("/home/dataset/gist/gist_base.fvecs", data_list);

    std::vector <VectorDataType> query_data_list;
    ReadVectorData("/home/dataset/gist/query.fvecs", query_data_list);

    std::vector <std::vector<long>> ground_truth;
    ReadGroundTruth("/home/dataset/gist/gt_256.ivecs", ground_truth);

    int m = data_list.size();
    int n = data_list[0].data.size();
    // Configure the CUDA kernel launch parameters
    int threadsPerBlock = 256; // threads per block
    int blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock; // number of blocks needed

    // Use a priority queue to get the top-k results of each query
    auto startQuery = std::chrono::high_resolution_clock::now();
    std::vector <std::vector<long>> result_list;
    float *cu_matrix, *cu_vec, *cu_result;
    float *h_matrix, *h_distances;
    h_matrix = new float[m * n];
    h_distances = new float[m];
    // Initialize the vectors
    cudaError_t err = cudaMalloc((void **) &cu_matrix, sizeof(float) * m * n);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed: %s\n", cudaGetErrorString(err));
    }
    cudaMalloc((void **) &cu_vec, sizeof(float) * n);
    cudaMalloc((void **) &cu_result, sizeof(float) * m);
    // Copy data_list into h_matrix
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_matrix[i * n + j] = data_list[i].data[j];
        }
    }
    cudaMemcpy(cu_matrix, h_matrix, m * n * sizeof(float), cudaMemcpyHostToDevice);
    for (size_t i = 0; i < query_data_list.size(); ++i) {
        std::priority_queue <std::pair<float, long>> pq;
        // Copy the data from host to device
        cudaMemcpy(cu_vec, query_data_list[i].data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        // Launch the CUDA kernel
        L2DistanceMatrix<<<blocksPerGrid, threadsPerBlock>>>(cu_matrix, cu_vec, cu_result, m, n);
        // Copy the results from device back to host
        cudaMemcpy(h_distances, cu_result, m * sizeof(float), cudaMemcpyDeviceToHost);
        for (int j = 0; j < m; ++j) {
            float dis = h_distances[j];
            if (pq.size() < k) {
                pq.push(std::make_pair(dis, data_list[j].vid));
            } else {
                if (dis < pq.top().first) {
                    pq.pop();
                    pq.push(std::make_pair(dis, data_list[j].vid));
                }
            }
        }
        std::vector<long> result;
        while (!pq.empty()) {
            result.emplace_back(pq.top().second);
            pq.pop();
        }
        std::reverse(result.begin(), result.end());
        result_list.emplace_back(result);
    }
    // Free device memory
    cudaFree(cu_matrix);
    cudaFree(cu_vec);
    cudaFree(cu_result);
    delete[] h_matrix;
    delete[] h_distances;
    auto endQuery = std::chrono::high_resolution_clock::now();
    auto queryTime = std::chrono::duration_cast<std::chrono::milliseconds>(endQuery - startQuery).count();
    std::cout << "Query time: " << queryTime << "ms\n";
    EvaluateAnswer(result_list, ground_truth, 128);
    return 0;
}