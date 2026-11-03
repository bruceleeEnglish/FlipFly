#include <iostream>
#include <getopt.h>
#include <fstream>
#include <ctime>
#include <cmath>
#include <queue>
#include "../Logger/Logger.h"
#include "../FileIO/File_IO.h"
#include "../FileIO/DataType.hpp"

VectorDimensionType (* metric_func)(const VectorDimensionType* a, const VectorDimensionType* b, int d);

struct Compare {
    bool operator()(const std::pair<float, VidType>& a, const std::pair<float, VidType>& b) {
        return a.first < b.first;
    }
};

int main() {
    //// parse arguments
    int K = 256;
    std::string dataset_path = "/home/zzlin/dataset/OpenAIArXiv/openai_base.bin";
    std::string query_path = "/home/zzlin/dataset/OpenAIArXiv/openai_query.bin";
    std::string ground_truth_path = "/home/zzlin/dataset/OpenAIArXiv/gt_256.ivecs";
    int metric = 0;

    if (metric == 0) {
        metric_func = L2Distance;
    } else if (metric == 1) {
        metric_func = IPDistance;
    } else {
        std::cerr << "Error: Invalid metric. Please provide a valid metric using the -m option." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "K: " << K << std::endl;
    std::cout << "dataset_path: " << dataset_path << std::endl;
    std::cout << "query_path: " << query_path << std::endl;
    std::cout << "ground_truth_path: " << ground_truth_path << std::endl;
    std::cout << "metric: " << metric << std::endl;

    //// read data and query
    Logger logger;
    std::vector <VectorDataType> data_list;
    ReadVectorData(dataset_path, data_list);
    std::vector <VectorDataType> query_list;
    std::vector <size_t> query_k_list;
    ReadVectorQuery(query_path, query_list, query_k_list);

    std::cout << "dataset size: " << data_list.size() << std::endl;
    std::cout << "query size: " << query_list.size() << std::endl;
    std::cout << "dim: " << data_list[0].Dimension() << std::endl;

    //// generate ground truth
    logger.SetStartTimer();
    int dim = data_list[0].Dimension();
    std::vector<std::vector<VidType>> answer_list;
    for (const auto& query: query_list) {
        std::priority_queue<std::pair<float, VidType>, std::vector<std::pair<float, VidType>>, Compare> max_heap;
        for (const auto& point: data_list) {
            float dist = metric_func(query.data.data(), point.data.data(), dim);
            if (max_heap.size() < K) {
                max_heap.push(std::make_pair(dist, point.vid));
            } else if (dist < max_heap.top().first) {
                max_heap.pop();
                max_heap.push(std::make_pair(dist, point.vid));
            }
        }
        std::vector<VidType> answer;
        while (!max_heap.empty()) {
            answer.push_back(max_heap.top().second);
            max_heap.pop();
        }
        std::reverse(answer.begin(), answer.end());
        answer_list.push_back(answer);
    }
    logger.SetEndTimer();
    double duration_time = logger.GetDurationTime();
    std::cout << "generate ground truth time: " << duration_time << std::endl;
    //// dump ground truth
    DumpGroundTruth(ground_truth_path, answer_list);
    return 0;
}