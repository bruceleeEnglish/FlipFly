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

int main(int argc, char *argv[]) {
    //// parse arguments
    const struct option opts[] = {
            {"help",                no_argument,       0, 'h'},
            {"K",                   required_argument, 0, 'k'},
            {"dataset_path",        required_argument, 0, 'n'},
            {"query_path",          required_argument, 0, 'q'},
            {"ground_truth_path",   required_argument, 0, 'g'},
            {"metric",              required_argument, 0, 'm'},
    };
    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    int K = 256;
    std::string dataset_path;
    std::string query_path;
    std::string ground_truth_path;
    std::string log_file;
    int metric = 0;
    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "k:n:q:g:m:", opts, &ind);
        switch (iarg) {
            case 'k':
                if (optarg) K = atoi(optarg);
                break;
            case 'n':
                if (optarg) dataset_path = optarg;
                break;
            case 'q':
                if (optarg) query_path = optarg;
                break;
            case 'g':
                if (optarg) ground_truth_path = optarg;
                break;
            case 'm':
                if (optarg) metric = atoi(optarg);
                break;
        }
    }

    //// check arguments
    if (dataset_path.empty()) {
        std::cerr << "Error: dataset_path is empty. Please provide a valid path using the -n option." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (query_path.empty()) {
        std::cerr << "Error: query_path is empty. Please provide a valid path using the -q option." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (ground_truth_path.empty()) {
        std::cerr << "Error: ground_truth_path is empty. Please provide a valid path using the -g option." << std::endl;
        exit(EXIT_FAILURE);
    }

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

    //// check
    std::vector<std::vector<VidType>> answer_list;
    ReadGroundTruth(ground_truth_path, answer_list);
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            float dist = metric_func(query_list[i].data.data(), data_list[answer_list[i][j]].data.data(), data_list[0].Dimension());
            std::cout << dist << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}