#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <getopt.h>
#include "ivf/ivf_cuda2.cuh"
#include "../Logger/Logger.h"
#include "../FileIO/File_IO.h"

using SearchKnnFuncPtr = std::priority_queue <std::pair<float, size_t>> (IVF_CUDA::*)(float *, size_t, size_t, float);
SearchKnnFuncPtr knn_ptr = nullptr;

int main(int argc, char *argv[]) {
    //// parse arguments
    const struct option opts[] = {
            {"help",              no_argument,       0, 'h'},
            {"k",                 required_argument, 0, 'k'},
            {"dataset_path",      required_argument, 0, 'n'},
            {"query_path",        required_argument, 0, 'q'},
            {"ground_truth_path", required_argument, 0, 'g'},
            {"index_path",        required_argument, 0, 'i'},
            {"operator",          required_argument, 0, 'o'},
    };
    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    int K = 100;
    std::string dataset_path;
    std::string query_path;
    std::string ground_truth_path;
    std::string index_path;
    int op = 0;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "k:n:q:g:i:o:", opts, &ind);
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
            case 'i':
                if (optarg) index_path = optarg;
                break;
            case 'o':
                if (optarg) op = atoi(optarg);
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

    if (index_path.empty()) {
        std::cerr << "Error: index_path is empty. Please provide a valid path using the -i option." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "K: " << K << std::endl;
    std::cout << "dataset_path: " << dataset_path << std::endl;
    std::cout << "query_path: " << query_path << std::endl;
    std::cout << "ground_truth_path: " << ground_truth_path << std::endl;
    std::cout << "index_path: " << index_path << std::endl;
    std::cout << "operator: " << op << std::endl;

    //// read query and groundtruth
    Logger logger;
    std::vector <VectorDataType> query_list;
    std::vector <size_t> query_k_list;
    ReadVectorQuery(query_path, query_list, query_k_list);

    std::vector <std::vector<VidType>> ground_truth;
    ReadGroundTruth(ground_truth_path, ground_truth);

    std::string msg = std::to_string(K);
    if (op >= 6) {
        msg = "ivf" + msg;
    }

    //// load index
    int dim = query_list[0].Dimension();
    IVF_CUDA ivf;
    ivf.load(index_path.c_str());
    ivf.setup_dco(dataset_path, op, msg);
    knn_ptr = &IVF_CUDA::searchKNN;


    //// search
    std::vector <size_t> probes;
    for (int i = 1; i <= 20; ++i) {
        probes.push_back(i * 20);
    }

    double rotation_time = 0;
    if (op >= 2) {
        for (auto &query: query_list) {
            HPLogger hp_logger;
            ivf.query_transform(query.data.data());
            hp_logger.SetEndTimer();
            rotation_time += hp_logger.GetDurationTime();
        }
    }
    std::cout << "rotation time = " << std::fixed << std::setprecision(6) << rotation_time
              << " [ms] " << std::endl;

    if (op == 5) {
        knn_ptr = &IVF_CUDA::searchKNNIP;
        logger.SetStartTimer();
        ivf.compute_base_square(false);
        logger.SetEndTimer();
        std::cout << "compute base square time: " << logger.GetDurationTime() << " [ms] " << std::endl;
    }

    if (op == 7) {
        knn_ptr = &IVF_CUDA::searchKNNOPQ;
        logger.SetStartTimer();
        ivf.encoder_origin_data();
        logger.SetEndTimer();
        std::cout << "encoder origin data time: " << logger.GetDurationTime() << " [ms] " << std::endl;
    }

    HPLogger hp_logger;

    for (auto probe: probes) {
        double duration_time = 0;
        std::vector <std::vector<VidType>> answer_list;
        for (auto query: query_list) {
            hp_logger.SetStartTimer();
            std::vector <VidType> result;
            ivf.setQuery(query.data.data());
            std::priority_queue <std::pair<float, size_t>> Result = (ivf.*knn_ptr)(query.data.data(), K, probe,
                                                                                   std::numeric_limits<float>::max());
            while (!Result.empty()) result.emplace_back(Result.top().second), Result.pop();
            std::reverse(result.begin(), result.end());
            answer_list.push_back(result);
            hp_logger.SetEndTimer();
            duration_time += hp_logger.GetDurationTime();
//            std::cout << "query" << query.vid << ": " << "search time = " << std::fixed << std::setprecision(6)
//                      << hp_logger.GetDurationTime() << " [ms] " << std::endl;
        }
        std::cout << "probe" << probe << ": " << "search time = " << std::fixed << std::setprecision(6) << duration_time
                  << " [ms] " << " ";
        EvaluateAnswer(answer_list, ground_truth, K);
    }
    return 0;
}