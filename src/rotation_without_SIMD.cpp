#define EIGEN_DONT_VECTORIZE
#define EIGEN_DONT_PARALLELIZE
#define _STATISTICS_

#include <iostream>
#include <getopt.h>
#include <fstream>
#include <ctime>
#include <cmath>
#include <queue>
#include "../utils.h"
#include "../Logger/Logger.h"
#include "../FileIO/File_IO.h"

using SearchKnnFuncPtr =
std::priority_queue<std::pair<float, hnswlib::labeltype>>
(hnswlib::HierarchicalNSW<float>::*)(const void *, size_t, hnswlib::BaseFilterFunctor *) const;

SearchKnnFuncPtr knn_ptr = nullptr;

int main(int argc, char *argv[]) {

//#ifdef EIGEN_VECTORIZE_SSE
//    std::cout << "Using SSE instruction set" << std::endl;
//#endif
//
//#ifdef EIGEN_VECTORIZE_AVX
//    std::cout << "Using AVX instruction set" << std::endl;
//#endif
//
//#ifdef EIGEN_VECTORIZE_AVX2
//    std::cout << "Using AVX2 instruction set" << std::endl;
//#endif
//
//#ifdef EIGEN_VECTORIZE_AVX512
//    std::cout << "Using AVX-512 instruction set" << std::endl;
//#endif
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

    int K = 20;
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

#ifdef _FIX_DIMENSION_
    if (init_d == 0) {
        std::cout << "Error: dimension not init" << std::endl;
        exit(1);
    }
    if (op == 0 || op == 7) {
        std::cout << "Error: operator not support" << std::endl;
        exit(1);
    }
#endif

    //// read query and groundtruth
    Logger logger;
    std::vector<VectorDataType> query_list;
    std::vector<size_t> query_k_list;
    ReadVectorQuery(query_path, query_list, query_k_list);

    std::vector<std::vector<VidType>> ground_truth;
    ReadGroundTruth(ground_truth_path, ground_truth);

    std::string msg = std::to_string(K);
    if (op == 6 || op == 7 || op == 14 || op == 15 || op == 16 || op == 17) {
        msg = "hnsw" + msg;
        if (delta_d != 32) {
            msg = "delta" + std::to_string(delta_d) + "_" + msg;
        }
    }

    //// load index
    int dim = query_list[0].Dimension();
    std::unique_ptr<DCO> DCO = setup_DCO(dataset_path, dim, op, msg);

    //// search
    std::vector<size_t> efs;
    efs.push_back(100);
    efs.push_back(K);
    for (int i = 1; i <= 15; ++i) {
        efs.push_back(i * 100);
    }
    if (op >= 2) {
        for (auto &query: query_list) {
            HPLogger hp_logger;
            DCO->query_transform(query.data.data());
            hp_logger.SetEndTimer();
            DCO->add_rota_time(hp_logger.GetDurationTime());
        }
    }
#ifdef _STATISTICS_
    DCO->print_statistics();
#endif
    return 0;
}