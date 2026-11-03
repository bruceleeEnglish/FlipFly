#include <iostream>
#include <getopt.h>
#include <fstream>
#include <ctime>
#include <cmath>
#include <queue>
#include <set>
#include "../utils.h"
#include "../Logger/Logger.h"
#include "../FileIO/File_IO.h"

void EvaluateAnswerSingle(std::vector<VidType> &answer_list, std::vector<VidType> &truth_list, int K) {
        std::set<VidType> truth_set(truth_list.begin(), std::next(truth_list.begin(), K));

        double recall = 0.0;

        for (VidType vid: answer_list) {
            if (truth_set.count(vid) > 0) {
                recall += 1.0;
            }
        }
        if (answer_list.size() > 0)
            recall /= answer_list.size();

        printf("recall = %.6f\n", recall);
}

using SearchKnnFuncPtr =
std::priority_queue<std::pair<float, hnswlib::labeltype>>
(hnswlib::HierarchicalNSW<float>::*)(const void *, size_t, hnswlib::BaseFilterFunctor *) const;

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
    if (op == 6 || op == 7 || op == 14 || op == 15) {
        msg = "hnsw" + msg;
        if (delta_d != 32) {
            msg = "delta" + std::to_string(delta_d) + "_" + msg;
        }
    }

    //// load index
    int dim = query_list[0].Dimension();
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, index_path);
    alg_hnsw->dist_compare_operator_ = setup_DCO(dataset_path, dim, op, msg);
    knn_ptr = &hnswlib::HierarchicalNSW<float>::searchKnn;

    //// search
    std::vector<size_t> efs;
    efs.push_back(200);
    if (op >= 2) {
        for (auto &query: query_list) {
            HPLogger hp_logger;
            alg_hnsw->dist_compare_operator_->query_transform(query.data.data());
            hp_logger.SetEndTimer();
            alg_hnsw->dist_compare_operator_->add_rota_time(hp_logger.GetDurationTime());
        }
    }

    if (op == 5) {
        knn_ptr = &hnswlib::HierarchicalNSW<float>::searchKnnIP;
        logger.SetStartTimer();
        alg_hnsw->compute_base_square(false);
        logger.SetEndTimer();
        std::cout << "compute base square time: " << logger.GetDurationTime() << " [ms] " << std::endl;
    }

    if (op == 7) {
        knn_ptr = &hnswlib::HierarchicalNSW<float>::searchKnnOPQ;
        logger.SetStartTimer();
        alg_hnsw->encoder_origin_data();
        logger.SetEndTimer();
        std::cout << "encoder origin data time: " << logger.GetDurationTime() << " [ms] " << std::endl;
    }

    if (op == 9 || op == 10 || op == 11 || op == 12) {
        knn_ptr = &hnswlib::HierarchicalNSW<float>::searchKnnNorm;
    }

    if (op == 13) {
        knn_ptr = &hnswlib::HierarchicalNSW<float>::searchKnnNormIP;
        logger.SetStartTimer();
        alg_hnsw->compute_base_product_mean(true);
        logger.SetEndTimer();
        std::cout << "compute base product mean time: " << logger.GetDurationTime() << " [ms] " << std::endl;
    }

    if (op == 15) {
        knn_ptr = &hnswlib::HierarchicalNSW<float>::searchKnnOPQIP;
        logger.SetStartTimer();
        alg_hnsw->encoder_origin_data_ip();
        logger.SetEndTimer();
        std::cout << "ip encoder origin data time: " << logger.GetDurationTime() << " [ms] " << std::endl;
    }

//    alg_hnsw->dist_compare_operator_->cancel_SIMD();
//    std::cout << "without SIMD" << std::endl;
    HPLogger hp_logger;
    for (auto ef : efs) {
        alg_hnsw->setEf(ef);
        double duration_time = 0;
        std::vector<std::vector<VidType>> answer_list;
        for (auto query: query_list) {
            hp_logger.SetStartTimer();
            std::vector<VidType> result;
            std::priority_queue<std::pair<float, hnswlib::labeltype>> Result = (alg_hnsw->*knn_ptr)(query.data.data(), K, nullptr);
            while (!Result.empty()) result.emplace_back(Result.top().second), Result.pop();
            std::reverse(result.begin(), result.end());
            answer_list.push_back(result);
            hp_logger.SetEndTimer();
            duration_time += hp_logger.GetDurationTime();
            EvaluateAnswerSingle(result, ground_truth[answer_list.size() - 1], K);
        }
    }
#ifdef _STATISTICS_
    alg_hnsw->dist_compare_operator_->print_statistics();
#endif
    delete alg_hnsw;
    return 0;
}
