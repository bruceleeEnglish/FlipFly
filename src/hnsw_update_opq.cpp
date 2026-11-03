#define _OPQ_builder_
#include <iostream>
#include <getopt.h>
#include <fstream>
#include <ctime>
#include <cmath>
#include <queue>
#include "../../utils.h"
#include "../../Logger/Logger.h"
#include "../../FileIO/File_IO.h"
#include "../../FileIO/DataType.hpp"
#include "../../include/hnswlib/hnswlib.h"

using SearchKnnFuncPtr =
std::priority_queue<std::pair<float, hnswlib::labeltype>>
(hnswlib::HierarchicalNSW<float>::*)(const void *, size_t, hnswlib::BaseFilterFunctor *) const;

SearchKnnFuncPtr knn_ptr = nullptr;

void save_pq_mp_binary(const std::string& filename, std::vector<uint8_t> &pq_mp_) {
    std::ofstream out(filename, std::ios::binary);
    size_t size = pq_mp_.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(reinterpret_cast<const char*>(pq_mp_.data()), size * sizeof(uint8_t));
    out.close();
}

void load_pq_mp_binary(const std::string& filename, std::vector<uint8_t> &pq_mp_) {
    std::ifstream in(filename, std::ios::binary);
    size_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    pq_mp_.resize(size);
    in.read(reinterpret_cast<char*>(pq_mp_.data()), size * sizeof(uint8_t));
    in.close();
}

void save_node_cluster_dist_binary(const std::string& filename, std::vector<float> &node_cluster_dist_) {
    std::ofstream out(filename, std::ios::binary);
    size_t size = node_cluster_dist_.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(reinterpret_cast<const char*>(node_cluster_dist_.data()), size * sizeof(float));
    out.close();
}

void load_node_cluster_dist_binary(const std::string& filename, std::vector<float> &node_cluster_dist_) {
    std::ifstream in(filename, std::ios::binary);
    size_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    node_cluster_dist_.resize(size);
    in.read(reinterpret_cast<char*>(node_cluster_dist_.data()), size * sizeof(float));
    in.close();
}

int main(int argc, char *argv[]) {
    //// parse arguments
//    int K = 100;
//    std::string dataset_path = "/home/dataset/gist/train_600000/Qgist_base.fvecs";
//    std::string query_path = "/home/dataset/gist/query.fvecs";
//    std::string ground_truth_path = "/home/dataset/gist/gist_size700000_gt.ivecs";
//    std::string index_path = "/home/dataset/gist/train_600000/tmp_Qgist_M16_efConstruction500_operator0.index";
//    int op = 7;
//    int start_pos = 600000;
//    int end_pos = 700000;

    //// parse arguments
    const struct option opts[] = {
            {"help",              no_argument,       0, 'h'},
            {"k",                 required_argument, 0, 'k'},
            {"dataset_path",      required_argument, 0, 'n'},
            {"query_path",        required_argument, 0, 'q'},
            {"ground_truth_path", required_argument, 0, 'g'},
            {"index_path",        required_argument, 0, 'i'},
            {"operator",          required_argument, 0, 'o'},
            {"start",             required_argument, 0, 's'},
            {"end   ",            required_argument, 0, 'e'},
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
    int start_pos = 0;
    int end_pos = 0;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "k:n:q:g:i:o:s:e:", opts, &ind);
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
            case 's':
                if (optarg) start_pos = atoi(optarg);
                break;
            case 'e':
                if (optarg) end_pos = atoi(optarg);
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
    std::cout << "start_pos: " << start_pos << std::endl;
    std::cout << "end_pos: " << end_pos << std::endl;

    //// read data
    std::vector<VectorDataType> data_list;
    ReadVectorData(dataset_path, data_list);
    std::vector<VectorDataType> query_list;
    std::vector<size_t> query_k_list;
    ReadVectorQuery(query_path, query_list, query_k_list);
    std::vector<std::vector<VidType>> ground_truth;
    ReadGroundTruth(ground_truth_path, ground_truth);

    //// load index
    int dim = query_list[0].Dimension();
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, index_path);
    alg_hnsw->dist_compare_operator_ = setup_DCO(dataset_path, dim, op, "");


    //// rotation
    int curr_elements = alg_hnsw->getCurrentElementCount();
    if (op >= 2) {
        for (int i = 0; i < data_list.size(); ++i) {
            alg_hnsw->dist_compare_operator_->query_transform((void *) data_list[i].data.data());
            if (i % 50000 == 0) std::cout << "data progress: " << i << " / " << data_list.size() << std::endl;
        }

        for (int i = 0; i <curr_elements; ++i) {
            char *dataPoint = alg_hnsw->getDataByInternalId(i);
            alg_hnsw->dist_compare_operator_->query_transform((void *) dataPoint);
            if (i % 50000 == 0) std::cout << "index progress: " << i << " / " << curr_elements << std::endl;
        }
    }

    //// update index
    std::string msg = std::to_string(K);
    if (op >= 6) {
        msg = "hnsw" + msg;
        if (delta_d != 32) {
            msg = "delta" + std::to_string(delta_d) + "_" + msg;
        }
    }

//    alg_hnsw->dist_compare_operator_.reset();
    alg_hnsw->dist_compare_operator_ = setup_DCO(dataset_path, dim, op, msg);
    knn_ptr = &hnswlib::HierarchicalNSW<float>::searchKnn;

    if (op == 7) {
        knn_ptr = &hnswlib::HierarchicalNSW<float>::searchKnnOPQ;
        DDCopq * ddcopq = dynamic_cast<DDCopq *>(alg_hnsw->dist_compare_operator_.get());
        ddcopq->pq_mp_.resize(data_list.size() * ddcopq->sub_vector_);
        ddcopq->node_cluster_dist_.resize(data_list.size());
        double ave_dist = 0.0;
            for (int i = 0; i < data_list.size(); i++) {
                float dist_to_centroid = 0.0;
                for (int j = 0; j < ddcopq->sub_vector_; j++) {
                    uint8_t belong = 0;
                    float dist = ddcopq->distfunc_sub_((float *) data_list[i].data.data() + j * ddcopq->sub_dim_,
                                                       ddcopq->pq_book_[j][0].data(),
                                                       ddcopq->dist_func_param_sub_);
                    for (int k = 1; k < ddcopq->sub_cluster_count_; k++) {
                        float new_dist = ddcopq->distfunc_sub_((float *) data_list[i].data.data() + j * ddcopq->sub_dim_,
                                                               ddcopq->pq_book_[j][k].data(),
                                                               ddcopq->dist_func_param_sub_);
                        if (new_dist < dist) {
                            belong = k;
                            dist = new_dist;
                        }
                    }
                    dist_to_centroid += dist;
                    ddcopq->pq_mp_[i * ddcopq->sub_vector_ + j] = belong;
                }
                ddcopq->node_cluster_dist_[i] = dist_to_centroid;
                ave_dist += dist_to_centroid;
                if (i % 50000 == 0) std::cerr << "Encoder progress: " << i << " / " << data_list.size() << std::endl;
            }
            std::cerr << "Encoder ave dist:: " << ave_dist / data_list.size() << std::endl;
//        save_pq_mp_binary("/home/dataset/gist/train_600000/pq_mp.bin", ddcopq->pq_mp_);
//        save_node_cluster_dist_binary("/home/dataset/gist/train_600000/node_cluster_dist.bin", ddcopq->node_cluster_dist_);
//        load_pq_mp_binary("/home/dataset/gist/train_600000/pq_mp.bin", ddcopq->pq_mp_);
//        load_node_cluster_dist_binary("/home/dataset/gist/train_600000/node_cluster_dist.bin", ddcopq->node_cluster_dist_);
    }

    Logger logger_update;
    logger_update.SetStartTimer();
    if (start_pos < end_pos) {
        for (int i = start_pos; i < end_pos; i++) {
//            if ((i-start_pos) % 1000 == 0) std::cout << "update progress: " << i << " / " << end_pos << std::endl;
            alg_hnsw->addPoint(data_list[i].data.data(), data_list[i].vid);
        }
    } else if(start_pos > end_pos) {
        for (int i = start_pos - 1; i >= end_pos; i--) {
            alg_hnsw->markDelete(data_list[i].vid);
        }
    } else {
        std::cout << "start_pos == end_pos" << std::endl;
    }
    logger_update.SetEndTimer();
    std::cout << "update time = " << std::fixed << std::setprecision(6) << logger_update.GetDurationTime() << " [ms] "<< std::endl;

    Logger logger;

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
            std::cout << "query" << query.vid << ": " << "search time = " << std::fixed << std::setprecision(6) << hp_logger.GetDurationTime() << " [ms] " << std::endl;
        }
        std::cout << "ef" << ef << ": "<< "search time = " << std::fixed << std::setprecision(6) << duration_time << " [ms] "<< " ";
        EvaluateAnswer(answer_list, ground_truth, K);
    }
#ifdef _STATISTICS_
    alg_hnsw->dist_compare_operator_->print_statistics();
#endif
    delete alg_hnsw;
    return 0;
}