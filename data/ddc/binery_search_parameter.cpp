#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <getopt.h>
#include "../../FileIO/File_IO.h"
#include "../../include/hnswlib/hnswlib.h"
#include "../../Logger/Logger.h"
#include "../../utils.h"
#include "../../global.h"


size_t count_bound = 10000;

int main(int argc, char *argv[]) {
    //// parse arguments
    const struct option opts[] = {
            {"help",              no_argument,       0, 'h'},
            {"k",                 required_argument, 0, 'k'},
            {"dataset",           required_argument, 0, 'n'},
            {"query_path",        required_argument, 0, 'q'},
            {"ground_truth_path", required_argument, 0, 'g'},
            {"linear_path",       required_argument, 0, 'l'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    std::string query_path;
    std::string ground_truth_path;
    std::string dataset_path;
    std::string linear_path;
    int K = 100;
    double recall = 0.995;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "k:n:q:g:l:", opts, &ind);
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
            case 'l':
                if (optarg) linear_path = optarg;
                break;
        }
    }

    //// check arguments
    if (dataset_path.empty()) {
        std::cerr << "Error: dataset_path is empty. Please provide a valid path using the -n option." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (query_path.empty()) {
        std::cerr << "Error: dataset_path is empty. Please provide a valid path using the -n option." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (ground_truth_path.empty()) {
        std::cerr << "Error: dataset_path is empty. Please provide a valid path using the -n option." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (linear_path.empty()) {
        std::cerr << "Error: dataset_path is empty. Please provide a valid path using the -n option." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "K: " << K << std::endl;
    std::cout << "dataset_path: " << dataset_path << std::endl;
    std::cout << "query_path: " << query_path << std::endl;
    std::cout << "ground_truth_path: " << ground_truth_path << std::endl;
    std::cout << "linear_path: " << linear_path << std::endl;
    std::cout << "recall: " << recall << std::endl;

    //// load data
    std::vector<VectorDataType> data_list;
    ReadVectorData(dataset_path, data_list);
    std::vector<VectorDataType> query_list;
    ReadVectorData(query_path, query_list);
    std::vector<std::vector<VidType>> ground_truth_list;
    ReadGroundTruth(ground_truth_path, ground_truth_list);

    int dim = query_list[0].Dimension();
    std::vector<float> acc, thresh;
    std::vector<std::vector<float>> app;
    std::unique_ptr<DDCpca> DCO = std::make_unique<DDCpca>(dataset_path, dim, false);
    for (auto &query: query_list) {
        DCO->query_transform(query.data.data());
    }
    size_t sub_dim = delta_d;
    size_t model_count = dim / sub_dim;
    if (dim % sub_dim) model_count++;
    app.resize(model_count);
    std::cerr << "test begin" << std::endl;
    count_bound = std::min(count_bound, (size_t) ground_truth_list.size());
    for (size_t i = 0; i < count_bound; i++) {
        std::vector<VidType> gt = ground_truth_list[i];
        float *q = query_list[i].data.data();
        float *p = data_list[gt[K - 1]].data.data();
        float thresh_dist = L2Distance(p, q, dim);
        for (int j = 0; j < K; j++) {
            p = data_list[gt[j]].data.data();
            float app_dist;
            float acc_dist = L2Distance(p, q, dim);
            app_dist = 0;
            size_t app_count = 0;
            for (unsigned k = 0; k < dim; k += sub_dim) {
                if (k + sub_dim > dim) app_dist += L2Distance(q + k, p + k, dim % sub_dim);
                else app_dist += L2Distance(q + k, p + k, sub_dim);
                app[app_count].push_back(app_dist);
                app_count++;
            }
            acc.push_back(acc_dist);
            thresh.push_back(thresh_dist);
        }
    }
    DCO->W_.resize(model_count);
    DCO->B_.resize(model_count);
    DCO->b_.resize(model_count);
    Eigen::VectorXf Y = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(acc.data(), acc.size());
    float y_mean = Y.mean();
    for (int i = 0; i < model_count; i++) {
        Eigen::VectorXf X = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(app[i].data(), app[i].size());
        float x_mean = X.mean();
        // compute OLS
        Eigen::VectorXf X_centered = X.array() - x_mean;
        Eigen::VectorXf Y_centered = Y.array() - y_mean;
        float w = (X_centered.cwiseProduct(Y_centered).sum()) / (X_centered.cwiseProduct(X_centered).sum());
        float b = y_mean - w * x_mean;
        std::cerr << "OLS w: " << w << std::endl;
        std::cerr << "OLS b: " << b << std::endl;
        DCO->W_[i] = w;
        DCO->B_[i] = b;
        DCO->b_[i] = b;
    }

    std::cerr << " models:: " << model_count << " sub dim:: " << sub_dim << std::endl;
    std::cerr << "target recall:: " << recall << std::endl;
    double exp_recall = 1.0 - (1.0 - recall) / (model_count - 1.0);
    std::cerr << "save finished with recall:: " << recall << " " << exp_recall << std::endl;
    size_t count_base = count_bound * K;
    std::ofstream fout(linear_path);
    fout.setf(std::ios::fixed, std::ios::floatfield);
    fout.precision(6);
    fout << model_count << std::endl;
    for (int i = 0; i < model_count; i++) {
        std::cerr << exp_recall << std::endl;
        if (i == model_count - 1) {
            DCO->W_[model_count - 1] = 1.0;
            DCO->B_[model_count - 1] = 0;
            DCO->b_[model_count - 1] = 0;
        } else {
            DCO->binary_search_single_linear(acc.size(), app[i].data(), acc.data(), thresh.data(), i, count_base,
                                             exp_recall);
        }
        fout << DCO->W_[i] << " " << DCO->B_[i] << " " << DCO->b_[i] << endl;
    }
    return 0;
}