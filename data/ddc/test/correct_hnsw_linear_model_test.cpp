#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <getopt.h>
#include "../../../FileIO/File_IO.h"
#include "../../../include/hnswlib/hnswlib.h"
#include "../../../Logger/Logger.h"
#include "../../../utils.h"
#include "../../../global.h"

size_t count_bound = 10000;
size_t elements_bound = 5000000;

int main(int argc, char **argv) {

    int K = 100;
    std::string dataset_path = "/home/zzlin/dataset/YouTube/LYouTube_base.fvecs";
    std::string query_path = "/home/zzlin/dataset/YouTube/YouTube_train.fvecs";
    std::string training_set_path = "/home/zzlin/dataset/YouTube/ddc/YouTube_hnsw_pca_100_training_set.fvecs";
    std::string linear_path = "/home/zzlin/dataset/YouTube/ddc/linear_hnsw100_pca.txt";
    double recall = 0.995;


    std::cout << "K: " << K << std::endl;
    std::cout << "dataset_path: " << dataset_path << std::endl;
    std::cout << "query_path: " << query_path << std::endl;
    std::cout << "training_set_path: " << training_set_path << std::endl;
    std::cout << "recall: " << recall << std::endl;
    std::cout << "linear_path: " << linear_path << std::endl;

    //// load query and training set
    std::vector<VectorDataType> query_list;
    ReadVectorData(query_path, query_list);

    std::vector<VectorDataType> training_set;
    ReadVectorData(training_set_path, training_set);

    int dim = query_list[0].Dimension();
    size_t sub_dim = delta_d;
    size_t feature_dim = 2, model_count = dim / sub_dim;
    if (dim % sub_dim) model_count++;
    feature_dim += model_count;
    count_bound = std::min(count_bound, (size_t) query_list.size());

    //// assemble
    std::vector<float> acc, thresh;
    std::vector<std::vector<float> > app;
    app.resize(model_count);
    for (auto &record: training_set) {
        acc.push_back(record.data[0]);
        for (int j = 0; j < model_count; ++j) {
            app[j].push_back(record.data[j + 1]);
        }
        thresh.push_back(record.data[model_count + 1]);
    }
    //// load linear model
    Logger logger;
    logger.SetStartTimer();
    double exp_recall = 1.0 - (1.0 - recall) / (model_count - 1.0);
    double cur_recall = 1.0;
    std::cerr << "save finished with recall:: " << recall << " " << exp_recall << std::endl;
    std::unique_ptr<DDCpca> DCO = std::make_unique<DDCpca>(dataset_path, dim, false);
    std::ifstream f(linear_path);
    if (f.good()) {
        size_t count_base = count_bound * K;
        DCO->load_model_param(linear_path);
        std::ofstream fout(linear_path);
        fout.setf(std::ios::fixed, std::ios::floatfield);
        fout.precision(6);
//        fout << DCO->model_count_ << std::endl;
        if (DCO->model_count_ != model_count) {
            std::cerr << "model count mismatch" << std::endl;
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < model_count; i++) {
            cur_recall *= exp_recall;
            std::cerr << cur_recall << std::endl;
            double tmp_recall = cur_recall;
            if (i == model_count - 1) tmp_recall = 1;
            DCO->binary_search_single_linear(acc.size(), app[i].data(), acc.data(), thresh.data(), i, count_base,
                                             tmp_recall);
//            fout << DCO->W_[i] << " " << DCO->B_[i] << " " << DCO->b_[i] << std::endl;
        }
    }
    logger.SetEndTimer();
    double duration_time = logger.GetDurationTime();
    std::cout << "correct linear model time: " << duration_time << "[ms]" << std::endl;
    return 0;
}
