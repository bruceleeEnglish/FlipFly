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
#include "../../include/ivf/ivf.h"

size_t count_bound = 10000;
size_t elements_bound = 5000000;

int main(int argc, char **argv) {
    //// parse arguments
    const struct option opts[] = {
            {"help",                no_argument,       0, 'h'},
            {"K",                   required_argument, 0, 'k'},
            {"dataset_path",        required_argument, 0, 'n'},
            {"index_path",          required_argument, 0, 'i'},
            {"query_path",          required_argument, 0, 'q'},
            {"ground_truth_path",   required_argument, 0, 'g'},
            {"codebook_path",       required_argument, 0, 'c'},
            {"recall",              required_argument, 0, 'r'},
            {"output_path",         required_argument, 0, 'o'},
            {"linear_path",         required_argument, 0, 'l'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    int K = 100;
    std::string dataset_path;
    std::string query_path;
    std::string ground_truth_path;
    std::string codebook_path;
    std::string index_path;
    std::string output_path;
    std::string linear_path;
    double recall = 0.995;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "k:n:i:q:g:c:r:o:l:", opts, &ind);
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
            case 'c':
                if (optarg) codebook_path = optarg;
                break;
            case 'i':
                if (optarg) index_path = optarg;
                break;
            case 'o':
                if (optarg) output_path = optarg;
                break;
            case 'r':
                if (optarg) recall = atof(optarg);
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

    if (codebook_path.empty()) {
        std::cerr << "Error: codebook_path is empty. Please provide a valid path using the -c option." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (output_path.empty()) {
        std::cerr << "Error: output_path is empty. Please provide a valid path using the -o option." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (linear_path.empty()) {
        std::cerr << "Error: linear_path is empty. Please provide a valid path using the -l option." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "K: " << K << std::endl;
    std::cout << "dataset_path: " << dataset_path << std::endl;
    std::cout << "query_path: " << query_path << std::endl;
    std::cout << "ground_truth_path: " << ground_truth_path << std::endl;
    std::cout << "codebook_path: " << codebook_path << std::endl;
    std::cout << "index_path: " << index_path << std::endl;
    std::cout << "output_path: " << output_path << std::endl;
    std::cout << "linear_path: " << linear_path << std::endl;
    std::cout << "recall: " << recall << std::endl;

    //// load query and groundtruth
    std::vector<VectorDataType> query_list;
    ReadVectorData(query_path, query_list);

    std::vector<std::vector<VidType>> ground_truth_list;
    ReadGroundTruth(ground_truth_path, ground_truth_list);

    //// load index
    int dim = query_list[0].Dimension();
    IVF ivf;
    ivf.load(index_path);
    ivf.dist_compare_operator_ = setup_DCO(dataset_path, dim, 0);
    int probe = 50;

    //// load transformation
    Logger logger;
    logger.SetStartTimer();
    std::unique_ptr<DDCopq> DCO = std::make_unique<DDCopq>(dataset_path, dim, false);
    count_bound = std::min(count_bound, (size_t) query_list.size());
    for (auto &query: query_list) {
        DCO->query_transform(query.data.data());
    }
    ivf.encoder_origin_data(DCO.get());

    ////  prepare test
    std::cerr << "test begin" << std::endl;
    std::vector<std::vector<std::tuple<unsigned, float, float> > > res(query_list.size());
    for (int i = 0; i < query_list.size(); ++i) {
        std::vector<std::tuple<unsigned, float, float>> result = ivf.searchKnnLogger(query_list[i].data.data(), K, probe);
        res[i] = result;
        if (i % 500 == 0) std::cerr << "test " << i << " done" << std::endl;
    }

    std::vector<float> acc, app, thresh, cluster;
    std::unordered_map<size_t, bool> KNNmap;
    std::vector<size_t> id_to_L1;
    size_t feature_dim = 3, model_count = 1;
    feature_dim += model_count;
    std::cerr << "feature dim:: " << feature_dim << " models:: " << model_count << std::endl;

    id_to_L1.resize(ivf.N_);
    for (int i = 0; i < id_to_L1.size(); i++) {
        id_to_L1[ivf.id_[i]] = i;
    }

    //// positive samples
    for (int i = 0; i < count_bound; i++) {
        std::vector<VidType> gt = ground_truth_list[i];
        float *q = query_list[i].data.data();
        DCO->calc_dist_map(q);
        float thresh_dist = L2Distance(q, ivf.res_data_ + id_to_L1[gt[K - 1]] * dim, dim);
        for (int j = 0; j < K; j++) {
            size_t L1_id = id_to_L1[gt[j]];
            float acc_dist = L2Distance(q, ivf.res_data_ + L1_id * dim, dim);
            float app_dist = DCO->naive_product_map_dist(L1_id);
            float cluster_dist = DCO->node_cluster_dist_[L1_id];
            app.push_back(app_dist);
            acc.push_back(acc_dist);
            cluster.push_back(cluster_dist);
            thresh.push_back(thresh_dist);
            KNNmap[L1_id] = true;
        }
    }

    std::cout<<"positive samples" << std::endl;

    //// negative samples
    for (int i = 0; i < count_bound; i++) {
        std::vector<VidType> gt = ground_truth_list[i];
        float *q = query_list[i].data.data();
        DCO->calc_dist_map(q);
        float thresh_dist = L2Distance(q, ivf.res_data_ + id_to_L1[gt[K - 1]] * dim, dim);
        for (auto u : res[i]) {
            size_t L1_id = std::get<0>(u);
            if (KNNmap[L1_id]) continue;
            float acc_dist = std::get<1>(u);
            float app_dist = DCO->naive_product_map_dist(L1_id);
            float cluster_dist = DCO->node_cluster_dist_[L1_id];
            app.push_back(app_dist);
            acc.push_back(acc_dist);
            cluster.push_back(cluster_dist);
            thresh.push_back(thresh_dist);
        }
        if (acc.size() > elements_bound) break;
    }

    std::cerr << "negative samples" << std::endl;

    ////  write training set
    std::ofstream out(output_path, std::ios::binary);
    int knn_count = 0;
    for (int tag = 0; tag < acc.size(); tag++) {
        if ((double) acc[tag] < (double) thresh[tag] + 1e-8) {
            out.write((char *) &feature_dim, sizeof(unsigned));
            out.write((char *) &acc[tag], sizeof(float));
            out.write((char *) &app[tag], sizeof(float));
            out.write((char *) &cluster[tag], sizeof(float));
            out.write((char *) &thresh[tag], sizeof(float));
            knn_count++;
        }
    }
    std::cerr << knn_count << std::endl;

    static std::default_random_engine Engine;
    static std::uniform_int_distribution<unsigned> rand(0, acc.size());
    for (int tag = 0; tag < std::min((unsigned long) elements_bound, acc.size()); tag++) {
        unsigned rand_index = rand(Engine);
        if (acc[rand_index] > thresh[rand_index]) {
            out.write((char *) &feature_dim, sizeof(unsigned));
            out.write((char *) &acc[rand_index], sizeof(float));
            out.write((char *) &app[rand_index], sizeof(float));
            out.write((char *) &cluster[rand_index], sizeof(float));
            out.write((char *) &thresh[rand_index], sizeof(float));
        }
    }
    out.close();

    logger.SetEndTimer();
    double duration_time = logger.GetDurationTime();
    std::cout << "generate training set time: " << duration_time << "[ms]" << std::endl;

    //// load linear model
    logger.SetStartTimer();
    std::ifstream f(linear_path);
    if (f.good()) {
        size_t count_base = count_bound * K;
        DCO->load_model_param(linear_path);
        std::ofstream fout(linear_path);
        fout.setf(std::ios::fixed, std::ios::floatfield);
        fout.precision(6);
        fout << DCO->model_count_ << std::endl;
        DCO->binary_search_multi_linear(acc.size(), app.data(), acc.data(), thresh.data(), cluster.data(), count_base, recall);
        fout << DCO->W_[0] << " " << DCO->W_[1] << " " << DCO->B_[0] << std::endl;
    }
    logger.SetEndTimer();
    duration_time = logger.GetDurationTime();
    std::cout << "correct linear model time: " << duration_time << "[ms]" << std::endl;

    return 0;
}