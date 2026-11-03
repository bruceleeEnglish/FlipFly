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

int main() {

    int K = 100;
    std::string dataset_path = "/home/zzlin/dataset/YouTube/LYouTube_base.fvecs";
    std::string query_path = "/home/zzlin/dataset/YouTube/YouTube_train.fvecs";
    std::string ground_truth_path = "/home/zzlin/dataset/YouTube/YouTube_train_ground_truth.ivecs";
    std::string index_path = "/home/zzlin/dataset/YouTube/hnsw_LYouTube_M16_efConstruction500_operator0.index";
    std::string output_path = "/home/zzlin/dataset/YouTube/ddc/YouTube_hnsw_pca_100_training_set.fvecs";
    double recall = 0.995;

    std::cout << "K: " << K << std::endl;
    std::cout << "dataset_path: " << dataset_path << std::endl;
    std::cout << "query_path: " << query_path << std::endl;
    std::cout << "ground_truth_path: " << ground_truth_path << std::endl;
    std::cout << "index_path: " << index_path << std::endl;
    std::cout << "output_path: " << output_path << std::endl;
    std::cout << "recall: " << recall << std::endl;

    //// load query and groundtruth
    std::vector<VectorDataType> query_list;
    ReadVectorData(query_path, query_list);

    std::vector<std::vector<VidType>> ground_truth_list;
    ReadGroundTruth(ground_truth_path, ground_truth_list);

    //// load index
    int dim = query_list[0].Dimension();
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, index_path);
    alg_hnsw->dist_compare_operator_ = setup_DCO(dataset_path, dim, 0);
    alg_hnsw->setEf(1000);
    std::cerr << alg_hnsw->cur_element_count << " " << dim << std::endl;

    //// load transformation
    std::unique_ptr<DDCpca> DCO = std::make_unique<DDCpca>(dataset_path, dim, false);
    count_bound = std::min(count_bound, (size_t) query_list.size());
    for (auto &query: query_list) {
        DCO->query_transform(query.data.data());
    }

    ////  prepare test
    std::cerr << "test begin" << std::endl;
    std::vector<std::vector<std::tuple<unsigned, float, float> > > res(query_list.size());
    for (int i = 0; i < query_list.size(); ++i) {
        std::vector<std::tuple<unsigned, float, float>> result = alg_hnsw->searchKnnLogger(query_list[i].data.data(), K, 2);
        res[i] = result;
        if (i % 500 == 0) std::cerr << "test " << i << " done" << std::endl;
    }

    std::vector<float> acc, thresh;
    std::vector<std::vector<float> > app;
    std::unordered_map<size_t, bool> KNNmap;
    std::vector<size_t> id_to_L1;
    size_t sub_dim = delta_d;
    size_t feature_dim = 2, model_count = dim / sub_dim;
    if (dim % sub_dim) model_count++;
    feature_dim += model_count;
    std::cerr << "feature dim:: " << feature_dim << " models:: " << model_count << " sub dim:: " << sub_dim
              << std::endl;
    std::cerr << "target recall:: " << recall << std::endl;

    app.resize(model_count);
    id_to_L1.resize(alg_hnsw->cur_element_count);
    for (int i = 0; i < id_to_L1.size(); i++) {
        id_to_L1[alg_hnsw->getExternalLabel(i)] = i;
    }

    //// positive samples
    for (int i = 0; i < count_bound; i++) {
        std::vector<VidType> gt = ground_truth_list[i];
        float *q = query_list[i].data.data();
        float thresh_dist = L2Distance(q, (float *) alg_hnsw->getDataByInternalId(id_to_L1[gt[K - 1]]), dim);
        for (int j = 0; j < K; j++) {
            size_t L1_id = id_to_L1[gt[j]];
            float acc_dist = L2Distance(q, (float *) alg_hnsw->getDataByInternalId(L1_id), dim);
            float app_dist = 0;
            size_t app_count = 0;
            auto *p = (float *) alg_hnsw->getDataByInternalId(L1_id);
            for (size_t k = 0; k < dim; k += sub_dim) {
                if (k + sub_dim > dim) app_dist += L2Distance(q + k, p + k, dim % sub_dim);
                else app_dist += L2Distance(q + k, p + k, sub_dim);
                app[app_count].push_back(app_dist);
                app_count++;
            }
            acc.push_back(acc_dist);
            thresh.push_back(thresh_dist);
            KNNmap[L1_id] = true;
        }
    }
    std::cerr << sub_dim << std::endl;

    //// negative samples
    for (int i = 0; i < count_bound; i++) {
        std::vector<VidType> gt = ground_truth_list[i];
        float *q = query_list[i].data.data();
        float thresh_dist = L2Distance(q, (float *) alg_hnsw->getDataByInternalId(id_to_L1[gt[K - 1]]), dim);
        for (auto u : res[i]) {
            size_t L1_id = std::get<0>(u);
            if (KNNmap[L1_id]) continue;
            float node_dist = std::get<1>(u);
            float app_dist = 0;
            unsigned app_count = 0;
            auto *p = (float *) alg_hnsw->getDataByInternalId(L1_id);
            for (size_t k = 0; k < dim; k += sub_dim) {
                if (k + sub_dim > dim) app_dist += L2Distance(q + k, p + k, dim % sub_dim);
                else app_dist += L2Distance(q + k, p + k, sub_dim);
                app[app_count].push_back(app_dist);
                app_count++;
            }
            acc.push_back(node_dist);
            thresh.push_back(thresh_dist);
        }
        if (acc.size() > elements_bound) break;
    }
    delete alg_hnsw;
    return 0;
}