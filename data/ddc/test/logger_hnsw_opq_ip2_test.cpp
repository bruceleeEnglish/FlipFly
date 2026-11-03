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

float normalizeVector(float* vec, size_t dim) {
    float magnitude = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        magnitude += vec[i] * vec[i];
    }
    magnitude = std::sqrt(magnitude);

    if (magnitude > 1e-8f) {
        for (size_t i = 0; i < dim; ++i) {
            vec[i] /= magnitude;
        }
    }
    return magnitude;
}

int main(int argc, char **argv) {
    int K = 100;
    std::string dataset_path = "/home/zzlin/dataset/wiki/Qwiki_base.fvecs";
    std::string query_path = "/home/zzlin/dataset/wiki/wiki_train.fvecs";
    std::string ground_truth_path = "/home/zzlin/dataset/wiki/wiki_train_ground_truth_ip.ivecs";
    std::string index_path = "/home/zzlin/dataset/wiki/hnsw_wiki_M16_efConstruction500_operator8.index";
    std::string output_path = "/home/zzlin/dataset/wiki/ddc/wiki_hnsw_opq_100_training_set_ip2.fvecs";
    std::string linear_path = "/home/zzlin/dataset/wiki/ddc/linear_hnsw100_opq_ip2.txt";
    double recall = 0.995;


    //// load query and groundtruth
    std::vector<VectorDataType> query_list;
    ReadVectorData(query_path, query_list);

    std::vector<std::vector<VidType>> ground_truth_list;
    ReadGroundTruth(ground_truth_path, ground_truth_list);

    //// load index
    int dim = query_list[0].Dimension();
    hnswlib::InnerProductSpace space(dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, index_path);
    alg_hnsw->dist_compare_operator_ = setup_DCO(dataset_path, dim, 0);
    alg_hnsw->setEf(1000);
    std::cerr << alg_hnsw->cur_element_count << " " << dim << std::endl;

    //// load transformation
    Logger logger;
    logger.SetStartTimer();
    std::unique_ptr<DDCopqIP2> DCO = std::make_unique<DDCopqIP2>(dataset_path, dim, false);
    count_bound = std::min(count_bound, (size_t) query_list.size());

    ////  prepare test
    std::cerr << "test begin" << std::endl;
    std::vector<std::vector<std::tuple<unsigned, float, float> > > res(query_list.size());
    for (int i = 0; i < query_list.size(); ++i) {
        std::vector<std::tuple<unsigned, float, float>> result = alg_hnsw->searchKnnLogger(query_list[i].data.data(), K, 2);
        res[i] = result;
        if (i % 500 == 0) std::cerr << "test " << i << " done" << std::endl;
    }

    std::string new_index_path = "/home/zzlin/dataset/wiki/hnsw_norm_Qwiki_M16_efConstruction500_operator8.index";

    delete alg_hnsw;

    alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, new_index_path);

//    int max_elements = alg_hnsw->cur_element_count;
//    for (int i = 0; i < max_elements; ++i) {
//        char *dataPoint = alg_hnsw->getDataByInternalId(i);
//        float norm = normalizeVector((float *) dataPoint, dim);
//    }

//    alg_hnsw->encoder_origin_data_ip2(DCO.get());

    load_pq_mp_binary("/home/zzlin/dataset/wiki/norm_pq_mp.bin", DCO->pq_mp_);
    load_node_cluster_dist_binary("/home/zzlin/dataset/wiki/norm_node_cluster_dist.bin", DCO->node_cluster_dist_);
    std::vector<float> acc, app, thresh, cluster;
    std::unordered_map<size_t, bool> KNNmap;
    std::vector<size_t> id_to_L1;
    size_t feature_dim = 3, model_count = 1;
    feature_dim += model_count;
    std::cerr << "feature dim:: " << feature_dim << " models:: " << model_count << std::endl;

    id_to_L1.resize(alg_hnsw->cur_element_count);
    for (int i = 0; i < id_to_L1.size(); i++) {
        id_to_L1[alg_hnsw->getExternalLabel(i)] = i;
    }

    std::vector<float> query_norms(query_list.size());
    for (int i = 0; i < query_list.size(); ++i) {
        query_norms[i]= normalizeVector(query_list[i].data.data(), dim);
        DCO->query_transform(query_list[i].data.data());
    }
    std::vector<float> tmp_vector(dim);
    float* source_data;
    float magni = 0;
    //// positive samples
    for (int i = 0; i < count_bound; i++) {
        std::vector<VidType> gt = ground_truth_list[i];
        float *q = query_list[i].data.data();
        source_data = (float *) alg_hnsw->getDataByInternalId(id_to_L1[gt[K - 1]]);
        std::copy(source_data, source_data + dim, tmp_vector.begin());
//        magni = normalizeVector(tmp_vector.data(), dim);
//        DCO->query_transform(tmp_vector.data());
        DCO->calc_dist_map(q);
        magni = DCO->base_square_root_[id_to_L1[gt[K - 1]]];
        float thresh_ip = (1 - IPDistance(q, tmp_vector.data(), dim)) * query_norms[i] * magni;
        for (int j = 0; j < K; j++) {
            size_t L1_id = id_to_L1[gt[j]];
            source_data = (float *) alg_hnsw->getDataByInternalId(L1_id);
            std::copy(source_data, source_data + dim, tmp_vector.begin());
            magni = DCO->base_square_root_[L1_id];
            float thresh_dist = 2 - (2 * thresh_ip)/(magni * query_norms[i]);
//            magni = normalizeVector(tmp_vector.data(), dim);
//            DCO->query_transform(tmp_vector.data());
            float acc_dist = L2Distance(q, tmp_vector.data(), dim);
            float app_dist = DCO->naive_product_map_dist(L1_id);
            float cluster_dist = DCO->node_cluster_dist_[L1_id];
            app.push_back(app_dist);
            acc.push_back(acc_dist);
            cluster.push_back(cluster_dist);
            thresh.push_back(thresh_dist);
            KNNmap[L1_id] = true;
        }
    }

    //// negative samples
    for (int i = 0; i < count_bound; i++) {
        std::vector<VidType> gt = ground_truth_list[i];
        float *q = query_list[i].data.data();
        source_data = (float *) alg_hnsw->getDataByInternalId(id_to_L1[gt[K - 1]]);
        std::copy(source_data, source_data + dim, tmp_vector.begin());
//        magni = normalizeVector(tmp_vector.data(), dim);
//        DCO->query_transform(tmp_vector.data());
        DCO->calc_dist_map(q);
        magni = DCO->base_square_root_[id_to_L1[gt[K - 1]]];
        float thresh_ip = (1 - IPDistance(q, tmp_vector.data(), dim)) * query_norms[i] * magni;
        for (auto u : res[i]) {
            size_t L1_id = std::get<0>(u);
            if (KNNmap[L1_id]) continue;
            source_data = (float *) alg_hnsw->getDataByInternalId(L1_id);
            std::copy(source_data, source_data + dim, tmp_vector.begin());
//            magni = normalizeVector(tmp_vector.data(), dim);
//            DCO->query_transform(tmp_vector.data());
            magni = DCO->base_square_root_[L1_id];
            float thresh_dist = 2 - (2 * thresh_ip)/(magni * query_norms[i]);
            float acc_dist = L2Distance(q, tmp_vector.data(), dim);
            float app_dist = DCO->naive_product_map_dist(L1_id);
            float cluster_dist = DCO->node_cluster_dist_[L1_id];
            app.push_back(app_dist);
            acc.push_back(acc_dist);
            cluster.push_back(cluster_dist);
            thresh.push_back(thresh_dist);
        }
        if (acc.size() > elements_bound) break;
    }

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

    delete alg_hnsw;
    return 0;
}