#ifndef DCOS_IVF_H
#define DCOS_IVF_H

#include <limits>
#include <queue>
#include <vector>
#include <algorithm>
#include <map>
#include <cstring>
#include <fstream>
#include <memory>
#include <cassert>
#include "../../Logger/Logger.h"
#include "../../DCO/DCO.hpp"
#include "../../DCO/L2/DDCres.hpp"
#include "../../DCO/L2/DDCopq.hpp"

class IVF {
public:
    size_t N_;
    size_t D_;
    size_t C_;
    size_t d_; // the dimensionality of first a few dimensions

    float *L1_data_;
    float *res_data_;
    float *centroids_;

    size_t *start_;
    size_t *len_;
    size_t *id_;

    ///////////////
    std::unique_ptr<DCO> dist_compare_operator_{nullptr};
    ///////////////

    IVF();

    void
    build(const std::vector<std::vector<float>> &X, const std::vector<std::vector<float>> &centroids, int adaptive = 0);

    std::priority_queue<std::pair<float, size_t>>
    searchKnn(float *query, size_t k, size_t nprobe, float distK = std::numeric_limits<float>::max()) const;

    std::priority_queue<std::pair<float, size_t>>
    searchKnnIP(float *query, size_t k, size_t nprobe, float distK = std::numeric_limits<float>::max()) const;

    std::priority_queue<std::pair<float, size_t>>
    searchKnnOPQ(float *query, size_t k, size_t nprobe, float distK = std::numeric_limits<float>::max()) const;

    std::priority_queue<std::pair<float, size_t>>
    searchKnnOPQSIMD(float *query, size_t k, size_t nprobe, float distK = std::numeric_limits<float>::max()) const;

    std::vector<std::tuple<unsigned, float, float>> searchKnnLogger(float *query, size_t k, size_t nprobe) const;

    void compute_base_square(bool learned) const;

    void encoder_origin_data(DCO *ddc = nullptr);

    void rotate_data();

    void rebuild();

    ~IVF();

    void save(std::string filename);

    void load(std::string filename);
};

IVF::IVF() {
    N_ = D_ = C_ = d_ = 0;
    start_ = len_ = id_ = NULL;
    L1_data_ = res_data_ = centroids_ = NULL;
}

void
IVF::build(const std::vector<std::vector<float>> &X, const std::vector<std::vector<float>> &centroids, int adaptive) {
    N_ = X.size();
    C_ = centroids.size();
    D_ = X[0].size();

    assert(D_ > 32);
    start_ = new size_t[C_];
    len_ = new size_t[C_];
    id_ = new size_t[N_];

    std::vector<size_t> *temp = new std::vector<size_t>[C_];

    for (int i = 0; i < N_; i++) {
        int belong = 0;
        float dist_min = dist_compare_operator_->dist_calculate(X[i].data(), centroids[0].data());
        for (int j = 1; j < C_; j++) {
            std::pair<bool, float> flag_dist = dist_compare_operator_->dist_compare(dist_min, X[i].data(),
                                                                                    centroids[j].data());
            float dist = flag_dist.second;
            if (flag_dist.first) {
                dist_min = dist;
                belong = j;
            }
        }
        if (i % 50000 == 0) {
            std::cerr << "Processing - " << i << " / " << N_ << std::endl;
        }
        temp[belong].push_back(i);
    }
    std::cerr << "Cluster Generated!" << std::endl;

    size_t sum = 0;
    for (int i = 0; i < C_; i++) {
        len_[i] = temp[i].size();
        start_[i] = sum;
        sum += len_[i];
        for (int j = 0; j < len_[i]; j++) {
            id_[start_[i] + j] = temp[i][j];
        }
    }

    if (adaptive == 1) d_ = init_d;
    else if (adaptive == 0) d_ = D_;
    else d_ = 0;

    L1_data_ = new float[N_ * d_ + 1];
    res_data_ = new float[N_ * (D_ - d_) + 1];
    centroids_ = new float[C_ * D_];

    for (int i = 0; i < N_; i++) {
        int x = id_[i];
        for (int j = 0; j < D_; j++) {
            if (j < d_) L1_data_[i * d_ + j] = X[x][j];
            else res_data_[i * (D_ - d_) + j - d_] = X[x][j];
        }
    }

    for (int i = 0; i < C_; ++i) {
        std::memcpy(centroids_ + i * D_, centroids[i].data(), D_ * sizeof(float));
    }
    delete[] temp;
}

std::priority_queue<std::pair<float, size_t>> IVF::searchKnn(float *query, size_t k, size_t nprobe, float distK) const {
    std::pair<float, size_t> *centroid_dist = new std::pair<float, size_t>[C_];

    for (int i = 0; i < C_; i++) {
#ifdef _STATISTICS_
        HPLogger hp_logger;
#endif
        centroid_dist[i].first = dist_compare_operator_->dist_calculate((void *) query, (void *) (centroids_ + i * D_));
#ifdef _STATISTICS_
        hp_logger.SetEndTimer();
        dist_compare_operator_->add_dist_time(hp_logger.GetDurationTime());
#endif
        centroid_dist[i].second = i;
    }

    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + C_);

    size_t ncan = 0;
    for (int i = 0; i < nprobe; i++)
        ncan += len_[centroid_dist[i].second];
    float *dist = new float[ncan];
    std::pair<float, size_t> *candidates = new std::pair<float, size_t>[ncan];
    int *obj = new int[ncan];

    int cur = 0;
    for (int i = 0; i < nprobe; i++) {
        int cluster_id = centroid_dist[i].second;
        for (int j = 0; j < len_[cluster_id]; j++) {
            size_t can = start_[cluster_id] + j;
#ifdef _STATISTICS_
            HPLogger hp_logger;
#endif
            float tmp_dist = dist_compare_operator_->fix_dist_calculate(query, L1_data_ + can * d_);
#ifdef _STATISTICS_
            hp_logger.SetEndTimer();
            dist_compare_operator_->add_dist_time(hp_logger.GetDurationTime());
#endif
            if (d_ > 0) dist[cur] = tmp_dist;
            else dist[cur] = 0;
            obj[cur] = can;
            cur++;
        }
    }
    std::priority_queue<std::pair<float, size_t>> KNNs;

    if (d_ == D_) {
        for (int i = 0; i < ncan; i++) {
            candidates[i].first = dist[i];
            candidates[i].second = id_[obj[i]];
        }
        std::partial_sort(candidates, candidates + k, candidates + ncan);

        for (int i = 0; i < k; i++) {
            KNNs.emplace(candidates[i].first, candidates[i].second);
        }
    }

    if (d_ < D_) {
        auto cur_dist = dist;
        for (int i = 0; i < nprobe; i++) {
            int cluster_id = centroid_dist[i].second;
            for (int j = 0; j < len_[cluster_id]; j++) {
                size_t can = start_[cluster_id] + j;
#ifdef _STATISTICS_
                HPLogger hp_logger;
#endif
                std::pair<bool, float> flag_dist;
//#ifdef _FIX_DIMENSION_
//                assert(d_ == 0);
//                float fix_dist = dist_compare_operator_->fix_dist_calculate(res_data_ + can * (D_ - d_), query + d_);
//                flag_dist = dist_compare_operator_->dist_compare(distK,
//                                                                 res_data_ + can * (D_ - d_),
//                                                                 query + d_,
//                                                                 fix_dist, init_d);
//#else
                flag_dist = dist_compare_operator_->dist_compare(distK,
                                                                 res_data_ + can * (D_ - d_) - d_,
                                                                 query,
                                                                 *cur_dist, d_); ///**
//#endif
                float tmp_dist = flag_dist.second;
#ifdef _STATISTICS_
                hp_logger.SetEndTimer();
                dist_compare_operator_->add_dist_time(hp_logger.GetDurationTime());
#endif
                if (flag_dist.first) {
                    KNNs.emplace(tmp_dist, id_[can]);
                    if (KNNs.size() > k) KNNs.pop();
                }
                if (KNNs.size() == k && KNNs.top().first < distK) {
                    distK = KNNs.top().first;
                }
                cur_dist++;
            }
        }
    }

    delete[] centroid_dist;
    delete[] dist;
    delete[] candidates;
    delete[] obj;
    return KNNs;
}

std::priority_queue<std::pair<float, size_t>>
IVF::searchKnnIP(float *query, size_t k, size_t nprobe, float distK) const {
    dist_compare_operator_->get_query_square((float *) query);
    std::pair<float, size_t> *centroid_dist = new std::pair<float, size_t>[C_];

    for (int i = 0; i < C_; i++) {
#ifdef _STATISTICS_
        HPLogger hp_logger;
#endif
        centroid_dist[i].first = dist_compare_operator_->dist_calculate((void *) query,
                                                                        (void *) (centroids_ + i * D_));
#ifdef _STATISTICS_
        hp_logger.SetEndTimer();
        dist_compare_operator_->add_dist_time(hp_logger.GetDurationTime());
#endif
        centroid_dist[i].second = i;
    }

    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + C_);

    size_t ncan = 0;
    for (int i = 0; i < nprobe; i++)
        ncan += len_[centroid_dist[i].second];
    float *dist = new float[ncan];
    std::pair<float, size_t> *candidates = new std::pair<float, size_t>[ncan];
    int *obj = new int[ncan];

    int cur = 0;
    for (int i = 0; i < nprobe; i++) {
        int cluster_id = centroid_dist[i].second;
        for (int j = 0; j < len_[cluster_id]; j++) {
            size_t can = start_[cluster_id] + j;
#ifdef _STATISTICS_
            HPLogger hp_logger;
#endif
            float tmp_dist = dist_compare_operator_->get_pre_sum(can) -
                             2 * dist_compare_operator_->fix_dist_calculate(query, L1_data_ + can * d_);
#ifdef _STATISTICS_
            hp_logger.SetEndTimer();
            dist_compare_operator_->add_dist_time(hp_logger.GetDurationTime());
#endif
            dist[cur] = tmp_dist;
            obj[cur] = can;
            cur++;
        }
    }
    std::priority_queue<std::pair<float, size_t>> KNNs;

    auto cur_dist = dist;
    for (int i = 0; i < nprobe; i++) {
        int cluster_id = centroid_dist[i].second;
        for (int j = 0; j < len_[cluster_id]; j++) {
            size_t can = start_[cluster_id] + j;
#ifdef _STATISTICS_
            HPLogger hp_logger;
#endif
            std::pair<bool, float> flag_dist;
//#ifdef _FIX_DIMENSION_
//            assert(d_ == 0);
//            float fix_dist = dist_compare_operator_->fix_dist_calculate(res_data_ + can * (D_ - d_), query + d_);
//            flag_dist = dist_compare_operator_->dist_compare(distK,
//                                                             res_data_ + can * (D_ - d_),
//                                                             query + d_,
//                                                             *cur_dist - 2 * fix_dist, init_d);
//#else
            flag_dist = dist_compare_operator_->dist_compare(distK,
                                                             res_data_ + can * (D_ - d_) - d_,
                                                             query,
                                                             *cur_dist, d_);
//#endif
            float tmp_dist = flag_dist.second;
#ifdef _STATISTICS_
            hp_logger.SetEndTimer();
            dist_compare_operator_->add_dist_time(hp_logger.GetDurationTime());
#endif
            if (flag_dist.first) {
                KNNs.emplace(tmp_dist, id_[can]);
                if (KNNs.size() > k) KNNs.pop();
            }
            if (KNNs.size() == k && KNNs.top().first < distK) {
                distK = KNNs.top().first;
            }
            cur_dist++;
        }
    }

    delete[] centroid_dist;
    delete[] dist;
    delete[] candidates;
    delete[] obj;
    return KNNs;
}

std::priority_queue<std::pair<float, size_t>>
IVF::searchKnnOPQSIMD(float *query, size_t k, size_t nprobe, float distK) const {
    std::pair<float, size_t> *centroid_dist = new std::pair<float, size_t>[C_];

    for (int i = 0; i < C_; i++) {
#ifdef _STATISTICS_
        HPLogger hp_logger;
#endif
        centroid_dist[i].first = dist_compare_operator_->dist_calculate((void *) query, (void *) (centroids_ + i * D_));
#ifdef _STATISTICS_
        hp_logger.SetEndTimer();
        dist_compare_operator_->add_dist_time(hp_logger.GetDurationTime());
#endif
        centroid_dist[i].second = i;
    }

    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + C_);

    auto *ddcopq = dynamic_cast<DDCopq *>(dist_compare_operator_.get());
    ddcopq->calc_dist_map((float *) query);

    std::priority_queue<std::pair<float, size_t>> KNNs;
    float thresh = distK;
    for (int i = 0; i < nprobe; i++) {
        size_t cluster_id = centroid_dist[i].second;
        std::vector<int> ids;
        for (int j = 0; j < len_[cluster_id]; j++) {
            size_t can = start_[cluster_id] + j;
            ids.push_back(can);
        }
        while (ids.size() % 4 != 0) ids.push_back(0);
        auto res = ddcopq->sse4_dist_scan(ids.data(), ids.size());
        for (int j = 0; j < len_[cluster_id]; j++) {
            size_t can = start_[cluster_id] + j;
            if (KNNs.size() < k) {
                float tmp_dist = ddcopq->dist_calculate(res_data_ + can * D_, query);
                KNNs.emplace(tmp_dist, id_[can]);
            } else {
#ifdef _STATISTICS_
                HPLogger hp_logger;
#endif
                if (ddcopq->linear_classifier_default_pq(res[j], ddcopq->node_cluster_dist_[can], thresh)) continue;
#ifdef _STATISTICS_
                hp_logger.SetEndTimer();
                dist_compare_operator_->add_dist_time(hp_logger.GetDurationTime());
#endif
                float tmp_dist = ddcopq->dist_calculate(query, res_data_ + can * D_);
                if (tmp_dist < KNNs.top().first) {
                    KNNs.pop();
                    KNNs.emplace(tmp_dist, id_[can]);
                    thresh = KNNs.top().first;
                }
            }
        }
    }
    delete[] centroid_dist;
    return KNNs;
}

std::priority_queue<std::pair<float, size_t>>
IVF::searchKnnOPQ(float *query, size_t k, size_t nprobe, float distK) const {
    std::pair<float, size_t> *centroid_dist = new std::pair<float, size_t>[C_];

    for (int i = 0; i < C_; i++) {
#ifdef _STATISTICS_
        HPLogger hp_logger;
#endif
        centroid_dist[i].first = dist_compare_operator_->dist_calculate((void *) query, (void *) (centroids_ + i * D_));
#ifdef _STATISTICS_
        hp_logger.SetEndTimer();
        dist_compare_operator_->add_dist_time(hp_logger.GetDurationTime());
#endif
        centroid_dist[i].second = i;
    }

    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + C_);

    auto *ddcopq = dynamic_cast<DDCopq *>(dist_compare_operator_.get());
    ddcopq->calc_dist_map((float *) query);

    std::priority_queue<std::pair<float, size_t>> KNNs, CoarseQ;
    for (int i = 0; i < nprobe; i++) {
        size_t cluster_id = centroid_dist[i].second;
        std::vector<int> ids;
        for (int j = 0; j < len_[cluster_id]; j++) {
            size_t can = start_[cluster_id] + j;
            float tmp_dist = ddcopq->naive_product_map_dist(can);
            CoarseQ.emplace(-tmp_dist, can);
        }
    }

    size_t Topk = k;
    while (Topk--) {
        size_t can = CoarseQ.top().second;
        CoarseQ.pop();
        float tmp_dist = ddcopq->dist_calculate(query, res_data_ + can * D_);
        KNNs.emplace(tmp_dist, id_[can]);
    }
    float thresh = KNNs.top().first;
    while (!CoarseQ.empty()) {
        if (CoarseQ.empty()) break;
        size_t can = CoarseQ.top().second;
        float pq_dist = -CoarseQ.top().first;
        CoarseQ.pop();
        if (ddcopq->linear_classifier_default_pq(pq_dist, ddcopq->node_cluster_dist_[can], thresh))
            continue;
        float tmp_dist = ddcopq->dist_calculate(query, res_data_ + can * D_);
        if (KNNs.size() < k) KNNs.emplace(tmp_dist, id_[can]);
        else if (tmp_dist < KNNs.top().first) {
            KNNs.emplace(tmp_dist, id_[can]);
            if (KNNs.size() > k) KNNs.pop();
            thresh = KNNs.top().first;
        }
    }

    delete[] centroid_dist;
    return KNNs;
}

//std::priority_queue<std::pair<float, size_t>> IVF::searchKnnOPQ(float *query, size_t k, size_t nprobe, float distK) const {
//    std::pair<float, size_t> *centroid_dist = new std::pair<float, size_t>[C_];
//    auto *ddcopq = dynamic_cast<DDCopq *>(dist_compare_operator_.get());
//    for (int i = 0; i < C_; i++) {
//#ifdef _STATISTICS_
//        HPLogger hp_logger;
//#endif
//        centroid_dist[i].first = dist_compare_operator_->dist_calculate((void *) query, (void *) (centroids_ + i * D_));
//#ifdef _STATISTICS_
//        hp_logger.SetEndTimer();
//        dist_compare_operator_->add_dist_time(hp_logger.GetDurationTime());
//#endif
//        centroid_dist[i].second = i;
//    }
//
//    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + C_);
//
//    size_t ncan = 0;
//    for (int i = 0; i < nprobe; i++)
//        ncan += len_[centroid_dist[i].second];
//    float *dist = new float[ncan];
//    std::pair<float, size_t> *candidates = new std::pair<float, size_t>[ncan];
//    int *obj = new int[ncan];
//
//    int cur = 0;
//    for (int i = 0; i < nprobe; i++) {
//        int cluster_id = centroid_dist[i].second;
//        for (int j = 0; j < len_[cluster_id]; j++) {
//            size_t can = start_[cluster_id] + j;
//            float tmp_dist = 0;
//            if (d_ > 0) dist[cur] = tmp_dist;
//            else dist[cur] = 0;
//            obj[cur] = can;
//            cur++;
//        }
//    }
//    std::priority_queue<std::pair<float, size_t>> KNNs;
//
//    if (d_ == D_) {
//        for (int i = 0; i < ncan; i++) {
//            candidates[i].first = dist[i];
//            candidates[i].second = id_[obj[i]];
//        }
//        std::partial_sort(candidates, candidates + k, candidates + ncan);
//
//        for (int i = 0; i < k; i++) {
//            KNNs.emplace(candidates[i].first, candidates[i].second);
//        }
//    }
//
//    if (d_ < D_) {
//        auto cur_dist = dist;
//        for (int i = 0; i < nprobe; i++) {
//            int cluster_id = centroid_dist[i].second;
//            for (int j = 0; j < len_[cluster_id]; j++) {
//                size_t can = start_[cluster_id] + j;
//                float dist1 = ddcopq->naive_product_map_dist(can);
//#ifdef _STATISTICS_
//                HPLogger hp_logger;
//#endif
//                bool flag_consider_candidate = false;
//                float dist0 = 0;
//                if (ddcopq->linear_classifier_default_pq(dist1, ddcopq->node_cluster_dist_[can],
//                                                         distK)){
//                    flag_consider_candidate = false;
//                } else {
//                    dist0 = dist_compare_operator_->dist_calculate(query, res_data_ + can * (D_ - d_) - d_);
//                    if (dist0 < distK) {
//                        flag_consider_candidate = true;
//                    }
//                }
//                float tmp_dist = dist0;
//#ifdef _STATISTICS_
//                hp_logger.SetEndTimer();
//                dist_compare_operator_->add_dist_time(hp_logger.GetDurationTime());
//#endif
//                if (flag_consider_candidate) {
//                    KNNs.emplace(tmp_dist, id_[can]);
//                    if (KNNs.size() > k) KNNs.pop();
//                }
//                if (KNNs.size() == k && KNNs.top().first < distK) {
//                    distK = KNNs.top().first;
//                }
//                cur_dist++;
//            }
//        }
//    }
//
//    delete[] centroid_dist;
//    delete[] dist;
//    delete[] candidates;
//    delete[] obj;
//    return KNNs;
//}

std::vector<std::tuple<unsigned, float, float> > IVF::searchKnnLogger(float *query, size_t k, size_t nprobe) const {
    // the default value of distK is +inf
    std::pair<float, size_t> *centroid_dist = new std::pair<float, size_t>[C_];
    std::vector<std::tuple<unsigned, float, float> > search_logger;
    // Find out the closest N_{probe} centroids to the query vector.
    for (int i = 0; i < C_; i++) {
        centroid_dist[i].first = dist_compare_operator_->dist_calculate(query, centroids_ + i * D_);
        centroid_dist[i].second = i;
    }
    // Find out the closest N_{probe} centroids to the query vector.
    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + C_);

    size_t ncan = 0;
    for (int i = 0; i < nprobe; i++)
        ncan += len_[centroid_dist[i].second];
    // Scan a few initial dimensions and store the distances.
    // For IVF (i.e., apply FDScanning), it should be D.
    std::priority_queue<float> res_queue;
    for (int i = 0; i < nprobe; i++) {
        int cluster_id = centroid_dist[i].second;
        for (int j = 0; j < len_[cluster_id]; j++) {
            size_t can = start_[cluster_id] + j;
            float tmp_dist;
            if (d_ == D_)
                tmp_dist = dist_compare_operator_->dist_calculate(query, L1_data_ + can * d_);
            else
                tmp_dist = dist_compare_operator_->dist_calculate(query, res_data_ + can * D_);
            if (res_queue.size() < k) res_queue.push(tmp_dist);
            else if (tmp_dist < res_queue.top()) {
                res_queue.push(tmp_dist);
                if (res_queue.size() > k) res_queue.pop();
            }
            if (res_queue.size() == k)
                search_logger.emplace_back(can, tmp_dist, res_queue.top());
        }
    }
    delete[] centroid_dist;
    return search_logger;
}

void IVF::compute_base_square(bool learned) const {
    auto *ddcres = dynamic_cast<DDCres *>(dist_compare_operator_.get());
    if (!ddcres) {
        std::cerr << "Error: Invalid DCO." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    ddcres->base_square_.resize(N_);
    float *extra = ddcres->extra_mean_.data();
    float *tmp_L1 = L1_data_, *tmp_res = res_data_;
    for (int i = 0; i < N_; i++) {
        float square = 0.0;
        for (int j = 0; j < d_; j++) {
            if (!learned) tmp_L1[j] -= extra[j];
            square += tmp_L1[j] * tmp_L1[j];
        }
        for (int j = 0; j < D_ - d_; j++) {
            if (!learned) tmp_res[j] -= extra[j + d_];
            square += tmp_res[j] * tmp_res[j];
        }
        ddcres->base_square_[i] = square;
        tmp_L1 += d_;
        tmp_res += (D_ - d_);
    }
    if (!learned) {
        float *tmp_centroid = centroids_;
        for (int i = 0; i < C_; i++) {
            for (int j = 0; j < D_; j++) {
                tmp_centroid[j] -= extra[j];
            }
            tmp_centroid += D_;
        }
    }
}

void IVF::encoder_origin_data(DCO *ddc) {
    if (d_ != 0) {
        std::cerr << "Error: Invalid DCO." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    DDCopq *ddcopq = nullptr;
    if (ddc) {
        ddcopq = dynamic_cast<DDCopq *>(ddc);
    } else {
        ddcopq = dynamic_cast<DDCopq *>(dist_compare_operator_.get());
    }
    if (!ddcopq) {
        std::cerr << "Error: Invalid DCO." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    ddcopq->pq_mp_.resize(N_ * ddcopq->sub_vector_);
    ddcopq->node_cluster_dist_.resize(N_);
    double ave_dist = 0.0;
    for (int i = 0; i < N_; i++) {
        float dist_to_centroid = 0.0;
        for (int j = 0; j < ddcopq->sub_vector_; j++) {
            uint8_t belong = 0;
            float dist = ddcopq->distfunc_sub_(res_data_ + i * D_ + j * ddcopq->sub_dim_, ddcopq->pq_book_[j][0].data(),
                                               ddcopq->dist_func_param_sub_);
            for (int k = 1; k < ddcopq->sub_cluster_count_; k++) {
                float new_dist = ddcopq->distfunc_sub_(res_data_ + i * D_ + j * ddcopq->sub_dim_,
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
        if (i % 50000 == 0) std::cerr << "Encoder progress: " << i << " / " << N_ << std::endl;
    }
    std::cerr << "Encoder ave dist:: " << ave_dist / N_ << std::endl;
}

void IVF::rotate_data() {
    float *tmp = new float[D_];
    for (int i = 0; i < N_; ++i) {
        std::memcpy(tmp, L1_data_ + i * d_, d_ * sizeof(float));
        std::memcpy(tmp + d_, res_data_ + i * (D_ - d_), (D_ - d_) * sizeof(float));
        dist_compare_operator_->query_transform(tmp);
        std::memcpy(L1_data_ + i * d_, tmp, d_ * sizeof(float));
        std::memcpy(res_data_ + i * (D_ - d_), tmp + d_, (D_ - d_) * sizeof(float));
        if (i % 5000 == 0) std::cerr << "Rotate " << i << " / " << N_ << std::endl;
    }
    for (int i = 0; i < C_; ++i) {
        std::memcpy(tmp, centroids_ + i * D_, D_ * sizeof(float));
        dist_compare_operator_->query_transform(tmp);
        std::memcpy(centroids_ + i * D_, tmp, D_ * sizeof(float));
    }
    delete[] tmp;
}

void IVF::save(std::string filename) {
    std::ofstream output(filename, std::ios::binary);

    output.write((char *) &N_, sizeof(size_t));
    output.write((char *) &D_, sizeof(size_t));
    output.write((char *) &C_, sizeof(size_t));
    output.write((char *) &d_, sizeof(size_t));

    if (d_ > 0) output.write((char *) L1_data_, N_ * d_ * sizeof(float));
    if (d_ < D_) output.write((char *) res_data_, N_ * (D_ - d_) * sizeof(float));
    output.write((char *) centroids_, C_ * D_ * sizeof(float));

    output.write((char *) start_, C_ * sizeof(size_t));
    output.write((char *) len_, C_ * sizeof(size_t));
    output.write((char *) id_, N_ * sizeof(size_t));

    output.close();
}

void IVF::load(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::cerr << filename << std::endl;

    if (!input.is_open())
        throw std::runtime_error("Cannot open file");

    input.read((char *) &N_, sizeof(size_t));
    input.read((char *) &D_, sizeof(size_t));
    input.read((char *) &C_, sizeof(size_t));
    input.read((char *) &d_, sizeof(size_t));
    std::cerr << N_ << " " << D_ << " " << C_ << " " << d_ << std::endl;

    L1_data_ = new float[N_ * d_ + 10];
    res_data_ = new float[N_ * (D_ - d_) + 10];
    centroids_ = new float[C_ * D_];

    start_ = new size_t[C_];
    len_ = new size_t[C_];
    id_ = new size_t[N_];

    if (d_ > 0)input.read((char *) L1_data_, N_ * d_ * sizeof(float));
    if (d_ < D_)input.read((char *) res_data_, N_ * (D_ - d_) * sizeof(float));
    input.read((char *) centroids_, C_ * D_ * sizeof(float));

    input.read((char *) start_, C_ * sizeof(size_t));
    input.read((char *) len_, C_ * sizeof(size_t));
    input.read((char *) id_, N_ * sizeof(size_t));

    input.close();
}

void IVF::rebuild() {
    float *tmp_L1_data = L1_data_;
    float *tmp_res_data = res_data_;

    d_ = init_d;

    L1_data_ = new float[N_ * d_ + 10];
    res_data_ = new float[N_ * (D_ - d_) + 10];

    for (size_t i = 0; i < N_; ++i) {
        std::memcpy(L1_data_ + i * d_, tmp_res_data + i * D_, d_ * sizeof(float));
        std::memcpy(res_data_ + i * (D_ - d_), tmp_res_data + i * D_ + d_, (D_ - d_) * sizeof(float));
    }

    delete[] tmp_res_data;
    delete[] tmp_L1_data;
}

IVF::~IVF() {
    if (id_ != NULL)delete[] id_;
    if (len_ != NULL)delete[] len_;
    if (start_ != NULL)delete[] start_;
    if (L1_data_ != NULL)delete[] L1_data_;
    if (res_data_ != NULL)delete[] res_data_;
    if (centroids_ != NULL)delete[] centroids_;
}

#endif //DCOS_IVF_H
