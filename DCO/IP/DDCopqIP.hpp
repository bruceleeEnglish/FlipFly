#ifndef DCOS_DDCOPQIP_H
#define DCOS_DDCOPQIP_H

#include <bits/types.h>
#include <Eigen/Dense>
#include "../DCO.hpp"
#include "../../global.h"
#include "../../FileIO/File_IO.h"
#include "../../include/hnswlib/hnswlib.h"

class DDCopqIP : public DCO {
public:
    DDCopqIP(std::string dataset, size_t dim, std::string type = "hnsw")
            : DDCopqIP(dataset, dim, true, type) {
    }

    DDCopqIP(std::string dataset, size_t dim, bool load_model, std::string type = "hnsw")
            : space_(dim),
              dim_(dim) {

        distfunc_ = space_.get_dist_func();
        dist_func_param_ = space_.get_dist_func_param();
        res_.resize(128);
        load_transform_material(dataset, load_model, type);
    }

    void load_transform_material(const std::string &dataset_path, bool load_model, std::string type) {
        size_t last_slash_pos = dataset_path.find_last_of('/');
        std::string directory;
        if (last_slash_pos != std::string::npos) {
            directory = dataset_path.substr(0, last_slash_pos);
        } else {
            std::cerr << "Error: Invalid file path." << std::endl;
            std::exit(EXIT_FAILURE);
        }
        std::string codebook_path = directory + "/ddc/codebook.centroid";
        load_product_codebook(codebook_path);
        distfunc_sub_ = space_.get_dist_func_by_dim(sub_dim_);
        dist_func_param_sub_ = &sub_dim_;
        std::string transformation_path = directory + "/ddc/opq_matrix.fvecs";
        std::vector<VectorDataType> Orthogonal;
        ReadVectorData(transformation_path, Orthogonal);
        size_t origin_dim = Orthogonal.size(), project_dim = Orthogonal[0].Dimension();
        project_dim_ = project_dim;
        if (origin_dim != dim_) {
            std::cerr << "Error: Invalid dimension." << std::endl;
            std::exit(EXIT_FAILURE);
        }
        transform_matrix_.resize(static_cast<Eigen::Index>(origin_dim), static_cast<Eigen::Index>(project_dim));
        for (int i = 0; i < origin_dim; i++) {
            for (int j = 0; j < project_dim; j++) {
                transform_matrix_(i, j) = Orthogonal[i].data[j];
            }
        }
        if (load_model) {
            std::string model_name = "linear_" + type + "_opq_ip" + ".txt";
            std::string linear_path = directory + "/ddc/" + model_name;
            load_model_param(linear_path);
        }
    }

    void load_product_codebook(std::string filename) {
        std::ifstream in(filename, std::ios::binary);
        in.read((char *) &sub_vector_, sizeof(unsigned));
        in.read((char *) &sub_cluster_count_, sizeof(unsigned));
        in.read((char *) &sub_dim_, sizeof(unsigned));
        std::cerr << "sub vec:: " << sub_vector_ << " sub cluster:: " << sub_cluster_count_ << " sub dim:: "
                  << sub_dim_ << std::endl;
        pq_book_.resize(sub_vector_);
        dist_mp_.resize(sub_vector_ * sub_cluster_count_);
        for (int i = 0; i < sub_vector_; i++) {
            pq_book_[i].resize(sub_cluster_count_);
            for (int j = 0; j < sub_cluster_count_; j++) {
                pq_book_[i][j].resize(sub_dim_);
                in.read((char *) pq_book_[i][j].data(), sizeof(float) * sub_dim_);
            }
        }
    }

    void load_model_param(std::string model_path) {
        std::ifstream fin(model_path);
        if (!fin.good()) return;
        size_t num;
        fin >> num;
        model_count_ = num;
        W_.resize(2);
        B_.resize(1);
        fin >> W_[0] >> W_[1] >> B_[0];
        fin.close();
    }

    void calc_dist_map(float *query) {
        for (unsigned i = 0; i < sub_vector_; i++) {
            for (unsigned j = 0; j < sub_cluster_count_; j++) {
                dist_mp_[i * sub_cluster_count_ + j] = distfunc_sub_(query + i * sub_dim_, &pq_book_[i][j][0],
                                                                     dist_func_param_sub_);
            }
        }
    }

    void binary_search_multi_linear_ip(unsigned num, const float *app_dist, const float *acc_dist, const float *thresh,
                                    const float *cluster_dist, size_t count_base, double recall, float eps = 1e-5) {
        double l = 0.0, r = 0.0, res;
        for (int i = 0; i < num; i++) {
            if (thresh[i] > r) r = thresh[i];
            if (thresh[i] < l) l = thresh[i];
        }
        if (l == 0.0 && r == 0.0) {
            l = -1.0;
            r = 1.0;
        } else {
            l *= 1.01;
            r *= 1.01;
        }
        std::cerr << l << " " << r << " " << W_[0] << " " << W_[1] << std::endl;
        while (r - l > eps) {
            double mid = (l + r) / 2.0;
            unsigned bad_count = 0;
            for (int i = 0; i < num; i++) {
                if (app_dist[i] * W_[0] + cluster_dist[i] * W_[1] + mid > thresh[i] &&
                    (double) acc_dist[i] < (double) thresh[i] + 1e-6) {
                    bad_count++;
                }
            }
            double test_recall = (double) ((double) count_base - (double) bad_count) / (double) count_base;
            if (test_recall < recall) {
                r = mid - eps;
            } else {
                std::cerr << mid << " <-gap-> " << r << " recall::" << test_recall << " bad-> " << bad_count
                          << std::endl;
                res = mid;
                l = mid + eps;
            }
        }
        B_[0] = (float) res;
    }

    float naive_product_map_dist(size_t id) const {
        float res = 0;
        for (int i = 0; i < sub_vector_; i++) {
            res += dist_mp_[i * sub_cluster_count_ + pq_mp_[id * sub_vector_ + i]];
        }
        return 1 - res;
    }

    bool linear_classifier_default_pq(float app_dist, float cluster_dist, float thresh_dist) {
//#ifdef _STATISTICS_
//        this->tot_dimension += sub_vector_;
//#endif
        return app_dist * W_[0] + cluster_dist * W_[1] + B_[0] >= thresh_dist;
    }

    float *sse4_dist_scan(const int *id, size_t num) {
        if (num > res_.size()) res_.resize(num + 1);
        auto res = res_.data();
#ifdef USE_SSE
        float arr[4];
        for (int i = 0; i < num; i += 4) {
            __m128 candidate_dist;
            const uint8_t *const pqcode0 = pq_mp_.data() + id[i] * sub_vector_;
            const uint8_t *const pqcode1 = pq_mp_.data() + id[i + 1] * sub_vector_;
            const uint8_t *const pqcode2 = pq_mp_.data() + id[i + 2] * sub_vector_;
            const uint8_t *const pqcode3 = pq_mp_.data() + id[i + 3] * sub_vector_;
            sse4_product_map_dist(pqcode0, pqcode1, pqcode2, pqcode3, dist_mp_.data(), candidate_dist);
            _mm_store_ps(arr, candidate_dist);
            std::memcpy(res + i, arr, 16);
        }
        for (int i = 0; i < num; ++i) {
            res[i] = 1 - res[i];
        }
#endif
        return res;
    }

    void sse4_product_map_dist(const uint8_t *const pqcode0,
                               const uint8_t *const &pqcode1,
                               const uint8_t *const &pqcode2,
                               const uint8_t *const &pqcode3,
                               float *dists,
                               __m128 &candidates
    ) const {
        candidates = _mm_set_ps(
                dists[pqcode3[0]],
                dists[pqcode2[0]],
                dists[pqcode1[0]],
                dists[pqcode0[0]]
        );
        // Such perf critical loop. Pls unroll
        for (size_t j = 1; j < sub_vector_; ++j) {
            const float *const cdist = dists + j * sub_cluster_count_;
            __m128 partial = _mm_set_ps(
                    cdist[pqcode3[j]],
                    cdist[pqcode2[j]],
                    cdist[pqcode1[j]],
                    cdist[pqcode0[j]]
            );
            candidates = _mm_add_ps(candidates, partial);
        }
    }

    void query_transform(const void *query, size_t n = 1) {
        float *q = (float *) query;
        Eigen::MatrixXf query_matrix(n, dim_);
        Eigen::MatrixXf res(n, project_dim_);
        for (int i = 0; i < n; i++) {
            std::memcpy(query_matrix.row(i).data(), &q[i * dim_], dim_ * sizeof(float));
        }
        res = query_matrix * transform_matrix_;
        for (int i = 0; i < n; i++) {
            std::memcpy(&q[i * project_dim_], res.row(i).data(), project_dim_ * sizeof(float));
        }
    }

    std::pair<bool, float>
    dist_compare(const float &threshold, const void *data, const void *query, float res = 0, size_t i = 0) {
#ifdef _STATISTICS_
        this->tot_dist_calculation++;
#endif
        res = this->distfunc_(data, query, dist_func_param_);
#ifdef _STATISTICS_
        this->tot_dimension += dim_;
#endif
        if (res >= threshold) return std::make_pair(false, res);
        return std::make_pair(true, res);
    }

    float dist_calculate(const void *data, const void *query) {
#ifdef _STATISTICS_
        this->tot_dist_calculation++;
        this->tot_full_dist++;
        this->tot_dimension += dim_;
#endif
        return this->distfunc_(data, query, dist_func_param_);
    }

    void cancel_SIMD() {
        distfunc_ = hnswlib::InnerProductDistance;
        SIMD_ = false;
    }

    int mark() {
        return 15;
    }

    std::vector<uint8_t> pq_mp_;
    std::vector<float> node_cluster_dist_;
    size_t sub_dim_{0}, sub_vector_{0}, sub_cluster_count_{0};
    hnswlib::DISTFUNC<float> distfunc_sub_;
    void *dist_func_param_sub_{nullptr};
    std::vector<std::vector<std::vector<float>>> pq_book_;
    std::vector<float> dist_mp_;
    size_t project_dim_;
    size_t model_count_ = 0;
    std::vector<float> W_, B_, b_;
    std::vector<float> res_;
    bool SIMD_{true};

private:
    hnswlib::InnerProductSpace space_;
    size_t dim_;
    hnswlib::DISTFUNC<float> distfunc_;
    void *dist_func_param_{nullptr};
    Eigen::MatrixXf transform_matrix_;
};

#endif //DCOS_DDCOPQIP_H
