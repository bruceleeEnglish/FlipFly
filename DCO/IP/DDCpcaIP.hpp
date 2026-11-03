#ifndef DCOS_DDCPCAIP_H
#define DCOS_DDCPCAIP_H
#pragma once

#include "../DCO.hpp"
#include "../../global.h"
#include "../../FileIO/File_IO.h"
#include "../../include/hnswlib/hnswlib.h"

class DDCpcaIP : public DCO {
public:
    DDCpcaIP(std::string dataset, size_t dim, std::string type = "hnsw")
            : DDCpcaIP(dataset, dim, true, type) {
    }

    DDCpcaIP(std::string dataset, size_t dim, bool load_model, std::string type = "hnsw")
            : space_(dim),
              dim_(dim),
              space_ip_(dim),
              dim_delta_(delta_d),
              dim_residual_((dim - init_d) % delta_d),
              flag_residual_(dim_residual_ > 0) {

        distfunc_ = space_ip_.get_dist_func();
        distfunc_delta_ = space_ip_.get_dist_func_by_dim(dim_delta_);
        distfunc_residual_ = space_ip_.get_dist_func_by_dim(dim_residual_);
        dist_func_param_ = space_ip_.get_dist_func_param();
        dist_func_param_delta_ = &dim_delta_;
        dist_func_param_residual_ = &dim_residual_;
        load_transform_material(dataset, load_model, type);
        dimension_scanning.resize(dim_ + 1, 0);
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
        std::string transformation_path = directory + "/dade/P.fvecs";
        std::vector<VectorDataType> Orthogonal;
        ReadVectorData(transformation_path, Orthogonal);
        transform_matrix_.resize(static_cast<Eigen::Index>(dim_), static_cast<Eigen::Index>(dim_));
        for (int i = 0; i < dim_; i++)
            for (int j = 0; j < dim_; j++)
                transform_matrix_(i, j) = Orthogonal[i].data[j];
        if (load_model) {
            std::string model_name = "linear_" + type + "_pca_ip" + ".txt";
            std::string linear_path = directory + "/ddc/" + model_name;
            load_model_param(linear_path);
        }
    }

    void load_model_param(std::string model_path) {
        std::ifstream fin(model_path);
        if (!fin.good()) return;
        size_t num;
        fin >> num;
        W_.resize(num);
        B_.resize(num);
        b_.resize(num);
        model_count_ = num;
        for (int i = 0; i < num; i++) {
            fin >> W_[i] >> B_[i] >> b_[i];
        }

        if (!flag_residual_) {
            W_[num - 1] = 1.0;
            B_[num - 1] = 0.0;
            b_[num - 1] = 0.0;
        }

        std::cerr << " fix dim:: " << dim_ - dim_residual_ << " res dim:: " << dim_residual_ << " linear count:: "
                  << num
                  << std::endl;
        fin.close();
    }

    void binary_search_single_linear_ip(size_t num, const float *app_dist, const float *acc_dist, const float *thresh,
                                     size_t id, size_t count_base, double recall, float eps = 1e-5,
                                     bool verbose = true) {
        double l = 0.0, r = 0.0, res;
        for (int i = 0; i < num; i++) {
            double boundary_candidate = thresh[i] * W_[id] * 1.01;
            if (boundary_candidate > r) r = boundary_candidate;
            if (boundary_candidate < l) l = boundary_candidate;
        }
        if (l == 0.0 && r == 0.0) {
            l = -1.0;
            r = 1.0;
        } else {
            l *= 1.01;
            r *= 1.01;
        }
        if (verbose)
            std::cerr << l << " <-left right-> " << r << std::endl;
        while (r - l > eps) {
            double mid = (l + r) / 2.0;
            size_t bad_count = 0;
            for (int i = 0; i < num; i++) {
                if (app_dist[i] * W_[id] + mid > thresh[i] && (double) acc_dist[i] < (double) thresh[i] + 1e-6) {
                    bad_count++;
                }
            }
            bad_count = std::min(bad_count, count_base);
            double test_recall = (double) ((double) count_base - (double) bad_count) / (double) count_base;
            if (test_recall < recall) {
                r = mid - eps;
            } else {
                if (verbose)
                    std::cerr << mid << " <-gap-> " << r << " recall::" << test_recall << " bad-> " << bad_count
                              << std::endl;
                res = mid;
                l = mid + eps;
            }
        }
        B_[id] = (float) res;
    }


    void query_transform(const void *query, size_t n = 1) {
        float *q = (float *) query;
        Eigen::MatrixXf query_matrix(n, dim_);
        Eigen::MatrixXf res(n, dim_);
        for (int i = 0; i < n; i++) {
            std::memcpy(query_matrix.row(i).data(), &q[i * dim_], dim_ * sizeof(float));
        }
        res = query_matrix * transform_matrix_;
        for (int i = 0; i < n; i++) {
            std::memcpy(&q[i * dim_], res.row(i).data(), dim_ * sizeof(float));
        }
    }

    bool learned_inference(const float &res, const float &threshold, size_t tag_model) const {
        return res * W_[tag_model] + B_[tag_model] >= threshold;
    }

    std::pair<bool, float>
    dist_compare(const float &threshold, const void *data, const void *query, float res = 0, size_t i = 0) {
#ifdef _STATISTICS_
        this->tot_dist_calculation++;
#endif
        res = 1;
        size_t tag_model = i / dim_delta_;
        size_t cur = i;
        float *q = (float *) query;
        float *d = (float *) data;
        if (tag_model != 0) {
            if (learned_inference(res, threshold, tag_model - 1)) {
#ifdef _STATISTICS_
                this->tot_dimension += dim_delta_ * tag_model;
                dimension_scanning[dim_delta_ * tag_model]++;
#endif
                return std::make_pair(false, res * W_[0] + b_[0]);
            }
            d += dim_init_;
            q += dim_init_;
        }
        size_t fix_dim = dim_ - dim_residual_;
        for (; cur < fix_dim; cur += dim_delta_) {
            res -= distfunc_delta_((void *) d, (void *) q, dist_func_param_delta_);
            d += dim_delta_;
            q += dim_delta_;
            if (learned_inference(res, threshold, tag_model)) {
#ifdef _STATISTICS_
                this->tot_dimension += dim_delta_ * (tag_model + 1);
                dimension_scanning[dim_delta_ * (tag_model + 1)]++;
#endif
                return std::make_pair(false, res * W_[tag_model] + b_[tag_model]);
            }
            tag_model++;
        }
#ifdef _STATISTICS_
        this->tot_dimension += dim_;
        dimension_scanning[dim_]++;
#endif
        if (flag_residual_) {
            res -= distfunc_residual_((void *) d, (void *) q, dist_func_param_residual_);
            if (res >= threshold) return std::make_pair(false, res);
        }
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
        distfunc_delta_ = hnswlib::InnerProduct;
        distfunc_residual_ = hnswlib::InnerProduct;
    }

    int mark() {
        return 14;
    }

    size_t model_count_ = 0;
    std::vector<float> W_, B_, b_;

private:
    hnswlib::L2Space space_;
    hnswlib::InnerProductSpace space_ip_;
    size_t dim_;
    size_t dim_delta_;
    size_t dim_residual_;
    hnswlib::DISTFUNC<float> distfunc_;
    hnswlib::DISTFUNC<float> distfunc_delta_;
    hnswlib::DISTFUNC<float> distfunc_residual_;
    void *dist_func_param_{nullptr};
    void *dist_func_param_delta_{nullptr};
    void *dist_func_param_residual_{nullptr};
    bool flag_residual_ = false;
    Eigen::MatrixXf transform_matrix_;
    size_t dim_init_{0};
};

#endif //DCOS_DDCPCAIP_H
