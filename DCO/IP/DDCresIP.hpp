#ifndef DCOS_DDCRESIP_H
#define DCOS_DDCRESIP_H
#pragma once

#include "../DCO.hpp"
#include "../../global.h"
#include "../../FileIO/File_IO.h"
#include "../../include/hnswlib/hnswlib.h"

class DDCresIP : public DCO {
public:
    DDCresIP(std::string dataset, size_t dim, std::string type = "100")
            : space_(dim),
              space_ip_(dim),
              dim_(dim),
              dim_delta_(delta_d),
              dim_residual_((dim - init_d) % delta_d),
              flag_residual_(dim_residual_ > 0),
              sigma_(ddc_sigma) {

        distfunc_ = space_ip_.get_dist_func();
        distfunc_delta_ = space_ip_.get_dist_func_by_dim(dim_delta_);
        distfunc_residual_ = space_ip_.get_dist_func_by_dim(dim_residual_);
        dist_func_param_ = space_.get_dist_func_param();
        dist_func_param_delta_ = &dim_delta_;
        dist_func_param_residual_ = &dim_residual_;
        learned_ = true;
        load_transform_material(dataset, type);
        dimension_scanning.resize(dim_ + 1, 0);
    }

    void load_transform_material(const std::string &dataset_path, std::string type) {
        size_t last_slash_pos = dataset_path.find_last_of('/');
        std::string directory;
        if (last_slash_pos != std::string::npos) {
            directory = dataset_path.substr(0, last_slash_pos);
        } else {
            std::cerr << "Error: Invalid file path." << std::endl;
            std::exit(EXIT_FAILURE);
        }
        std::string transformation_path = directory + "/ddc/pca_matrix_" + type + "_ip.fvecs";
        if (learned_) {
            transformation_path = directory + "/ddc/pca_matrix.fvecs";
        }
        std::vector<VectorDataType> Orthogonal;
        ReadVectorData(transformation_path, Orthogonal);
        size_t hybrid, origin_dim;
        hybrid = Orthogonal.size();
        origin_dim = hybrid - 3;
        mean_ = Orthogonal[0].data;
        extra_mean_ = Orthogonal[1].data;
        var_ = Orthogonal[2].data;
        pre_query_.clear();
        pre_query_.resize(dim_ + 1);
        transform_matrix_.resize(static_cast<Eigen::Index>(dim_), static_cast<Eigen::Index>(dim_));
        for (int i = 0; i < dim_; i++)
            for (int j = 0; j < dim_; j++)
                transform_matrix_(i, j) = Orthogonal[i + 3].data[j];
        mean_square_ = 0;
        for (int i = 0; i < dim_; i++)
            mean_square_ += mean_[i] * mean_[i];
        mean_trans_.resize(dim_);
        Eigen::MatrixXf mean_matrix(1, dim_);
        Eigen::MatrixXf mean_trans_matrix(1, dim_);
        for (int j = 0; j < dim_; ++j) {
            mean_matrix(0, j) = mean_[j];
        }
        mean_trans_matrix = mean_matrix * transform_matrix_;
        for (int j = 0; j < dim_; ++j) {
            mean_trans_[j] = mean_trans_matrix(0, j);
        }
    }

    void query_transform(const void *query, size_t n = 1) {
        float *q = (float *) query;
        Eigen::MatrixXf query_matrix(n, dim_);
        Eigen::MatrixXf res(n, dim_);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < dim_; ++j) {
                query_matrix(i, j) = q[i * dim_ + j] - mean_[j];
            }
        }
        res = query_matrix * transform_matrix_;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < dim_; ++j) {
                if (!learned_)
                    q[i * dim_ + j] = res(i, j) - extra_mean_[j];
                else
                    q[i * dim_ + j] = res(i, j);
            }
        }
    }

    bool uniform_inference(float &res, const float &threshold, int &tag_dim) const {
        if (res - pre_query_[tag_dim] >= threshold) return true;
        else return false;
    }

    std::pair<bool, float>
    dist_compare(const float &threshold, const void *data, const void *query, float res = 0, size_t tag_dim = 0) {
#ifdef _STATISTICS_
        this->tot_dist_calculation++;
#endif
        int cur = tag_dim;
        float *q = (float *) query;
        float *d = (float *) data;
//        if (cur != 0) {
//            if (uniform_inference(res, threshold, cur)) {
//#ifdef _STATISTICS_
//                this->tot_dimension += tag_dim;
//#endif
//                return std::make_pair(false, res);
//            }
//        }
        int fix_dim = dim_ - dim_residual_;
        while (cur < fix_dim) {
            res -= distfunc_delta_((void *) d, (void *) q, dist_func_param_delta_);
            d += dim_delta_;
            q += dim_delta_;
            cur += dim_delta_;
            if (uniform_inference(res, threshold, cur)) {
#ifdef _STATISTICS_
                this->tot_dimension += cur;
                dimension_scanning[cur]++;
#endif
                return std::make_pair(false, res);
            }
        }
        if (flag_residual_) {
            res -= distfunc_residual_((void *) d, (void *) q, dist_func_param_residual_);
        }
#ifdef _STATISTICS_
        this->tot_dimension += dim_;
        dimension_scanning[dim_]++;
#endif
        if (res >= threshold) return std::make_pair(false, res);
        else return std::make_pair(true, res);
    }

    float dist_calculate(const void *data, const void *query, size_t data_id) {
#ifdef _STATISTICS_
        this->tot_dist_calculation++;
        this->tot_full_dist++;
        this->tot_dimension += dim_;
#endif
        return this->distfunc_(data, query, dist_func_param_) - query_mean_ - MPD_[data_id];
    }

    float get_pre_sum(size_t id) const {
        return 1 - query_mean_ - MPD_[id];
    }

    void cancel_SIMD() {
        distfunc_ = hnswlib::InnerProductDistance;
        distfunc_delta_ = hnswlib::InnerProduct;
        distfunc_residual_ = hnswlib::InnerProduct;
    }

    void get_query_square(const float *q) {
        query_square_ = 0;
        query_mean_ = mean_square_;
        for (int i = 0; i < dim_; i++) {
            query_square_ += q[i] * q[i];
            query_mean_ += q[i] * mean_trans_[i];
            pre_query_[i] = q[i] * q[i] * var_[i];
        }
        pre_query_[dim_] = 0;
        for (int i = (int) dim_ - 1; i >= 0; i--) {
            pre_query_[i] += pre_query_[i + 1];
        }
        for (int i = 0; i < dim_; i++) {
            pre_query_[i] = sqrt(pre_query_[i]);
            pre_query_[i] *= sigma_;
        }
    }


    int mark() {
        return 13;
    }

    size_t dim_;
    std::vector<float> mean_;
    std::vector<float> var_;
    std::vector<float> extra_mean_;
    std::vector<float> pre_query_;
    std::vector<float> base_square_;
    std::vector<float> mean_trans_;
    float query_square_{0};
    float query_mean_{0};
    float mean_square_{0};
    std::vector<float> MPD_;

private:
    hnswlib::L2Space space_;
    hnswlib::InnerProductSpace space_ip_;
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
    float sigma_ = 8.0;
    bool learned_{false};
};

#endif //DCOS_DDCRESIP_H
