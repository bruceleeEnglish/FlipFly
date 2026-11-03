#ifndef DCOS_ADSAMPLING_H
#define DCOS_ADSAMPLING_H

#include "../DCO.hpp"
#include "../../global.h"
#include "../../FileIO/File_IO.h"
#include "../../include/hnswlib/hnswlib.h"

class ADSampling : public DCO {
public:
    ADSampling(std::string dataset, size_t dim)
            : space_(dim),
              dim_(dim),
              dim_delta_(delta_d),
              dim_residual_((dim - init_d) % delta_d),
              flag_residual_(dim_residual_ > 0),
              epsilon0_(epsilon0){

        distfunc_ = space_.get_dist_func();
        distfunc_delta_ = space_.get_dist_func_by_dim(dim_delta_);
        distfunc_residual_ = space_.get_dist_func_by_dim(dim_residual_);
        dist_func_param_ = space_.get_dist_func_param();
        dist_func_param_delta_ = &dim_delta_;
        dist_func_param_residual_ = &dim_residual_;
        load_transform_material(dataset);

        dim_init_ = init_d;
        distfunc_init_ = space_.get_dist_func_by_dim(dim_init_);
        dist_func_param_init_ = &dim_init_;
        dimension_scanning.resize(dim_ + 1, 0);
    }

    void load_transform_material(std::string &dataset_path) {
        size_t last_slash_pos = dataset_path.find_last_of('/');
        std::string directory;
        if (last_slash_pos != std::string::npos) {
            directory = dataset_path.substr(0, last_slash_pos);
        } else {
            std::cerr << "Error: Invalid file path." << std::endl;
            std::exit(EXIT_FAILURE);
        }
        std::string transformation_path = directory + "/adsampling/O.fvecs";
        std::vector<VectorDataType> Orthogonal;
        ReadVectorData(transformation_path, Orthogonal);
        transform_matrix_.resize(static_cast<Eigen::Index>(dim_), static_cast<Eigen::Index>(dim_));
        for (int i = 0; i < dim_; i++)
            for (int j = 0; j < dim_; j++)
                transform_matrix_(i, j) = Orthogonal[i].data[j];
    }

    float ratio(const size_t &D, const size_t &i) const {
        if (i == D)return 1.0;
        return 1.0 * i / D * (1.0 + epsilon0_ / std::sqrt(i)) * (1.0 + epsilon0_ / std::sqrt(i));
    }

    std::pair<bool, float> dist_compare(const float &threshold, const void *data, const void *query, float res = 0, size_t i = 0) {
#ifdef _STATISTICS_
        this->tot_dist_calculation++;
#endif
        if (i && res >= threshold * ratio(dim_, i)) {
#ifdef _STATISTICS_
            this->tot_dimension += i;
            dimension_scanning[i]++;
#endif
            return std::make_pair(false, res * dim_ / i);
        }
        float *q = (float *) query;
        float *d = (float *) data;
        while (i + dim_delta_ <= dim_) {
            res += distfunc_delta_((void *) (d + i), (void *) (q + i), dist_func_param_delta_);
            i += dim_delta_;
            if (res >= threshold * ratio(dim_, i)) {
#ifdef _STATISTICS_
                this->tot_dimension += i;
                dimension_scanning[i]++;
#endif
                return std::make_pair(false, res * dim_ / i);
            }
        }
#ifdef _STATISTICS_
        this->tot_dimension += dim_;
        dimension_scanning[dim_]++;
#endif
        if (flag_residual_) {
            res += distfunc_residual_((void *) (d + i), (void *) (q + i), dist_func_param_residual_);
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

    float fix_dist_calculate(const void *data, const void *query) {
        return this->distfunc_init_(data, query, dist_func_param_init_);
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

    void cancel_SIMD() {
        distfunc_ = hnswlib::L2Sqr;
        distfunc_delta_ = hnswlib::L2Sqr;
        distfunc_residual_ = hnswlib::L2Sqr;
        distfunc_init_ = hnswlib::L2Sqr;
    }

    int mark() {
        return 3;
    }

private:
    hnswlib::L2Space space_;
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
    float epsilon0_{2.1};
    Eigen::MatrixXf transform_matrix_;

    size_t dim_init_;
    hnswlib::DISTFUNC<float> distfunc_init_;
    void *dist_func_param_init_{nullptr};
};

#endif //DCOS_ADSAMPLING_H
