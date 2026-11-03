// new_add: this entire file is a new addition (not present in upstream dco-benchmark)

//
// SRHT-ADSampling (Inner Product variant). Same logic as ADSamplingIP but
// using SRHT (FWHT + Rademacher diagonal) instead of a Haar QR rotation.
//
// Variance structure of the partial-distance estimator is identical to
// the Haar version (see SRHTADSampling.hpp for derivation), so the IP
// hypothesis test (`ratio` and threshold) is reused unchanged.
//

#ifndef DCOS_SRHT_ADSAMPLING_IP_H
#define DCOS_SRHT_ADSAMPLING_IP_H
#pragma once

#include "../DCO.hpp"
#include "../../global.h"
#include "../../FileIO/File_IO.h"
#include "../../include/hnswlib/hnswlib.h"

class SRHTADSamplingIP : public DCO {
public:
    SRHTADSamplingIP(std::string dataset, size_t dim)
            : space_(dim),
              dim_(dim),
              space_ip_(dim),
              dim_delta_(delta_d),
              dim_residual_((dim - init_d) % delta_d),
              flag_residual_(dim_residual_ > 0),
              epsilon0_(epsilon0) {

        if ((dim_ & (dim_ - 1)) != 0) {
            std::cerr << "SRHTADSamplingIP: dim=" << dim_
                      << " must be a power of 2 (pad your data first)." << std::endl;
            std::exit(EXIT_FAILURE);
        }

        distfunc_ = space_ip_.get_dist_func_by_dim(dim_);
        distfunc_delta_ = space_.get_dist_func_by_dim(dim_delta_);
        distfunc_residual_ = space_.get_dist_func_by_dim(dim_residual_);
        dist_func_param_ = &dim_;
        dist_func_param_delta_ = &dim_delta_;
        dist_func_param_residual_ = &dim_residual_;
        load_transform_material(dataset);
        dimension_scanning.resize(dim_ + 1, 0);

        scale_ = 1.0f / std::sqrt(static_cast<float>(dim_));
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
        std::string transformation_path = directory + "/srht/S.fvecs";
        std::vector<VectorDataType> Diagonal;
        ReadVectorData(transformation_path, Diagonal);
        if (Diagonal.size() != 1 || Diagonal[0].data.size() != dim_) {
            std::cerr << "SRHTADSamplingIP: expected one row of length D in S.fvecs"
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }
        rad_.assign(dim_, 0.0f);
        for (size_t i = 0; i < dim_; ++i) rad_[i] = Diagonal[0].data[i];

        std::string DSSR_path = directory + "/DSSR.fvecs";
        std::vector<VectorDataType> DSSR;
        ReadVectorData(DSSR_path, DSSR);
        base_square_root_.resize(DSSR[0].Dimension());
        for (int i = 0; i < DSSR[0].Dimension(); ++i) {
            base_square_root_[i] = DSSR[0].data[i];
        }
    }

    float ratio(const size_t &D, const size_t &i) const {
        if (i == D) return 1.0f;
        return 1.0f * i / D
               * (1.0f + epsilon0_ / std::sqrt(static_cast<float>(i)))
               * (1.0f + epsilon0_ / std::sqrt(static_cast<float>(i)));
    }

    std::pair<bool, float>
    dist_compare(const float &threshold, const void *data, const void *query,
                 float res = 0, size_t data_id = 0) {
#ifdef _STATISTICS_
        this->tot_dist_calculation++;
#endif
        size_t i = 0;
        float data_square_root_ = query_square_root_ * base_square_root_[data_id];
        float ip_threshold = 2 - 2 * (1 - threshold) / data_square_root_;
        float *q = (float *) query;
        float *d = (float *) data;
        while (i + dim_delta_ <= dim_) {
            res += distfunc_delta_((void *) (d + i), (void *) (q + i),
                                   dist_func_param_delta_);
            i += dim_delta_;
            if (res >= ip_threshold * ratio(dim_, i)) {
#ifdef _STATISTICS_
                this->tot_dimension += i;
                dimension_scanning[i]++;
#endif
                return std::make_pair(false, res);
            }
        }
#ifdef _STATISTICS_
        this->tot_dimension += dim_;
        dimension_scanning[dim_]++;
#endif
        if (flag_residual_) {
            res += distfunc_residual_((void *) (d + i), (void *) (q + i),
                                      dist_func_param_residual_);
            if (res >= ip_threshold) return std::make_pair(false, res);
        }
        return std::make_pair(true, 1 - (1 - res / 2) * data_square_root_);
    }

    float dist_calculate(const void *data, const void *query, size_t data_id) {
#ifdef _STATISTICS_
        this->tot_dist_calculation++;
        this->tot_full_dist++;
        this->tot_dimension += dim_;
#endif
        return 1 - this->distfunc_(data, query, dist_func_param_)
                   * query_square_root_ * base_square_root_[data_id];
    }

    /// Apply SRHT to query in-place, like the L2 variant.
    void query_transform(const void *query, size_t n = 1) {
        float *q = (float *) query;
        for (size_t row = 0; row < n; ++row) {
            float *x = q + row * dim_;
            for (size_t i = 0; i < dim_; ++i) x[i] *= rad_[i];
            fwht_inplace(x, dim_);
            for (size_t i = 0; i < dim_; ++i) x[i] *= scale_;
        }
    }

    void get_query_square(const float *q) {
        query_square_root_ = sqrt(distfunc_(q, q, dist_func_param_));
        float *temp_q = new float[dim_];
        for (int i = 0; i < dim_; ++i) {
            temp_q[i] = q[i] / query_square_root_;
        }
        std::memcpy(const_cast<float *>(q), temp_q, dim_ * sizeof(float));
        delete[] temp_q;
    }

    void cancel_SIMD() {
        distfunc_ = hnswlib::InnerProduct;
        distfunc_delta_ = hnswlib::L2Sqr;
        distfunc_residual_ = hnswlib::L2Sqr;
    }

    int mark() {
        return 19;   // new IP operator id
    }

    std::vector<float> base_square_root_;
    float query_square_root_{0};

private:
    static void fwht_inplace(float *x, size_t D) {
        for (size_t h = 1; h < D; h <<= 1) {
            for (size_t i = 0; i < D; i += (h << 1)) {
                for (size_t j = i; j < i + h; ++j) {
                    float a = x[j];
                    float b = x[j + h];
                    x[j]     = a + b;
                    x[j + h] = a - b;
                }
            }
        }
    }

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
    float epsilon0_{2.1f};

    std::vector<float> rad_;
    float scale_{1.0f};
};

#endif // DCOS_SRHT_ADSAMPLING_IP_H
