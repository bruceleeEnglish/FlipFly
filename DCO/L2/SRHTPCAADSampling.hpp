// new_add: this entire file is a new addition (not present in upstream dco-benchmark)

//
// SRHT + PCA-permutation ADSampling (operator id 20).
//
// Identical to SRHTADSampling (FWHT-based rotation, ADSampling estimator),
// PLUS one extra O(D) step: reorder the rotated coordinates by an offline
// permutation pi that puts the highest-variance SRHT dimensions first.
//
// Why "PCA permutation" but not "PCA rotation":
//   - We compute Var(y_i) for each SRHT-rotated dim i across the whole dataset
//     and take the descending sort as pi. That's an O(D) array index lookup
//     in online code — NOT a full D x D matrix multiply.
//   - This captures the "scan most-informative dims first" half of DDCres's
//     PCA. The other half (full decorrelating rotation) costs O(D^2) per query
//     and would defeat the FWHT speed advantage, so we skip it.
//
// Data files used:
//   <dataset_dir>/srht_pca/S.fvecs       2 rows of length D:
//                                          row 0 = +/-1 Rademacher diagonal
//                                          row 1 = pi as float (cast to int32)
//   <dataset_dir>/T<dataset>_base.fvecs  data already SRHT-rotated AND permuted
//
// The hypothesis test (`ratio()`) is identical to ADSampling/SRHT — variance
// structure of the partial-distance estimator is unchanged (still 2*||o-q||^4/d).
// Permutation only changes WHICH d dims we look at first.
//

#ifndef DCOS_SRHT_PCA_ADSAMPLING_H
#define DCOS_SRHT_PCA_ADSAMPLING_H

#include "../DCO.hpp"
#include "../../global.h"
#include "../../FileIO/File_IO.h"
#include "../../include/hnswlib/hnswlib.h"

class SRHTPCAADSampling : public DCO {
public:
    SRHTPCAADSampling(std::string dataset, size_t dim)
            : space_(dim),
              dim_(dim),
              dim_delta_(delta_d),
              dim_residual_((dim - init_d) % delta_d),
              flag_residual_(dim_residual_ > 0),
              epsilon0_(epsilon0) {

        if ((dim_ & (dim_ - 1)) != 0) {
            std::cerr << "SRHTPCAADSampling: dim=" << dim_
                      << " must be a power of 2." << std::endl;
            std::exit(EXIT_FAILURE);
        }

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

        scale_ = 1.0f / std::sqrt(static_cast<float>(dim_));
        scratch_.assign(dim_, 0.0f);
    }

    void load_transform_material(std::string &dataset_path) {
        size_t last_slash_pos = dataset_path.find_last_of('/');
        std::string directory = (last_slash_pos != std::string::npos)
                                ? dataset_path.substr(0, last_slash_pos) : "";
        if (directory.empty()) {
            std::cerr << "Error: Invalid file path." << std::endl;
            std::exit(EXIT_FAILURE);
        }
        std::string transformation_path = directory + "/srht_pca/S.fvecs";
        std::vector<VectorDataType> Diagonal;
        ReadVectorData(transformation_path, Diagonal);
        if (Diagonal.size() != 2 || Diagonal[0].data.size() != dim_
                || Diagonal[1].data.size() != dim_) {
            std::cerr << "SRHTPCAADSampling: expected 2 rows of length D in "
                      << transformation_path << std::endl;
            std::exit(EXIT_FAILURE);
        }
        rad_.assign(dim_, 0.0f);
        perm_.assign(dim_, 0);
        for (size_t i = 0; i < dim_; ++i) {
            rad_[i]  = Diagonal[0].data[i];
            perm_[i] = static_cast<int32_t>(Diagonal[1].data[i] + 0.5f);
            if (perm_[i] < 0 || (size_t)perm_[i] >= dim_) {
                std::cerr << "SRHTPCAADSampling: bad perm index " << perm_[i]
                          << " at i=" << i << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
    }

    /// Same `ratio()` as ADSampling / SRHTADSampling — variance structure unchanged.
    float ratio(const size_t &D, const size_t &i) const {
        if (i == D) return 1.0f;
        return 1.0f * i / D
               * (1.0f + epsilon0_ / std::sqrt(static_cast<float>(i)))
               * (1.0f + epsilon0_ / std::sqrt(static_cast<float>(i)));
    }

    /// Same dist_compare scan loop as ADSampling.
    std::pair<bool, float> dist_compare(const float &threshold, const void *data,
                                        const void *query, float res = 0,
                                        size_t i = 0) {
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
            res += distfunc_delta_((void *) (d + i), (void *) (q + i),
                                   dist_func_param_delta_);
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
            res += distfunc_residual_((void *) (d + i), (void *) (q + i),
                                      dist_func_param_residual_);
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

    /// Apply (1/sqrt(D)) * H * D_rad to the query, then reorder via pi.
    void query_transform(const void *query, size_t n = 1) {
        float *q = (float *) query;
        for (size_t row = 0; row < n; ++row) {
            float *x = q + row * dim_;
            // Step 1: SRHT
            for (size_t i = 0; i < dim_; ++i) x[i] *= rad_[i];
            fwht_inplace(x, dim_);
            for (size_t i = 0; i < dim_; ++i) x[i] *= scale_;
            // Step 2: O(D) permutation
            for (size_t i = 0; i < dim_; ++i) scratch_[i] = x[perm_[i]];
            std::memcpy(x, scratch_.data(), dim_ * sizeof(float));
        }
    }

    void cancel_SIMD() {
        distfunc_ = hnswlib::L2Sqr;
        distfunc_delta_ = hnswlib::L2Sqr;
        distfunc_residual_ = hnswlib::L2Sqr;
        distfunc_init_ = hnswlib::L2Sqr;
    }

    int mark() {
        return 20;   // SRHT + PCA-permutation operator id
    }

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

    std::vector<float>   rad_;     // +/-1 diagonal
    std::vector<int32_t> perm_;    // permutation pi
    std::vector<float>   scratch_; // O(D) buffer for permutation

    float scale_{1.0f};

    size_t dim_init_;
    hnswlib::DISTFUNC<float> distfunc_init_;
    void *dist_func_param_init_{nullptr};
};

#endif // DCOS_SRHT_PCA_ADSAMPLING_H
