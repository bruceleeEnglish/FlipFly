// new_add: this entire file is a new addition (not present in upstream dco-benchmark)

//
// SRHT-ADSampling: Subsampled Randomized Hadamard Transform variant of ADSampling.
//
// Replaces the O(D^2) Haar QR rotation with the O(D log D) SRHT:
//     S = (1/sqrt(D)) * H_D * D_rad
// where:
//     H_D    : Walsh-Hadamard matrix, D = 2^k
//     D_rad  : diagonal matrix of independent +/-1 entries (Rademacher)
//
// Theoretical equivalence with Original ADSampling:
//   - Both produce orthogonal rotations (S^T S = I)
//   - For any vector x, each component (S x)_i has variance ||x||^2 / D
//   - Hence the partial-distance estimator has the SAME ||o-q||^4/d variance
//     scaling, so we reuse ADSampling's `ratio` function unchanged.
//
// Data files used:
//   <dataset_dir>/srht/S.fvecs           single-row containing the +/-1 diagonal
//   <dataset_dir>/S<dataset>_base.fvecs  data already rotated by the same SRHT
//
// File naming follows the convention used by ADSampling (O.fvecs, O*_base.fvecs).
//

#ifndef DCOS_SRHT_ADSAMPLING_H
#define DCOS_SRHT_ADSAMPLING_H

#include "../DCO.hpp"
#include "../../global.h"
#include "../../FileIO/File_IO.h"
#include "../../include/hnswlib/hnswlib.h"

class SRHTADSampling : public DCO {
public:
    SRHTADSampling(std::string dataset, size_t dim)
            : space_(dim),
              dim_(dim),
              dim_delta_(delta_d),
              dim_residual_((dim - init_d) % delta_d),
              flag_residual_(dim_residual_ > 0),
              epsilon0_(epsilon0) {

        // D must be a power of 2 for the in-place butterfly FWHT
        if ((dim_ & (dim_ - 1)) != 0) {
            std::cerr << "SRHTADSampling: dim=" << dim_
                      << " must be a power of 2 (pad your data first)." << std::endl;
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
        scratch_.resize(dim_);
    }

    /// Loads the +/-1 diagonal stored as a single fvecs row.
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
            std::cerr << "SRHTADSampling: expected one row of length D in S.fvecs"
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }
        rad_.assign(dim_, 0.0f);
        for (size_t i = 0; i < dim_; ++i) {
            rad_[i] = Diagonal[0].data[i];
        }
    }

    /// Threshold ratio for the hypothesis test — IDENTICAL to ADSampling's
    /// because the variance structure of the two estimators matches.
    float ratio(const size_t &D, const size_t &i) const {
        if (i == D) return 1.0f;
        return 1.0f * i / D
               * (1.0f + epsilon0_ / std::sqrt(static_cast<float>(i)))
               * (1.0f + epsilon0_ / std::sqrt(static_cast<float>(i)));
    }

    /// Same partial-scan / early-rejection logic as ADSampling.
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

    /// Apply SRHT in-place to each query: q <- (1/sqrt(D)) * H_D * D_rad * q
    void query_transform(const void *query, size_t n = 1) {
        float *q = (float *) query;
        for (size_t row = 0; row < n; ++row) {
            float *x = q + row * dim_;
            // Step 1: D_rad * x
            for (size_t i = 0; i < dim_; ++i) x[i] *= rad_[i];
            // Step 2: H * (...)  via in-place butterfly FWHT
            fwht_inplace(x, dim_);
            // Step 3: scale by 1/sqrt(D)
            for (size_t i = 0; i < dim_; ++i) x[i] *= scale_;
        }
    }

    void cancel_SIMD() {
        distfunc_ = hnswlib::L2Sqr;
        distfunc_delta_ = hnswlib::L2Sqr;
        distfunc_residual_ = hnswlib::L2Sqr;
        distfunc_init_ = hnswlib::L2Sqr;
    }

    int mark() {
        return 18;   // new operator id; see utils.h dispatcher
    }

private:
    /// Iterative butterfly Fast Walsh-Hadamard Transform, in-place.
    /// Length D must be a power of 2. Result is the unnormalized H_D * x;
    /// the caller multiplies by 1/sqrt(D) for orthogonality.
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

    std::vector<float> rad_;          // +/-1 diagonal of D_rad (length D)
    float scale_{1.0f};               // 1 / sqrt(D)
    std::vector<float> scratch_;      // unused; reserved for future SIMD path

    size_t dim_init_;
    hnswlib::DISTFUNC<float> distfunc_init_;
    void *dist_func_param_init_{nullptr};
};

#endif // DCOS_SRHT_ADSAMPLING_H
