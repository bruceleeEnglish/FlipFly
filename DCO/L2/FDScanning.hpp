#ifndef DCOS_FSCANNING_H
#define DCOS_FSCANNING_H

#include "../DCO.hpp"
#include "../../global.h"
#include "../../include/hnswlib/hnswlib.h"

class FScanning : public DCO {
public:
    FScanning(std::string dataset, size_t dim)
            : space_(dim), dim_(dim) {
        this->distfunc_ = space_.get_dist_func();
        this->dist_func_param_ = space_.get_dist_func_param();
    }


    std::pair<bool, float> dist_compare(const float &threshold, const void *data, const void *query, float res = 0, size_t i = 0) {
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

    float fix_dist_calculate(const void *data, const void *query) {
        return 0;
    }

    void cancel_SIMD() {
        distfunc_ = hnswlib::L2Sqr;
    }

    int mark() {
        return 0;
    }

private:
    hnswlib::L2Space space_;
    size_t dim_;
    hnswlib::DISTFUNC<float> distfunc_;
    void *dist_func_param_{nullptr};
};
#endif //DCOS_FSCANNING_H
