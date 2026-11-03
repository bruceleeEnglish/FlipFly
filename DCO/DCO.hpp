#include<string>
#include "../Logger/Logger.h"

#ifndef DCOS_DCO_H
#define DCOS_DCO_H

class DCO {
public:

    DCO() {
        clear();
    }

    virtual std::pair<bool, float>
    dist_compare(const float &threshold, const void *data, const void *query, float res = 0, size_t i = 0) {
        std::cout << "DCO does not support dist compare" << std::endl;
        return std::make_pair(false, res);
    }

    virtual float dist_calculate(const void *data, const void *query) {
        std::cout << "DCO does not support dist calculate with two parameters" << std::endl;
        return 0;
    }

    virtual float dist_calculate(const void *data, const void *query, size_t data_id) {
        std::cout << "DCO does not support dist calculate with three parameters" << std::endl;
        return 0;
    }

    virtual void query_transform(const void *query, size_t n = 1) {};


    virtual int mark() {
        return -1;
    }

    virtual void clear() {
        this->distance_time = 0;
        this->tot_dimension = 0;
        this->tot_dist_calculation = 0;
        this->tot_full_dist = 0;
        this->predict_time = 0;
        this->predict_success = 0;
    }

    virtual void cancel_SIMD() {}

    virtual void get_query_square(const float *q) {
        std::cout << "DCO does not support query square" << std::endl;
    }

    virtual float get_pre_sum(size_t id) const {
        std::cout << "DCO does not support get_pre_sum" << std::endl;
        return 0;
    }

    virtual float fix_dist_calculate(const void *data, const void *query) {
        std::cout << "DCO does not support fix dist calculate" << std::endl;
        return 0;
    }

    void add_dist_time(double time) {
        this->distance_time += time;
    }

    void add_rota_time(double time) {
        this->rotation_time += time;
    }

    void add_predict_success() {
        this->predict_success += 1;
    }

    void add_predict_time() {
        this->predict_time += 1;
    }

    void print_statistics() {
        std::cout << "distance_time: " << distance_time << " ms" << std::endl;
        std::cout << "rotation_time: " << rotation_time << " ms" << std::endl;
        std::cout << "tot_dimension: " << tot_dimension << std::endl;
        std::cout << "tot_dist_calculation: " << tot_dist_calculation << std::endl;
        std::cout << "tot_full_dist: " << tot_full_dist << std::endl;
        std::cout << "predict_success: " << predict_success << std::endl;
        std::cout << "predict_time: " << predict_time << " ms" << std::endl;

        for (int i = 0; i < dimension_scanning.size(); ++i) {
            if (dimension_scanning[i] > 0) {
                std::cout << "\"" << i << "\" : " << dimension_scanning[i] << " , "<< std::endl;
            }
        }
    }

protected:
    double distance_time{0};
    double rotation_time{0};
    double tot_dimension{0};
    double tot_dist_calculation{0};
    double tot_full_dist{0};
    double predict_success{0};
    double predict_time{0};
    std::vector<double> dimension_scanning;
};

#endif //DCOS_DCO_H
