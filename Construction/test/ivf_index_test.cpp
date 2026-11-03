#include <iostream>
#include <getopt.h>
#include <fstream>
#include <ctime>
#include <cmath>
#include <queue>
#include "../../utils.h"
#include "../../include/ivf/ivf.h"

int main() {
    //// parse arguments
    std::string dataset_path = "/home/zzlin/dataset/YouTube/YouTube.bin";
    std::string centroid_path = "/home/zzlin/dataset/YouTube/YouTube_centroid_4096.fvecs";
    std::string index_path = "/home/zzlin/dataset/YouTube/ivf_YouTube_C4096_operator0.index";
    int op = 0;

    std::cout << "dataset_path: " << dataset_path << std::endl;
    std::cout << "index_path: " << index_path << std::endl;
    std::cout << "centroid_path: " << centroid_path << std::endl;
    std::cout << "operator: " << op << std::endl;

    //// read data
    Logger logger;
    std::vector<VectorDataType> data_list;
    ReadVectorData(dataset_path, data_list);

    std::vector<std::vector<float>> data_arr;
    for (auto &data: data_list) {
        data_arr.push_back(data.data);
    }

    data_list.clear();
    data_list.shrink_to_fit();

    std::vector<VectorDataType> centroid_list;
    ReadVectorData(centroid_path, centroid_list);

    std::vector<std::vector<float>> centroid_arr;
    for (auto &centroid: centroid_list) {
        centroid_arr.push_back(centroid.data);
    }

    centroid_list.clear();
    centroid_list.shrink_to_fit();

    size_t dim = centroid_arr[0].size();

    //// build index
    IVF ivf;
    ivf.dist_compare_operator_ = setup_DCO(dataset_path, dim, op, "");
    ivf.build(data_arr, centroid_arr, 2);
    ivf.save(index_path);
    return 0;
}
