#define _IP2L2_
#include <iostream>
#include <getopt.h>
#include <fstream>
#include <ctime>
#include <cmath>
#include <queue>
#include "../utils.h"
#include "../Logger/Logger.h"
#include "../FileIO/File_IO.h"
#include "../FileIO/DataType.hpp"
#include "../include/hnswlib/hnswlib.h"

int main(int argc, char *argv[]) {
    //// parse arguments
    const struct option opts[] = {
            {"help",                no_argument,       0, 'h'},
            {"dataset_path",        required_argument, 0, 'n'},
            {"index_path",          required_argument, 0, 'i'},
            {"M",                   required_argument, 0, 'm'},
            {"efConstruction",      required_argument, 0, 'e'},
            {"operator",            required_argument, 0, 'o'},
            {"size",                required_argument, 0, 'z'},
    };
    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    std::string dataset_path;
    std::string index_path;
    int M = 16;
    int efConstruction = 500;
    int op = 0;
    int size = -1;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "n:i:m:e:o:z:", opts, &ind);
        switch (iarg) {
            case 'n':
                if (optarg) dataset_path = optarg;
                break;
            case 'i':
                if (optarg) index_path = optarg;
                break;
            case 'm':
                if (optarg) M = atoi(optarg);
                break;
            case 'e':
                if (optarg) efConstruction = atoi(optarg);
                break;
            case 'o':
                if (optarg) op = atoi(optarg);
                break;
            case 'z':
                if (optarg) size = atoi(optarg);
                break;
        }
    }

    //// check arguments
    if (dataset_path.empty()) {
        std::cerr << "Error: dataset_path is empty. Please provide a valid path using the -n option." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (index_path.empty()) {
        std::cerr << "Error: index_path is empty. Please provide a valid path using the -i option." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "dataset_path: " << dataset_path << std::endl;
    std::cout << "index_path: " << index_path << std::endl;
    std::cout << "M: " << M << std::endl;
    std::cout << "efConstruction: " << efConstruction << std::endl;
    std::cout << "operator: " << op << std::endl;

    //// read data
    Logger logger;
    std::vector <VectorDataType> data_list;
    ReadVectorData(dataset_path, data_list);

    //// build index
    logger.SetStartTimer();
    int dim = data_list[0].Dimension();
    int N = data_list.size();
    if (size > 0) {
        std::cout << "size: " << size << std::endl;
        data_list.resize(size);
    } else {
        size = N;
    }
    hnswlib::HierarchicalNSW<float> *alg_hnsw;
    if (op < 8) {
        hnswlib::L2Space space(dim);
        alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, N, M, efConstruction);
    } else {
        hnswlib::InnerProductSpace space(dim);
        alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, N, M, efConstruction);
    }
    alg_hnsw->dist_compare_operator_ = setup_DCO(dataset_path, dim, op, "");
    if (op == 5) {
        auto *ddcres = dynamic_cast<DDCres *>(alg_hnsw->dist_compare_operator_.get());
        if (!ddcres) {
            std::cerr << "Error: Invalid DCO." << std::endl;
            std::exit(EXIT_FAILURE);
        }
        ddcres->base_square_.resize(size);
        float *extra = ddcres->extra_mean_.data();
        for (int i = 0; i < size; i++) {
            float square = 0.0;
            auto *base_data = data_list[i].data.data();
            for (int j = 0; j < ddcres->dim_; j++) {
                square += base_data[j] * base_data[j];
            }
            ddcres->base_square_[i] = square;
            if (i % 50000 == 0)
                std::cerr << "Compute base square progress: " << i << " / " << size
                          << std::endl;
        }
    }
    size_t report = 50000;
    for (int i = 0; i < size; i++) {
        alg_hnsw->addPoint(data_list[i].data.data(), data_list[i].vid);
        if (i % report == 0) {
            std::cerr << "Processing - " << i << " / " << size << std::endl;
        }
    }
    logger.SetEndTimer();
    double duration_time = logger.GetDurationTime();
    std::cout << "build index: " << duration_time << " [ms] "<< std::endl;

    //// save index
    alg_hnsw->saveIndex(index_path);
    delete alg_hnsw;
    return 0;
}