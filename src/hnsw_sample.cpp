#include <iostream>
#include <getopt.h>
#include <fstream>
#include <ctime>
#include <cmath>
#include <queue>
#include <vector>
#include <random>
#include <string>
#include "../FileIO/File_IO.h"

int main(int argc, char **argv) {
    //// parse arguments
    const struct option opts[] = {
            {"help",              no_argument,       0, 'h'},
            {"dataset_path",      required_argument, 0, 'n'},
            {"learning_path",     required_argument, 0, 'l'},
            {"sampling",          required_argument, 0, 's'},
            {"size",              required_argument, 0, 'z'},
    };
    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    std::string dataset_path;
    std::string learning_path;
    int sampling = 10000;
    int size = -1;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "n:l:s:z:", opts, &ind);
        switch (iarg) {
            case 'n':
                if (optarg) dataset_path = optarg;
                break;
            case 'l':
                if (optarg) learning_path = optarg;
                break;
            case 's':
                if (optarg) sampling = atoi(optarg);
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

    std::cout << "dataset_path: " << dataset_path << std::endl;
    std::cout << "learning_path: " << learning_path << std::endl;
    std::cout << "sampling: " << sampling << std::endl;

    //// read data
    std::vector <VectorDataType> data_list;
    ReadVectorData(dataset_path, data_list);

    if (size != -1) {
        data_list.resize(size);
        if (size < sampling) {
            std::cerr << "Error: Sampling size is larger than the dataset size. Adjusting to dataset size." << std::endl;
            sampling = size;
        }
    }

    // Check if sampling is greater than the size of data_list
    if (data_list.size() < static_cast<size_t>(sampling)) {
        std::cerr << "Sampling size is larger than the dataset size. Adjusting to dataset size." << std::endl;
        exit(EXIT_FAILURE);
    }

    //// shuffle
    unsigned seed = static_cast<unsigned>(std::time(nullptr));
    std::mt19937 g(seed);
    std::shuffle(data_list.begin(), data_list.end(), g);
    data_list.resize(sampling);

    //// dump
    DumpVectorData(learning_path, data_list);
    return 0;
}