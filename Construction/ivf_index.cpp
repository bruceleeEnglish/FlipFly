#include <iostream>
#include <getopt.h>
#include <fstream>
#include <ctime>
#include <cmath>
#include <queue>
#include "../utils.h"
#include "../Logger/Logger.h"
#include "../include/ivf/ivf.h"

int main(int argc, char *argv[]) {
    //// parse arguments
    const struct option opts[] = {
            {"help",                no_argument,       0, 'h'},
            {"dataset_path",        required_argument, 0, 'n'},
            {"centroid_path",       required_argument, 0, 'c'},
            {"index_path",          required_argument, 0, 'i'},
            {"operator",            required_argument, 0, 'o'},
    };
    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    std::string dataset_path;
    std::string centroid_path;
    std::string index_path;
    int op = 0;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "n:c:i:o:", opts, &ind);
        switch (iarg) {
            case 'n':
                if (optarg) dataset_path = optarg;
                break;
            case 'c':
                if (optarg) centroid_path = optarg;
                break;
            case 'i':
                if (optarg) index_path = optarg;
                break;
            case 'o':
                if (optarg) op = atoi(optarg);
                break;
        }
    }

    //// check arguments
    if (dataset_path.empty()) {
        std::cerr << "Error: dataset_path is empty. Please provide a valid path using the -n option." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (centroid_path.empty()) {
        std::cerr << "Error: centroid_path is empty. Please provide a valid path using the -c option." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (index_path.empty()) {
        std::cerr << "Error: index_path is empty. Please provide a valid path using the -i option." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "dataset_path: " << dataset_path << std::endl;
    std::cout << "index_path: " << index_path << std::endl;
    std::cout << "centroid_path: " << centroid_path << std::endl;
    std::cout << "operator: " << op << std::endl;

    //// read data
    std::vector <VectorDataType> data_list;
    ReadVectorData(dataset_path, data_list);

    std::vector<std::vector<float>> data_arr;
    for (auto & data : data_list) {
        data_arr.push_back(data.data);
    }

    data_list.clear();
    data_list.shrink_to_fit();

    std::vector <VectorDataType> centroid_list;
    ReadVectorData(centroid_path, centroid_list);

    std::vector<std::vector<float>> centroid_arr;
    for (auto & centroid : centroid_list) {
        centroid_arr.push_back(centroid.data);
    }

    centroid_list.clear();
    centroid_list.shrink_to_fit();

    size_t dim = centroid_arr[0].size();

    //// build index
    Logger logger;
    IVF ivf;
    ivf.dist_compare_operator_ = setup_DCO(dataset_path, dim, op, "");
    ivf.build(data_arr, centroid_arr, 2);
    ivf.save(index_path);
    logger.SetEndTimer();
    std::cout << "IVF build Time: " << logger.GetDurationTime() << "s" << std::endl;
    return 0;
}
