#include <iostream>
#include <getopt.h>
#include <fstream>
#include <ctime>
#include <cmath>
#include <queue>
#include "../include/hnswlib/hnswlib.h"
#include "../FileIO/File_IO.h"
#include "../include/ivf/ivf.h"
#include "../utils.h"

int main(int argc, char *argv[]) {
    //// parse arguments
    const struct option opts[] = {
            {"help",                no_argument,       0, 'h'},
            {"dataset_path",        required_argument, 0, 'n'},
            {"query_path",          required_argument, 0, 'q'},
            {"index_path",          required_argument, 0, 'i'},
            {"index_new_path",      required_argument, 0, 'e'},
            {"operator",            required_argument, 0, 'o'},
    };
    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    std::string dataset_path;
    std::string query_path;
    std::string index_path;
    std::string index_new_path;
    int op = 0;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "n:q:i:e:o:", opts, &ind);
        switch (iarg) {
            case 'n':
                if (optarg) dataset_path = optarg;
                break;
            case 'q':
                if (optarg) query_path = optarg;
                break;
            case 'i':
                if (optarg) index_path = optarg;
                break;
            case 'e':
                if (optarg) index_new_path = optarg;
                break;
            case 'o':
                if (optarg) op = std::stoi(optarg);
                break;
        }
    }

    //// check arguments
    if (dataset_path.empty()) {
        std::cerr << "Error: dataset_path is empty. Please provide a valid path using the -n option." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (query_path.empty()) {
        std::cerr << "Error: query_path is empty. Please provide a valid path using the -q option." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (index_path.empty()) {
        std::cerr << "Error: index_path is empty. Please provide a valid path using the -i option." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (index_new_path.empty()) {
        std::cerr << "Error: index_new_path is empty. Please provide a valid path using the -e option." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "dataset_path: " << dataset_path << std::endl;
    std::cout << "query_path: " << query_path << std::endl;
    std::cout << "index_path: " << index_path << std::endl;
    std::cout << "index_new_path: " << index_new_path << std::endl;
    std::cout << "operator: " << op << std::endl;

    //// read query and groundtruth
    std::vector<VectorDataType> query_list;
    std::vector<size_t> query_k_list;
    ReadVectorQuery(query_path, query_list, query_k_list);

    //// load index
    int dim = query_list[0].Dimension();
    IVF ivf;
    ivf.load(index_path.c_str());
    ivf.dist_compare_operator_ = setup_DCO(dataset_path, dim, op, "");

    ivf.rotate_data();
    ivf.save(index_new_path);
    return 0;
}
