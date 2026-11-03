#include <iostream>
#include <getopt.h>
#include <fstream>
#include <ctime>
#include <cmath>
#include <queue>
#include "../include/hnswlib/hnswlib.h"
#include "../FileIO/File_IO.h"
#include "../utils.h"

int main(int argc, char *argv[]) {
    //// parse arguments
    const struct option opts[] = {
            {"help",           no_argument,       0, 'h'},
            {"dataset_path",   required_argument, 0, 'n'},
            {"query_path",     required_argument, 0, 'q'},
            {"index_path",     required_argument, 0, 'i'},
            {"index_new_path", required_argument, 0, 'e'},
            {"operator",       required_argument, 0, 'o'},
            {"metric",         required_argument, 0, 'm'},
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
                if (optarg) op = atoi(optarg);
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

    if (index_path == index_new_path) {
        std::cerr
                << "Error: index_path and index_new_path are the same. Please provide a different path using the -e option."
                << std::endl;
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
    hnswlib::HierarchicalNSW<float> *alg_hnsw;
    if (op < 8) {
        hnswlib::L2Space space(dim);
        alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, index_path);
    } else {
        hnswlib::InnerProductSpace space_ip(dim);
        alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space_ip, index_path);
    }
    alg_hnsw->dist_compare_operator_ = setup_DCO(dataset_path, dim, op, "");
    int max_elements = alg_hnsw->cur_element_count;
    std::vector<float> mean_product_data;
    mean_product_data.resize(max_elements);
    for (int i = 0; i < max_elements; ++i) {
        char *dataPoint = alg_hnsw->getDataByInternalId(i);
        alg_hnsw->dist_compare_operator_->query_transform((void *) dataPoint);
        if (i % 50000 == 0) std::cout << "progress: " << i << " / " << max_elements << std::endl;
    }

    //// save index
    alg_hnsw->saveIndex(index_new_path);
    delete alg_hnsw;
    return 0;
}
