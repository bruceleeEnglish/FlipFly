#include <iostream>
#include <getopt.h>
#include <fstream>
#include <ctime>
#include <cmath>
#include <queue>
#include "../utils.h"
#include "../Logger/Logger.h"
#include "../FileIO/File_IO.h"

int main(int argc, char *argv[]) {
    //// parse arguments
    const struct option opts[] = {
            {"help",              no_argument,       0, 'h'},
            {"k",                 required_argument, 0, 'K'},
            {"dataset_path",      required_argument, 0, 'n'},
            {"learning_path",     required_argument, 0, 'l'},
            {"ground_truth_path", required_argument, 0, 'g'},
            {"size",              required_argument, 0, 'z'},
            {"metric",            required_argument, 0, 'm'},
    };
    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    int K = 100;
    std::string dataset_path;
    std::string learning_path;
    std::string ground_truth_path;
    int size = -1;
    int metric = 0;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "k:n:l:g:z:m:", opts, &ind);
        switch (iarg) {
            case 'k':
                if (optarg) K = atoi(optarg);
                break;
            case 'n':
                if (optarg) dataset_path = optarg;
                break;
            case 'l':
                if (optarg) learning_path = optarg;
                break;
            case 'g':
                if (optarg) ground_truth_path = optarg;
                break;
            case 'z':
                if (optarg) size = atoi(optarg);
                break;
            case 'm':
                if (optarg) metric = atoi(optarg);
                break;
        }
    }

    //// check arguments
    if (dataset_path.empty()) {
        std::cerr << "Error: dataset_path is empty. Please provide a valid path using the -n option." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (learning_path.empty()) {
        std::cerr << "Error: learning_path is empty. Please provide a valid path using the -l option." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (ground_truth_path.empty()) {
        std::cerr << "Error: ground_truth_path is empty. Please provide a valid path using the -g option." << std::endl;
        exit(EXIT_FAILURE);
    }


    std::cout << "K: " << K << std::endl;
    std::cout << "dataset_path: " << dataset_path << std::endl;
    std::cout << "learning_path: " << learning_path << std::endl;
    std::cout << "ground_truth_path: " << ground_truth_path << std::endl;
    std::cout << "size: " << size << std::endl;
    std::cout << "metric: " << metric << std::endl;

    //// read
    std::vector <VectorDataType> data_list;
    ReadVectorData(dataset_path, data_list);

    std::vector <VectorDataType> learning_list;
    ReadVectorData(learning_path, learning_list);

    if (size != -1) {
        data_list.resize(size);
    }

    //// add point
    int dim = data_list[0].Dimension();
    int max_elements = data_list.size();
    hnswlib::BruteforceSearch<float> *alg_brute;
    if (metric == 1) {
        hnswlib::InnerProductSpace space_ip(dim);
        alg_brute = new hnswlib::BruteforceSearch<float>(&space_ip, max_elements);
    } else {
        hnswlib::L2Space space(dim);
        alg_brute = new hnswlib::BruteforceSearch<float>(&space, max_elements);
    }
    for (int i = 0; i < max_elements; i++) {
        alg_brute->addPoint(data_list[i].data.data(), data_list[i].vid);
    }

    //// groundtruth
    int num = 0;
    std::vector<std::vector<VidType>> answer_list_brute;
    for (auto point : learning_list) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_brute->searchKnn(
                point.data.data(), K);
        std::vector<VidType> answer;
        while (!result.empty()) {
            answer.push_back(result.top().second);
            result.pop();
        }
        std::reverse(answer.begin(), answer.end());
        answer_list_brute.push_back(answer);
        if (num % 500 == 0) std::cout << "Processed " << num << " points." << std::endl;
        num++;
    }

    ////  dump
    DumpGroundTruth(ground_truth_path, answer_list_brute);
    delete alg_brute;
    return 0;
}
