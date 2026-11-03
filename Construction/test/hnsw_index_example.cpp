#include <iostream>
#include <getopt.h>
#include <queue>
#include "../../hnswlib/hnswlib/hnswlib.h"
#include "../../Logger/Logger.h"
#include "../../FileIO/File_IO.h"

int main() {
    std::string dataset_path = "/home/zzlin/dataset/wiki/wikipedia_base.bin.crop_nb_1000000";
    std::string index_path = "/home/zzlin/dataset/wiki/hnsw_wiki_M16_efConstruction500_operator0.index";
    int M = 16;
    int efConstruction = 500;
    int op = 0;

    //// read data
    Logger logger;
    std::vector <VectorDataType> data_list;
    ReadVectorData(dataset_path, data_list);

    //// build index
    logger.SetStartTimer();
    int dim = data_list[0].Dimension();
    int N = data_list.size();
    hnswlib::L2Space space(dim);
//    hnswlib::InnerProductSpace space(dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, N,
                                                                                    M, efConstruction);
    size_t report = 50000;
    for (int i = 0; i < N; i++) {
        alg_hnsw->addPoint(data_list[i].data.data(), data_list[i].vid);
        if (i % report == 0) {
            std::cerr << "Processing - " << i << " / " << N << std::endl;
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