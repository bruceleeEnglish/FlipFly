#pragma once
#ifndef DCOS_UTILS_H
#define DCOS_UTILS_H

#include <chrono>
#include <queue>
#include <unordered_set>
#include <Eigen/Dense>
#include "DCO/DCO.hpp"
#include "DCO/L2/FDScanning.hpp"
#include "DCO/L2/PDScanning.hpp"
#include "DCO/L2/ADSampling.hpp"
#include "DCO/L2/DADE.hpp"
#include "DCO/L2/RPDScanning.hpp"
#include "DCO/L2/DDCres.hpp"
#include "DCO/L2/DDCpca.hpp"
#include "DCO/L2/DDCopq.hpp"
// new_add: include SRHT-based DCO methods
#include "DCO/L2/SRHTADSampling.hpp"
#include "DCO/L2/SRHTPCAADSampling.hpp"
#include "DCO/IP/FDScanningIP.hpp"
#include "DCO/IP/PDScanningIP.hpp"
#include "DCO/IP/RPDScanningIP.hpp"
#include "DCO/IP/ADSamplingIP.hpp"
#include "DCO/IP/DADEIP.hpp"
#include "DCO/IP/DDCresIP.hpp"
#include "DCO/IP/DDCpcaIP.hpp"
#include "DCO/IP/DDCopqIP.hpp"
#include "DCO/IP/DDCpcaIP2.hpp"
#include "DCO/IP/DDCopqIP2.hpp"
// new_add: include SRHT IP variant
#include "DCO/IP/SRHTADSamplingIP.hpp"


std::unique_ptr<DCO> setup_DCO(const std::string& dataset, size_t dim, int op, std::string type="hnsw");


std::unique_ptr<DCO> setup_DCO(const std::string& dataset, size_t dim, int op, std::string type) {
    switch (op) {
        case 0:
            std::cout << "Using FDScanning" << std::endl;
            return std::make_unique<FScanning>(dataset, dim);
        case 1:
            std::cout << "Using PDScanning" << std::endl;
            return std::make_unique<PScanning>(dataset, dim);
        case 2:
            std::cout << "Using RPDScanning" << std::endl;
            return std::make_unique<RPScanning>(dataset, dim);
        case 3:
            std::cout << "Using ADSampling" << std::endl;
            return std::make_unique<ADSampling>(dataset, dim);
        case 4:
            std::cout << "Using DADE" << std::endl;
            return std::make_unique<DADE>(dataset, dim);
        case 5:
            std::cout << "Using DDCres" << std::endl;
            return std::make_unique<DDCres>(dataset, dim, type);
        case 6:
            std::cout << "Using DDCpca" << std::endl;
            return std::make_unique<DDCpca>(dataset, dim, type);
        case 7:
            std::cout << "Using DDCopq" << std::endl;
            return std::make_unique<DDCopq>(dataset, dim, type);
        case 8:
            std::cout << "Using FDScanningIP" << std::endl;
            return std::make_unique<FScanningIP>(dataset, dim);
        case 9:
            std::cout << "Using PDScanningIP" << std::endl;
            return std::make_unique<PScanningIP>(dataset, dim);
        case 10:
            std::cout << "Using RPDScanningIP" << std::endl;
            return std::make_unique<RPScanningIP>(dataset, dim);
        case 11:
            std::cout << "Using ADSamplingIP" << std::endl;
            return std::make_unique<ADSamplingIP>(dataset, dim);
        case 12:
            std::cout << "Using DADEIP" << std::endl;
            return std::make_unique<DADEIP>(dataset, dim);
        case 13:
            std::cout << "Using DDCresIP" << std::endl;
            return std::make_unique<DDCresIP>(dataset, dim, type);
        case 14:
            std::cout << "Using DDCpcaIP" << std::endl;
            return std::make_unique<DDCpcaIP>(dataset, dim, type);
        case 15:
            std::cout << "Using DDCopqIP" << std::endl;
            return std::make_unique<DDCopqIP>(dataset, dim, type);
        case 16:
            std::cout << "Using DDCpcaIP2" << std::endl;
            return std::make_unique<DDCpcaIP2>(dataset, dim, type);
        case 17:
            std::cout << "Using DDCopqIP2" << std::endl;
            return std::make_unique<DDCopqIP2>(dataset, dim, type);
        // new_add: SRHT-ADSampling family (operators 18, 19, 20)
        case 18:
            std::cout << "Using SRHTADSampling" << std::endl;
            return std::make_unique<SRHTADSampling>(dataset, dim);
        case 19:
            std::cout << "Using SRHTADSamplingIP" << std::endl;
            return std::make_unique<SRHTADSamplingIP>(dataset, dim);
        case 20:
            std::cout << "Using SRHTPCAADSampling" << std::endl;
            return std::make_unique<SRHTPCAADSampling>(dataset, dim);
        // new_add: end SRHT family
        default:
            return nullptr;
    }
    return nullptr;
}

#endif //DCOS_UTILS_H
