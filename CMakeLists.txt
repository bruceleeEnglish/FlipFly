cmake_minimum_required(VERSION 3.21)
project(DCOs)

set(CMAKE_CXX_STANDARD 17)

# Use the Eigen3 headers under the project-local third_party/eigen directory
# Eigen is header-only and needs no separate build
include_directories(${CMAKE_SOURCE_DIR}/third_party/eigen)

add_library(FileIO
        FileIO/File_IO.h
        FileIO/File_IO.cpp
        )

add_library(Logger
        Logger/Logger.h
        Logger/Logger.cpp
        )

add_library(hnswlib
        Logger
        include/hnswlib/bruteforce.h
        include/hnswlib/hnswlib.h
        include/hnswlib/hnswalg.h
        include/hnswlib/space_ip.h
        include/hnswlib/space_l2.h
        include/hnswlib/stop_condition.h
        include/hnswlib/visited_list_pool.h
        DCO/DCO.hpp
        DCO/L2/FDScanning.hpp
        DCO/L2/PDScanning.hpp
        DCO/L2/ADSampling.hpp
        DCO/L2/DADE.hpp
        DCO/L2/RPDScanning.hpp
        DCO/L2/DDCres.hpp
        DCO/L2/DDCpca.hpp
        DCO/L2/DDCopq.hpp
        # new_add: SRHT-based DCO methods (L2)
        DCO/L2/SRHTADSampling.hpp
        DCO/L2/SRHTPCAADSampling.hpp
        DCO/IP/FDScanningIP.hpp
        DCO/IP/PDScanningIP.hpp
        DCO/IP/RPDScanningIP.hpp
        DCO/IP/ADSamplingIP.hpp
        DCO/IP/DADEIP.hpp
        DCO/IP/DDCresIP.hpp
        DCO/IP/DDCpcaIP.hpp
        DCO/IP/DDCopqIP.hpp
        DCO/IP/DDCpcaIP2.hpp
        DCO/IP/DDCopqIP2.hpp
        # new_add: SRHT IP variant
        DCO/IP/SRHTADSamplingIP.hpp
        global.h
        utils.h
        )
set_target_properties(hnswlib PROPERTIES LINKER_LANGUAGE CXX)

add_library(ivf
        include/ivf/ivf.h
        )
set_target_properties(ivf PROPERTIES LINKER_LANGUAGE CXX)

add_executable(DCOs main.cpp)

add_subdirectory(GroundTruth)

add_subdirectory(Construction)

add_subdirectory(src)

add_subdirectory(data/ddc)

#add_subdirectory(GPU)