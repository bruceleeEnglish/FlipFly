cmake_minimum_required(VERSION 3.21)
project(ddc)

set(CMAKE_CXX_STANDARD 17)

add_executable(logger_hnsw_pca logger_hnsw_pca.cpp)
target_link_libraries(logger_hnsw_pca FileIO Logger hnswlib)
set_target_properties(logger_hnsw_pca PROPERTIES COMPILE_FLAGS "-O3")

add_executable(logger_hnsw_opq logger_hnsw_opq.cpp)
target_link_libraries(logger_hnsw_opq FileIO Logger hnswlib)
set_target_properties(logger_hnsw_opq PROPERTIES COMPILE_FLAGS "-O3")

add_executable(logger_ivf_pca logger_ivf_pca.cpp)
target_link_libraries(logger_ivf_pca FileIO Logger hnswlib ivf)
set_target_properties(logger_ivf_pca PROPERTIES COMPILE_FLAGS "-O3")

add_executable(logger_ivf_opq logger_ivf_opq.cpp)
target_link_libraries(logger_ivf_opq FileIO Logger hnswlib ivf)
set_target_properties(logger_ivf_opq PROPERTIES COMPILE_FLAGS "-O3")

add_executable(logger_hnsw_pca_ip logger_hnsw_pca_ip.cpp)
target_link_libraries(logger_hnsw_pca_ip FileIO Logger hnswlib)
set_target_properties(logger_hnsw_pca_ip PROPERTIES COMPILE_FLAGS "-O3")

add_executable(logger_hnsw_pca_ip2 logger_hnsw_pca_ip2.cpp)
target_link_libraries(logger_hnsw_pca_ip2 FileIO Logger hnswlib)
set_target_properties(logger_hnsw_pca_ip2 PROPERTIES COMPILE_FLAGS "-O3")

add_executable(logger_hnsw_opq_ip logger_hnsw_opq_ip.cpp)
target_link_libraries(logger_hnsw_opq_ip FileIO Logger hnswlib)
set_target_properties(logger_hnsw_opq_ip PROPERTIES COMPILE_FLAGS "-O3")

add_executable(logger_hnsw_opq_ip2 logger_hnsw_opq_ip2.cpp)
target_link_libraries(logger_hnsw_opq_ip2 FileIO Logger hnswlib)
set_target_properties(logger_hnsw_opq_ip2 PROPERTIES COMPILE_FLAGS "-O3")

add_executable(generate_hnsw_training_set_test test/generate_hnsw_training_set_test.cpp)
target_link_libraries(generate_hnsw_training_set_test FileIO Logger hnswlib)

add_executable(correct_hnsw_linear_model_test test/correct_hnsw_linear_model_test.cpp)
target_link_libraries(correct_hnsw_linear_model_test FileIO Logger hnswlib)

add_executable(binery_search_parameter test/binery_search_parameter_test.cpp)
target_link_libraries(binery_search_parameter FileIO Logger hnswlib)

#add_executable(logger_hnsw_pca_ip_test test/logger_hnsw_pca_ip_test.cpp)
#target_link_libraries(logger_hnsw_pca_ip_test FileIO Logger hnswlib)
#
#add_executable(logger_hnsw_pca_ip2_test test/logger_hnsw_pca_ip2_test.cpp)
#target_link_libraries(logger_hnsw_pca_ip2_test FileIO Logger hnswlib)
#
#add_executable(logger_hnsw_opq_ip2_test test/logger_hnsw_opq_ip2_test.cpp)
#target_link_libraries(logger_hnsw_opq_ip2_test FileIO Logger hnswlib)