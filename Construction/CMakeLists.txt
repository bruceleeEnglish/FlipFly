cmake_minimum_required(VERSION 3.21)
project(Construction)

set(CMAKE_CXX_STANDARD 17)

add_executable(hnsw_index hnsw_index.cpp)
target_link_libraries(hnsw_index FileIO Logger hnswlib)
set_target_properties(hnsw_index PROPERTIES COMPILE_FLAGS "-O3")

add_executable(hnsw_index_ip hnsw_index_ip.cpp)
target_link_libraries(hnsw_index_ip FileIO Logger hnswlib)
set_target_properties(hnsw_index_ip PROPERTIES COMPILE_FLAGS "-O3")

add_executable(ivf_index ivf_index.cpp)
target_link_libraries(ivf_index FileIO Logger ivf)
set_target_properties(ivf_index PROPERTIES COMPILE_FLAGS "-O3")

add_executable(hnsw_search_index hnsw_search_index.cpp)
target_link_libraries(hnsw_search_index FileIO Logger hnswlib)
set_target_properties(hnsw_search_index PROPERTIES COMPILE_FLAGS "-O3")

add_executable(hnsw_index_example test/hnsw_index_example.cpp)
target_link_libraries(hnsw_index_example FileIO Logger hnswlibO)
set_target_properties(hnsw_index_example PROPERTIES COMPILE_FLAGS "-O3")

add_executable(hnsw_index_test test/hnsw_index_test.cpp)
target_link_libraries(hnsw_index_test FileIO Logger)
#set_target_properties(hnsw_index_test PROPERTIES COMPILE_FLAGS "-O3")

add_executable(ivf_index_test test/ivf_index_test.cpp)
target_link_libraries(ivf_index_test FileIO Logger ivf)
