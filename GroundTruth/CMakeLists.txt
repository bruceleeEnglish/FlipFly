cmake_minimum_required(VERSION 3.21)
project(GroundTruth)

set(CMAKE_CXX_STANDARD 17)


add_executable(generate_ground_truth generate_ground_truth.cpp)
target_link_libraries(generate_ground_truth FileIO Logger)
set_target_properties(generate_ground_truth PROPERTIES COMPILE_FLAGS "-O3")

add_executable(check_result check_result.cpp)
target_link_libraries(check_result FileIO Logger)
set_target_properties(check_result PROPERTIES COMPILE_FLAGS "-O3")

add_executable(generate_example generate_example.cpp)
target_link_libraries(generate_example FileIO Logger)
