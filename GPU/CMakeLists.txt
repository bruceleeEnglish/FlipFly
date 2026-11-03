cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(GPU LANGUAGES CXX CUDA)

# set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# optimize compilation options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_library(ivf_cuda
        ivf/ivf_cuda2.cuh
        )
set_target_properties(ivf_cuda PROPERTIES LINKER_LANGUAGE CUDA)
target_compile_features(ivf_cuda PUBLIC cxx_std_14)

# Add the executable
add_executable(GPU main.cu)
set_property(TARGET GPU PROPERTY CUDA_ARCHITECTURES 86)
target_compile_options(GPU PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_link_libraries(GPU PRIVATE FileIO)

add_executable(ivf_cuda_search ivf_cuda_search.cu)
target_compile_options(ivf_cuda_search PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_link_libraries(ivf_cuda_search PRIVATE FileIO Logger ivf_cuda)
set_target_properties(ivf_cuda_search PROPERTIES COMPILE_FLAGS "-O3" -march=native -fopenmp)
