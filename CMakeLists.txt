cmake_minimum_required(VERSION 3.20)
project(learn_libcuDF CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(learn_libcuDF main.cu common_utils.h)

set_target_properties(learn_libcuDF PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(learn_libcuDF
        PUBLIC /home/dys/anaconda3/envs/cudf_dev/include)
target_link_libraries(learn_libcuDF
        PUBLIC /home/dys/anaconda3/envs/cudf_dev/lib/libcudf.so)