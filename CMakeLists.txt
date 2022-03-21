cmake_minimum_required(VERSION 3.20)
project(learn_libcuDF CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

set(HEADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(MY_INCLUDES /home/dys/anaconda3/envs/cudf_dev/include ${HEADER_DIR})
set(MY_LINKS /home/dys/anaconda3/envs/cudf_dev/lib/libcudf.so fmt::fmt)

add_executable(join     ${SRC_DIR}/join.cu      ${HEADER_DIR}/common_utils.h)
add_executable(groupby  ${SRC_DIR}/groupby.cu   ${HEADER_DIR}/common_utils.h)
add_executable(hashjoin ${SRC_DIR}/hashjoin.cu  ${HEADER_DIR}/common_utils.h)
add_executable(select   ${SRC_DIR}/select.cu    ${HEADER_DIR}/common_utils.h)
add_executable(filter   ${SRC_DIR}/filter.cu    ${HEADER_DIR}/common_utils.h)

set(CUDA_SEPARABLE_COMPILATION ON)
find_package(fmt)
target_include_directories(join     PUBLIC ${MY_INCLUDES})
target_include_directories(groupby  PUBLIC ${MY_INCLUDES})
target_include_directories(hashjoin PUBLIC ${MY_INCLUDES})
target_include_directories(select   PUBLIC ${MY_INCLUDES})
target_include_directories(filter  PUBLIC ${MY_INCLUDES})

target_link_libraries(join      PUBLIC ${MY_LINKS})
target_link_libraries(groupby   PUBLIC ${MY_LINKS})
target_link_libraries(hashjoin  PUBLIC ${MY_LINKS})
target_link_libraries(select    PUBLIC ${MY_LINKS})
target_link_libraries(filter    PUBLIC ${MY_LINKS})