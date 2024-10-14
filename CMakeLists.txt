cmake_minimum_required(VERSION 3.10)

project(LayerNormPlugin)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CUDA_NVCC_FLAGS}")

add_library(LayerNormPlugin SHARED LayerNorm.cu LayerNormPlugin.h LayerNormPlugin.cpp)

target_link_libraries(LayerNormPlugin ${CUDA_LIBRARIES})

# <------------------------TensorRT Related------------------------->
#find_package(TensorRT REQUIRED)
#include_directories(${TensorRT_INCLUDE_DIRS})
include_directories(/home/xiaoying/code/dep/TensorRT-8.6.1.6/include)
link_directories(/home/xiaoying/code/dep/TensorRT-8.6.1.6/lib)

add_executable(test main.cpp)
set_target_properties(test PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(test LayerNormPlugin)
target_link_libraries(test nvinfer)
target_link_libraries(test nvinfer_plugin)
target_link_libraries(test cudart)
