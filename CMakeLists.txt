cmake_minimum_required(VERSION 3.16)
project(PluginTest)

set(CMAKE_CXX_STANDARD 17)


add_definitions(-std=c++17)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_BUILD_TYPE Debug)

SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")


INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/include)

# cuda
FIND_PACKAGE(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# <------------------------TensorRT Related------------------------->
#find_package(TensorRT REQUIRED)
#include_directories(${TensorRT_INCLUDE_DIRS})
include_directories(/home/xiaoying/code/dep/TensorRT-8.6.1.6/include)
link_directories(/home/xiaoying/code/dep/TensorRT-8.6.1.6/lib)



add_subdirectory(v2 ${CMAKE_BINARY_DIR}/v2)
add_subdirectory(v3 ${CMAKE_BINARY_DIR}/v3)
