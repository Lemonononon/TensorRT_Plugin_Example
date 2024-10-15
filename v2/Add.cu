#include <cuda_runtime.h>
#include <iostream>
#include "AddPluginV2.h"

namespace nvinfer1
{
namespace plugin
{

template <typename T>
__global__ void AddForwardKernel(const T* input0, const T* input1, T* output, int numElements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numElements) {
        output[index] = static_cast<T>(static_cast<float>(input0[index]) + static_cast<float>(input1[index]));
    }
}

template <typename T>
void AddForward(const T* input0, const T* input1, T* output, int numElements, cudaStream_t stream) {
    int threadsPerBlock = 256;
    int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    // float or half

    AddForwardKernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(input0, input1, output, numElements);
}

template void AddForward<float>(const float* input0, const float* input1, float* output, int numElements, cudaStream_t stream);
template void AddForward<__half>(const __half* input0, const __half* input1, __half* output, int numElements, cudaStream_t stream);

}
}
