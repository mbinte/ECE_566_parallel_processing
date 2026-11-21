#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

inline void check_cuda(cudaError_t error, const char *operation)
{
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(operation) check_cuda((operation), #operation)
