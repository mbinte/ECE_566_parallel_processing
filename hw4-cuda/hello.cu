// Completed from the course hello skeleton and the submitted solution.
#include "../common/cuda_check.cuh"

__global__ void hello()
{
    printf("Hello from block: %u, thread: %u\n", blockIdx.x, threadIdx.x);
}

int main()
{
    hello<<<2, 2>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
