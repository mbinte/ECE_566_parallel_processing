// One block at a time per column. Threads access contiguous values in column-major storage.
#include "../common/cuda_check.cuh"
#include <algorithm>
#include <cmath>
#include <vector>

__device__ float block_sum(float value, float *partials)
{
    unsigned int t = threadIdx.x;
    partials[t] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (t < stride) partials[t] += partials[t + stride];
        __syncthreads();
    }
    float sum = partials[0];
    __syncthreads();
    return sum;
}

__global__ void normalize_columns(const float *input, float *output, int n)
{
    extern __shared__ float partials[];
    for (int col = blockIdx.x; col < n; col += gridDim.x) {
        int offset = col * n;
        float sum = 0;
        for (int row = threadIdx.x; row < n; row += blockDim.x)
            sum += input[offset + row];
        float mean = block_sum(sum, partials) / n;
        float variance_sum = 0;
        for (int row = threadIdx.x; row < n; row += blockDim.x) {
            float delta = input[offset + row] - mean;
            variance_sum += delta * delta;
        }
        float deviation = sqrtf(block_sum(variance_sum, partials) / n);
        for (int row = threadIdx.x; row < n; row += blockDim.x)
            output[offset + row] = deviation ? (input[offset + row] - mean) / deviation : 0;
    }
}

static long integer(const char *input, long low, long high)
{
    char *end;
    long value = std::strtol(input, &end, 10);
    return end != input && !*end && value >= low && value <= high ? value : -1;
}

int main(int argc, char **argv)
{
    int n = argc > 1 ? integer(argv[1], 1, 8000) : 1024;
    int blocks = argc > 2 ? integer(argv[2], 1, 65535) : 128;
    int threads = argc > 3 ? integer(argv[3], 1, 1024) : 256;
    long seed = argc > 4 ? integer(argv[4], 0, 2147483647) : 1;
    if (argc > 5 || n < 1 || blocks < 1 || threads < 1 || seed < 0 ||
        (threads & (threads - 1))) {
        std::fprintf(stderr, "Usage: %s [size [blocks [power-of-two threads [seed]]]]\n", argv[0]);
        return EXIT_FAILURE;
    }
    std::srand(static_cast<unsigned>(seed));
    std::vector<float> input(n * n), output(n * n);
    // Include a constant column to verify the zero-variance convention.
    for (int col = 0; col < n; ++col)
        for (int row = 0; row < n; ++row)
            input[col * n + row] = col ? std::rand() / static_cast<float>(RAND_MAX) : 2;

    float *di, *dout;
    size_t bytes = input.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&di, bytes));
    CUDA_CHECK(cudaMalloc(&dout, bytes));
    CUDA_CHECK(cudaMemcpy(di, input.data(), bytes, cudaMemcpyHostToDevice));
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    normalize_columns<<<blocks, threads, threads * sizeof(float)>>>(di, dout, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float elapsed;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, end));
    CUDA_CHECK(cudaMemcpy(output.data(), dout, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(di));
    CUDA_CHECK(cudaFree(dout));

    double maximum_error = 0;
    for (int col = 0; col < n; ++col) {
        int offset = col * n;
        double mean = 0, variance = 0;
        for (int row = 0; row < n; ++row) mean += input[offset + row];
        mean /= n;
        for (int row = 0; row < n; ++row) {
            double delta = input[offset + row] - mean;
            variance += delta * delta;
        }
        double deviation = std::sqrt(variance / n);
        for (int row = 0; row < n; ++row) {
            double expected = deviation ? (input[offset + row] - mean) / deviation : 0;
            double error = std::fabs(output[offset + row] - expected);
            if (!std::isfinite(error)) return EXIT_FAILURE;
            maximum_error = std::max(maximum_error, error);
        }
    }
    std::printf("kernel_ms=%.6f max_error=%.6g\n", elapsed, maximum_error);
    return maximum_error <= 1e-4 ? EXIT_SUCCESS : EXIT_FAILURE;
}
