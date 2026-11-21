// Completed from the course vector-add skeleton and the submitted solution.
#include "../common/cuda_check.cuh"
#include <cmath>
#include <vector>

__global__ void add(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main()
{
    constexpr int n = 4099, block = 256;
    constexpr size_t bytes = n * sizeof(float);
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) {
        a[i] = std::rand() / static_cast<float>(RAND_MAX);
        b[i] = std::rand() / static_cast<float>(RAND_MAX);
    }
    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));
    CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
    add<<<(n + block - 1) / block, block>>>(da, db, dc, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));

    std::printf("A[0] = %f\nB[0] = %f\nC[0] = %f\n", a[0], b[0], c[0]);
    for (int i = 0; i < n; ++i) {
        if (!(std::fabs(c[i] - (a[i] + b[i])) <= 1e-6f)) {
            std::fprintf(stderr, "Vector result differs at index %d\n", i);
            return EXIT_FAILURE;
        }
    }
    std::puts("vector sum correct!");
}
