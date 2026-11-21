// Completed from the course matrix-sums skeleton and the submitted solution.
#include "../common/cuda_check.cuh"
#include <vector>

__global__ void sums(const float *matrix, float *result, int n, bool columns)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float sum = 0;
    for (int j = 0; j < n; ++j)
        sum += matrix[columns ? j * n + i : i * n + j];
    result[i] = sum;
}

int main(int argc, char **argv)
{
    long size = 16384;
    if (argc == 2) {
        char *end;
        size = std::strtol(argv[1], &end, 10);
        if (end == argv[1] || *end) size = 0;
    }
    if (argc > 2 || size < 1 || size > 16384) {
        std::fprintf(stderr, "Usage: %s [size (1-16384)]\n", argv[0]);
        return EXIT_FAILURE;
    }
    int n = static_cast<int>(size);
    constexpr int block = 256;
    std::vector<float> matrix(n * n), result(n), rows(n), columns(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float value = (i % 7) + 2 * (j % 11);
            matrix[i * n + j] = value;
            rows[i] += value;
            columns[j] += value;
        }

    float *dm, *ds;
    CUDA_CHECK(cudaMalloc(&dm, matrix.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ds, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dm, matrix.data(), matrix.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    bool ok = true;
    for (int pass = 0; pass < 2; ++pass) {
        sums<<<(n + block - 1) / block, block>>>(dm, ds, n, pass == 1);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(result.data(), ds, n * sizeof(float), cudaMemcpyDeviceToHost));
        bool match = result == (pass == 1 ? columns : rows);
        std::printf("%s sums %s!\n", pass == 1 ? "column" : "row",
                    match ? "correct" : "incorrect");
        ok = ok && match;
    }
    CUDA_CHECK(cudaFree(dm));
    CUDA_CHECK(cudaFree(ds));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
