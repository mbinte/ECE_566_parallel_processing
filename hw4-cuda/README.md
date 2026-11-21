# HW4 · CUDA programming

The host code runs on the CPU. It launches kernels that run as many threads
on the GPU, grouped into blocks. A launch of the form
`kernel<<<blocks, threads_per_block>>>(arguments)` sets the grid size.
Computational kernels use

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

to map each thread to a unique index. If the grid is rounded up to cover
all data, the last block may contain threads with no assigned work. An
index check keeps those threads from accessing memory out of bounds.

| Program | What each GPU thread does | How the result is checked |
| --- | --- | --- |
| [hello.cu](hello.cu) | Prints its block and thread index. The grid has two blocks with two threads each. | `cudaDeviceSynchronize` waits for completion. The four lines may appear in any order. |
| [vector_add.cu](vector_add.cu) | Adds one pair of elements and writes `c[i] = a[i] + b[i]`. | The host checks every element. The length 4099 exercises the bounds check with a block size of 256. |
| [matrix_sums.cu](matrix_sums.cu) | Adds the entries in one row or one column. The host launches the same kernel once for each direction. | CPU reference totals are computed for both directions using asymmetric input. |

The matrix is row-major. Row `i` reads `matrix[i*n+j]`, which steps
through contiguous values within one thread. Column `i` reads
`matrix[j*n+i]`, which steps by `n` within one thread. For the column
launch, adjacent threads read adjacent columns at a given loop iteration.
Both launches read $n^2$ elements.

## Follow the data

Vector addition and matrix sums allocate GPU buffers with `cudaMalloc`,
copy inputs to the GPU, launch kernels, copy the result back to the CPU,
and free the GPU buffers. A successful blocking device-to-host copy
makes the output available for the CPU check. All programs check runtime
calls and immediate launch errors through
[cuda_check.cuh](../common/cuda_check.cuh).

From the [repository root](../README.md), run on a CUDA-capable host:

```sh
make cuda
build/hello
build/vector_add
build/matrix_sums 513
```

The optional matrix size of 513 exercises the final partial block and uses
less memory than the default. The [root README](../README.md#build)
describes the remote GPU option.
