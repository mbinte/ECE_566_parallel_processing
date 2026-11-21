# Extra credit · CUDA column normalization

[matrix_norm.cu](matrix_norm.cu) standardizes each matrix column using that
column's mean and population standard deviation. For a column with $n$
values $x_r$, the calculation is

$$
\mu = \frac{1}{n}\sum_r x_r,\qquad
\sigma = \sqrt{\frac{1}{n}\sum_r (x_r-\mu)^2},\qquad
z_r = \frac{x_r-\mu}{\sigma}.
$$

For example, the column `[1, 3]` has mean 2 and standard deviation 1, so
its output is `[-1, 1]`. A constant column has zero deviation and is
mapped to zeros.

## How the kernel works

1. `normalize_columns` gives each block a column. A block can process
   later columns by stepping forward by `gridDim.x`. The column-major
   address `col*n + row` lets adjacent threads access adjacent rows.
2. Threads stride through their column and sum their own subsets.
   `block_sum` places the partial sums in shared memory. In each
   reduction round, active threads combine pairs of partials and the
   block synchronizes before the next round. The halving scheme uses a
   power-of-two thread count.
3. Once the mean is available to all threads, they repeat the reduction
   on squared deviations. They then apply the normalization to their own
   subsets. Each output element has exactly one writer.

The two reductions keep intermediate totals within the block. There is
$O(n^2)$ total matrix work. Increasing the number of threads can reduce
the number of rows each thread visits, but also increases the shared-memory
reduction and synchronization work.

## Try it

From the [repository root](../README.md), after `make cuda`:

```sh
build/matrix_norm 513 64 128 1
```

Arguments are matrix size, grid blocks, threads per block, and seed.
Here there are more columns than blocks, so the grid-stride loop is used.
The host includes a constant column and compares all outputs with a
double-precision CPU calculation. CUDA events report kernel time separately
from the transfers and the CPU check.
