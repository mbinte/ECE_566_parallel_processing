# Design and verification

## Gaussian elimination

Both homework implementations retain the pivot-by-pivot elimination and
back substitution from the course starter, `Assignments/gauss.c`. They use a
dynamically sized augmented matrix in double precision. Strict diagonal
dominance gives the generated systems valid pivots without row swaps. The
right-hand side is formed from a known solution so every executable can check
its own answer. Pivot rows are explicitly normalized, including the final row.

Pthreads keeps the same workers alive across all pivots. A reusable
mutex/condition-variable barrier separates pivot normalization from cyclic row
updates, and a second barrier completes writes before the next pivot.
OpenMP uses one parallel region with a `single` pivot step and
`schedule(static, 1)` row updates. Both parallelize the cubic elimination
stage and leave the quadratic back substitution on the main thread.

MPI assigns row `i` to rank `i % p` for all pivots. Rank 0 packs rows in rank
order for a variable-size scatter. Each pivot owner broadcasts just the active
suffix of its normalized row. Each worker updates the rows it owns below the
pivot. After a variable-size gather, rank 0 restores row order and back
substitutes. The cyclic assignment spreads the shrinking row-update workload
across ranks and supports uneven row counts. The reported MPI time is the
maximum elapsed duration for scatter through gather. It excludes initialization,
packing, unpacking, back substitution, and validation.

## CUDA

The three HW4 solutions follow the supplied skeletons and submitted versions.
Runtime calls are checked individually, and launches are checked immediately.
The matrix test uses asymmetric integer inputs so that swapped row and column
kernels would fail the CPU comparison.

The extra-credit program replaces shared scalar updates in the original
normalization code with parallel block reductions. Each block processes one
column at a time and strides across additional columns as needed. Consecutive
threads read consecutive elements in column-major storage. Each thread
accumulates a partial sum, then a tree reduction computes the mean. A second
reduction computes the population variance about that mean. The kernel then
normalizes the column and writes zeros for a constant column. The executable
compares every value against a double-precision CPU calculation.

## FFT convolution

The course's `Project/fft.c` supplies the one-dimensional FFT. Two passes of
row FFTs, each followed by a transpose, implement the 2D transform and return
the image to its original axis order. The program transforms both inputs,
multiplies the complex coefficients elementwise, and applies the inverse
transform. The FFT's inverse scaling is applied in both dimensions.

The SPMD variants split contiguous row slabs, allowing non-divisors of 512
such as 12 ranks. Both gather on the group root for the transpose. The
point-to-point variant uses MPI sends and receives for image slabs, while the
collective variant uses variable-size scatter and gather. The group-root
transpose limits scalability but keeps the data exchange easy to compare
across the two modes.

The task variant uses four communicators with two ranks each. The first two
perform the two forward FFTs independently. Their leaders send the spectra to
the multiplication group, which passes the product to the inverse FFT group.
That last group returns the result to world rank 0. The total time excludes
file I/O. Compute time includes FFTs, transposes, and pointwise products.
Communication time includes MPI calls and their waiting time. Each number is
a maximum over ranks, so the reported maxima need not sum to total time.

## Local verification

The host was an Apple Silicon Mac using Apple Clang, Homebrew GCC 15.2.0 for
OpenMP, and Open MPI 5.0.8.

| Check | Result |
| --- | --- |
| Gaussian serial, Pthreads and OpenMP at sizes 1, 5, 257, 1024 | Passed. Parallel runs used 1, 4, and 8 threads. Largest solution error was 7.11e-15. |
| Gaussian MPI at one rank | Passed at the same four sizes and at size 5000. The size-5000 error was 1.75e-14 and the timed phase took 10.404381 seconds in one local run. |
| FFT convolution against both course reference outputs | Sample 1 matched exactly. Sample 2 had relative L2 error 5.88e-8. Both one-rank MPI SPMD programs matched the serial output byte for byte. |
| Independent sparse circular convolution | Serial and both one-rank SPMD variants passed, with maximum absolute error about 2.79e-4. |
| Multi-rank MPI | All MPI binaries compiled. The local launcher was denied permission to bind sockets, so multi-rank runs, including the eight-rank task variant, remain unverified. |
| CUDA | The host lacks `nvcc` and an NVIDIA GPU. CUDA compile and runtime checks remain pending. |

The timings above are local measurements. Cluster and GPU scaling results
require the hardware and rank counts listed in the README.
