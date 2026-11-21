# ECE 566 Parallel Processing

I took ECE 566 (Parallel Processing) in Spring 2024. This repository collects
my coding assignments and final project from that semester. Homework 1 was
written. Program inputs, build commands, and checks are below. [DESIGN.md](DESIGN.md) describes the algorithms and records the
local verification results.

| Folder | Programs |
| --- | --- |
| `hw2-shared-memory/` | Serial, Pthreads, and OpenMP Gaussian elimination |
| `hw3-mpi/` | Gaussian elimination with cyclic MPI row distribution |
| `hw4-cuda/` | CUDA Hello World, vector addition, and matrix row and column sums |
| `extra-cuda/` | CUDA column normalization and CPU reference check |
| `project-convolution/` | Serial FFT convolution, two MPI SPMD variants, and an MPI task-group variant |

## Build

Run the commands at the root of this folder. CPU programs require a C11
compiler with Pthreads and an OpenMP compiler. MPI targets require `mpicc`
and an MPI runtime. Build the CUDA targets on a machine with `nvcc` and run
them on a machine with a CUDA-capable NVIDIA GPU. The GPU can be on a remote
host. The Spring 2024 HW4 instructions list Chameleon as an option. The
convolution checker uses Python 3.8 or newer.

```sh
make cpu mpi
make cuda
```

On the Apple Silicon machine used for verification, Homebrew GCC 15 supplied
OpenMP. With Command Line Tools installed, use:

```sh
make cpu mpi OMP_CC=gcc-15 OMP_FLAGS="-fopenmp -isysroot $(xcrun --show-sdk-path)"
```

## HW2 and HW3 — Gaussian elimination

These programs generate a seeded, diagonally dominant system with a known
solution. Output includes elapsed seconds and maximum solution error. A check
failure returns a nonzero status.

```sh
build/gauss_serial 257
build/gauss_pthreads 257 4
build/gauss_openmp 257 4
mpiexec -n 4 build/gauss_mpi 257
```

The shared-memory executables take a matrix dimension, optional thread count,
and optional seed. The serial executable uses one thread and accepts the same
arguments. The MPI executable takes a matrix dimension and optional seed.
For the HW3 performance sweep, run on an MPI host with enough slots:

```sh
for ranks in 1 2 4 8 16; do
    mpiexec -n "$ranks" build/gauss_mpi 5000 1
done
```

## HW4 and extra CUDA

```sh
build/hello
build/vector_add
build/matrix_sums 513
build/matrix_norm 1024 128 256 1
```

Hello prints four GPU greetings in any order. Vector addition checks every
element. Matrix sums accepts a side length and defaults to the course size of
16384, which requires about 1 GiB for the matrix on both host and device.
Normalization accepts size, grid blocks, power-of-two threads per block, and
seed. It reports kernel-only time and maximum error against a double-precision
CPU calculation. For a GPU scaling sweep, vary matrix size, grid blocks, and
threads per block independently.

## Project — 2D circular convolution

Each executable accepts two 512 × 512 ASCII images and an output filename.
The output has one line per row. Use the course samples already in
`../Project/sample/` if this folder is next to the original course files:

```sh
build/conv_serial ../Project/sample/2_im1 ../Project/sample/2_im2 build/out_serial.txt
mpiexec -n 4 build/conv_sendrecv ../Project/sample/2_im1 ../Project/sample/2_im2 build/out_sr.txt
mpiexec -n 4 build/conv_collective ../Project/sample/2_im1 ../Project/sample/2_im2 build/out_coll.txt
mpiexec -n 8 build/conv_tasks ../Project/sample/2_im1 ../Project/sample/2_im2 build/out_tasks.txt
```

For a self-contained check with asymmetric synthetic input and an independent
direct convolution reference:

```sh
python3 tests/check_convolution.py build/conv_serial \
    "mpiexec -n 4 build/conv_sendrecv" "mpiexec -n 4 build/conv_collective" \
    "mpiexec -n 8 build/conv_tasks"
```

For performance, repeat the two SPMD commands at 1, 2, 4, 8, 12, and 16 ranks on
the same host or cluster. Compare medians and calculate speedup from each
variant's one-rank baseline. Compare all three MPI variants at eight ranks.
`conv_tasks` requires exactly eight ranks.
