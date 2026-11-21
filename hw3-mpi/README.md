# HW3 · Distributed Gaussian elimination with MPI

[gauss_mpi.c](gauss_mpi.c) solves $Ax=b$ with the same no-pivot row
operations as [HW2](../hw2-shared-memory/README.md). MPI ranks have
separate address spaces, so the code explicitly distributes rows and
moves the pivot data between processes.

## Which rank owns which row?

Global row `i` stays on rank `i % p`, where `p` is the number of
ranks. Its local index is `i / p` using integer division. For seven rows
and three ranks, the ownership looks like this:

| Rank | Global rows |
| --- | --- |
| 0 | 0, 3, 6 |
| 1 | 1, 4 |
| 2 | 2, 5 |

This interleaving spreads the rows near the bottom of the matrix across
ranks, so the shrinking update region remains reasonably balanced.
An augmented row contains `n + 1` doubles, including the right-hand side.

## What happens in the code?

1. Rank 0 generates the system and packs rows by destination. `MPI_Scatterv`
   sends each rank its own rows, including uneven batches.
2. At pivot `k`, rank `k % p` normalizes its row and broadcasts columns
   `k` through `n` with `MPI_Bcast`. Column `n` holds the modified
   right-hand side. Previous columns are no longer needed.
3. Every rank updates its local rows whose global index exceeds `k`.
   In the seven-row example, pivot 4 belongs to rank 1. Rank 2 updates
   row 5, rank 0 updates row 6, and rank 1 has no remaining rows to update.
4. `MPI_Gatherv` returns the triangular rows in rank order. Rank 0 restores
   global order, back-substitutes, and checks the answer against the known
   solution. All ranks receive the final pass or fail status.

Each pivot requires a broadcast. More ranks reduce per-rank arithmetic,
while communication and the sequential ordering of pivots limit speedup.

## Try it

From the [repository root](../README.md), after `make mpi`:

```sh
mpiexec -n 3 build/gauss_mpi 7
mpiexec -n 4 build/gauss_mpi 5000 1
```

The printed time is the maximum across ranks for the scatter-through-gather
phase, measured with `MPI_Wtime`. Setup and the final solution check run
outside that interval. See [verification results](../DESIGN.md#local-verification).
