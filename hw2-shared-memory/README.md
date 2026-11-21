# HW2 · Gaussian elimination with Pthreads and OpenMP

The program solves a dense system $Ax=b$. It stores each row of $A$ with
the corresponding entry of $b$ appended. The operations that change the
coefficients also change the right-hand side. See
[gauss_shared.c](gauss_shared.c) and [gauss.h](../common/gauss.h).

## From equations to row updates

At pivot $k$, `normalize` divides the pivot row by its diagonal entry.
For every row $i>k$, `eliminate` then subtracts the normalized pivot row
times the old entry in column $k$. That makes entry $(i,k)$ zero.
For example, the augmented rows `[2, 1 | 5]` and `[4, 3 | 11]`
become `[1, 0.5 | 2.5]` and `[0, 1 | 1]`. Back substitution
finds $x_1=1$ first, then $x_0=2$.

After the final pivot, the matrix is upper triangular with unit diagonal.
`solve_and_check` works upward, subtracting the contributions of unknowns
already solved. Elimination takes $O(n^3)$ work and back substitution takes
$O(n^2)$ work. The generated matrix is strictly diagonally dominant, which
makes this no-pivot approach appropriate for the test cases.

## Where the threads work

| Build | How it handles a pivot |
| --- | --- |
| Serial | One thread normalizes the pivot and updates all rows below it. |
| Pthreads | `worker` 0 normalizes the pivot. Worker `id` updates rows starting at `k + 1 + id`, stepping by the number of workers. |
| OpenMP | Inside one persistent parallel region, `single` normalizes the pivot and `for schedule(static, 1)` distributes rows cyclically. |

Two synchronization points keep each step correct. First, everyone waits
until the pivot row is ready to be read. Second, everyone finishes their
disjoint row updates before the next pivot begins. The Pthreads
`barrier` uses a mutex, a condition variable, and a generation counter
to distinguish successive rounds. OpenMP provides the barriers implicitly
at the end of the `single` and work-sharing loop. Persistent workers
avoid thread creation at each pivot, but barrier costs become more
visible as the number of remaining rows shrinks.

## Try it

From the [repository root](../README.md), after building the CPU targets:

```sh
build/gauss_serial 257
build/gauss_pthreads 257 4
build/gauss_openmp 257 4
```

The timer measures elimination. Each program generates $b$ from a known
solution and checks the recovered values after timing.
