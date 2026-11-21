/* Rank r owns global rows r, r+p, r+2p, ... throughout elimination. */
#include <mpi.h>
#include <string.h>
#include "../common/gauss.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    int n = argc > 1 ? dimension(argv[1]) : 0;
    if (!n || argc > 3) {
        if (!rank) fprintf(stderr, "Usage: %s N [seed]\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    int width = n + 1;
    int *count = allocate(p, sizeof(*count));
    int *offset = allocate(p, sizeof(*offset));
    for (int r = 0; r < p; ++r) {
        count[r] = (n / p + (r < n % p)) * width;
        if (r) offset[r] = offset[r - 1] + count[r - 1];
    }
    double *rows = allocate(count[rank] ? count[rank] : 1, sizeof(*rows));
    double *pivot = allocate(width, sizeof(*pivot));
    double *a = NULL, *packed = NULL;
    size_t bytes = width * sizeof(*rows);
    if (!rank) {
        a = allocate((size_t)n * width, sizeof(*a));
        packed = allocate((size_t)n * width, sizeof(*packed));
        initialize(a, n, argc == 3 ? (unsigned)strtoul(argv[2], NULL, 10) : 1);
        for (int i = 0; i < n; ++i)
            memcpy(packed + offset[i % p] + (size_t)(i / p) * width,
                   a + (size_t)i * width, bytes);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    MPI_Scatterv(packed, count, offset, MPI_DOUBLE, rows, count[rank],
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int k = 0; k < n; ++k) {
        int owner = k % p;
        if (rank == owner) {
            double *row = rows + (size_t)(k / p) * width;
            normalize(row, k, n);
            memcpy(pivot + k, row + k, (width - k) * sizeof(*pivot));
        }
        MPI_Bcast(pivot + k, width - k, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        int first = k / p + (rank <= owner);
        for (int i = first; i < count[rank] / width; ++i)
            eliminate(rows + (size_t)i * width, pivot, k, n);
    }
    MPI_Gatherv(rows, count[rank], MPI_DOUBLE, packed, count, offset,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start, maximum;
    MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int failed = 0;
    if (!rank) {
        printf("distributed_elimination_seconds=%.6f\n", maximum);
        for (int i = 0; i < n; ++i)
            memcpy(a + (size_t)i * width,
                   packed + offset[i % p] + (size_t)(i / p) * width, bytes);
        failed = solve_and_check(a, n);
    }
    MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    free(a);
    free(packed);
    free(rows);
    free(pivot);
    free(count);
    free(offset);
    MPI_Finalize();
    return failed;
}
