/* Adapted from the submitted FFT convolution project and course FFT. */
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "fft.c"

#ifndef SIDE
#define SIDE 512
#endif
enum { N = SIDE, PIXELS = N * N };

static complex a[PIXELS], b[PIXELS], scratch[PIXELS];

static int read_image(const char *path, complex *image)
{
    FILE *file = fopen(path, "r");
    if (!file) { perror(path); return 0; }
    int ok = 1;
    for (int i = 0; i < PIXELS; ++i) {
        if (fscanf(file, "%f", &image[i].r) != 1 || !isfinite(image[i].r)) {
            ok = 0;
            break;
        }
    }
    if (ok) {
        char extra;
        ok = fscanf(file, " %c", &extra) == EOF && !ferror(file);
    }
    fclose(file);
    if (!ok) fprintf(stderr, "%s must contain %d finite numbers\n", path, PIXELS);
    return ok;
}

static int write_image(const char *path)
{
    FILE *file = fopen(path, "w");
    if (!file) { perror(path); return 0; }
    for (int i = 0; i < PIXELS; ++i)
        fprintf(file, "%.9g%c", a[i].r, i % N == N - 1 ? '\n' : ' ');
    int ok = !ferror(file);
    if (fclose(file)) ok = 0;
    if (!ok) fprintf(stderr, "Failed to write %s\n", path);
    return ok;
}

static void transpose(complex *image)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            scratch[j * N + i] = image[i * N + j];
    memcpy(image, scratch, sizeof scratch);
}

static void multiply(complex *left, const complex *right, int count)
{
    for (int i = 0; i < count; ++i) {
        complex x = left[i], y = right[i];
        left[i] = (complex){x.r * y.r - x.i * y.i, x.r * y.i + x.i * y.r};
    }
}

#if defined(POINT_TO_POINT) || defined(COLLECTIVE) || defined(TASKS)
#include <mpi.h>

static MPI_Datatype pixel_type;
static double communication, computation;
static complex local_a[PIXELS], local_b[PIXELS];

typedef struct {
    MPI_Comm comm;
    int rank, size;
    int count[N], offset[N];
} Layout;

static Layout create_layout(MPI_Comm comm)
{
    Layout layout = {.comm = comm};
    MPI_Comm_rank(comm, &layout.rank);
    MPI_Comm_size(comm, &layout.size);
    for (int r = 0, start = 0; r < layout.size; ++r) {
        layout.count[r] = N * (N / layout.size + (r < N % layout.size));
        layout.offset[r] = start;
        start += layout.count[r];
    }
    return layout;
}

static void scatter(const Layout *l, const complex *image, complex *local)
{
    double start = MPI_Wtime();
#ifdef POINT_TO_POINT
    if (l->rank == 0) {
        memcpy(local, image, (size_t)l->count[0] * sizeof(*local));
        for (int r = 1; r < l->size; ++r)
            MPI_Send(image + l->offset[r], l->count[r], pixel_type, r, 0, l->comm);
    } else {
        MPI_Recv(local, l->count[l->rank], pixel_type, 0, 0, l->comm, MPI_STATUS_IGNORE);
    }
#else
    MPI_Scatterv(image, l->count, l->offset, pixel_type, local, l->count[l->rank],
                 pixel_type, 0, l->comm);
#endif
    communication += MPI_Wtime() - start;
}

static void gather(const Layout *l, complex *image, const complex *local)
{
    double start = MPI_Wtime();
#ifdef POINT_TO_POINT
    if (l->rank == 0) {
        memcpy(image, local, (size_t)l->count[0] * sizeof(*local));
        for (int r = 1; r < l->size; ++r)
            MPI_Recv(image + l->offset[r], l->count[r], pixel_type, r, 0, l->comm,
                     MPI_STATUS_IGNORE);
    } else {
        MPI_Send(local, l->count[l->rank], pixel_type, 0, 0, l->comm);
    }
#else
    MPI_Gatherv(local, l->count[l->rank], pixel_type, image, l->count, l->offset,
                pixel_type, 0, l->comm);
#endif
    communication += MPI_Wtime() - start;
}

static void fft_rows(const Layout *l, complex *image, int sign)
{
    scatter(l, image, local_a);
    double start = MPI_Wtime();
    for (int i = 0; i < l->count[l->rank]; i += N)
        c_fft1d(local_a + i, N, sign);
    computation += MPI_Wtime() - start;
    gather(l, image, local_a);
}

static void fft2(const Layout *l, complex *image, int sign)
{
    for (int pass = 0; pass < 2; ++pass) {
        fft_rows(l, image, sign);
        if (l->rank == 0) {
            double start = MPI_Wtime();
            transpose(image);
            computation += MPI_Wtime() - start;
        }
    }
}

static void multiply_distributed(const Layout *l)
{
    scatter(l, a, local_a);
    scatter(l, b, local_b);
    double start = MPI_Wtime();
    multiply(local_a, local_b, l->count[l->rank]);
    computation += MPI_Wtime() - start;
    gather(l, a, local_a);
}

#ifdef TASKS
static void transfer(int rank, int from, int to, complex *image)
{
    double start = MPI_Wtime();
    if (rank == from) MPI_Send(image, PIXELS, pixel_type, to, 1, MPI_COMM_WORLD);
    if (rank == to) MPI_Recv(image, PIXELS, pixel_type, from, 1, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
    communication += MPI_Wtime() - start;
}
#endif

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#ifdef TASKS
    int valid = size == 8 && N >= 2;
#else
    int valid = size <= N;
#endif
    if (argc != 4 || !valid) {
        if (!rank) fprintf(stderr, "Usage: %s image1 image2 output (check rank count)\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    int ok = 1;
    if (!rank) ok = read_image(argv[1], a) && read_image(argv[2], b);
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ok) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    MPI_Type_create_struct(2, (int[]){1, 1},
                           (MPI_Aint[]){offsetof(complex, r), offsetof(complex, i)},
                           (MPI_Datatype[]){MPI_FLOAT, MPI_FLOAT}, &pixel_type);
    MPI_Type_commit(&pixel_type);
    MPI_Comm comm = MPI_COMM_WORLD;
#ifdef TASKS
    int task = rank / 2;
    MPI_Comm_split(MPI_COMM_WORLD, task, rank, &comm);
#endif
    Layout layout = create_layout(comm);
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
#ifdef TASKS
    transfer(rank, 0, 2, b);
    if (task == 0) fft2(&layout, a, -1);
    if (task == 1) fft2(&layout, b, -1);
    transfer(rank, 0, 4, a);
    transfer(rank, 2, 4, b);
    if (task == 2) multiply_distributed(&layout);
    transfer(rank, 4, 6, a);
    if (task == 3) fft2(&layout, a, 1);
    transfer(rank, 6, 0, a);
#else
    fft2(&layout, a, -1);
    fft2(&layout, b, -1);
    multiply_distributed(&layout);
    fft2(&layout, a, 1);
#endif
    double times[] = {MPI_Wtime() - start, computation, communication};
    double maximum[3];
    MPI_Reduce(times, maximum, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        printf("total_s=%.6f compute_max_s=%.6f communication_max_s=%.6f\n",
               maximum[0], maximum[1], maximum[2]);
        ok = write_image(argv[3]);
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
#ifdef TASKS
    MPI_Comm_free(&comm);
#endif
    MPI_Type_free(&pixel_type);
    MPI_Finalize();
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
#else
static void fft2(complex *image, int sign)
{
    for (int pass = 0; pass < 2; ++pass) {
        for (int i = 0; i < N; ++i) c_fft1d(image + i * N, N, sign);
        transpose(image);
    }
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "Usage: %s image1 image2 output\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (!read_image(argv[1], a) || !read_image(argv[2], b)) return EXIT_FAILURE;
    fft2(a, -1);
    fft2(b, -1);
    multiply(a, b, PIXELS);
    fft2(a, 1);
    return write_image(argv[3]) ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif
