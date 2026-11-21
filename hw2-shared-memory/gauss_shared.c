/* All workers update disjoint rows. Two barriers per pivot protect
   normalization and ensure updates finish before the next pivot. */
#include "../common/gauss.h"
#if defined(USE_PTHREAD)
#include <pthread.h>

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t condition = PTHREAD_COND_INITIALIZER;
static int arrived, generation, workers, n;
static double *matrix;

static void barrier(void)
{
    pthread_mutex_lock(&mutex);
    int current = generation;
    if (++arrived == workers) {
        arrived = 0;
        ++generation;
        pthread_cond_broadcast(&condition);
    } else {
        while (current == generation) pthread_cond_wait(&condition, &mutex);
    }
    pthread_mutex_unlock(&mutex);
}

static void *worker(void *arg)
{
    int id = *(int *)arg;
    for (int k = 0; k < n; ++k) {
        double *pivot = matrix + (size_t)k * (n + 1);
        if (id == 0) normalize(pivot, k, n);
        barrier();
        for (int i = k + 1 + id; i < n; i += workers)
            eliminate(matrix + (size_t)i * (n + 1), pivot, k, n);
        barrier();
    }
    return NULL;
}
#endif

static void run(double *a, int size, int threads)
{
#if defined(USE_PTHREAD)
    matrix = a;
    n = size;
    workers = threads;
    pthread_t *pool = allocate(threads, sizeof(*pool));
    int *ids = allocate(threads, sizeof(*ids));
    for (int i = 0; i < threads; ++i) {
        ids[i] = i;
        if (pthread_create(&pool[i], NULL, worker, &ids[i])) {
            fputs("Failed to create worker\n", stderr);
            exit(EXIT_FAILURE);
        }
    }
    for (int i = 0; i < threads; ++i) pthread_join(pool[i], NULL);
    free(ids);
    free(pool);
#else
#ifdef _OPENMP
    #pragma omp parallel num_threads(threads)
#else
    (void)threads;
#endif
    for (int k = 0; k < size; ++k) {
        double *pivot = a + (size_t)k * (size + 1);
#ifdef _OPENMP
        #pragma omp single
#endif
        normalize(pivot, k, size);
#ifdef _OPENMP
        #pragma omp for schedule(static, 1)
#endif
        for (int i = k + 1; i < size; ++i)
            eliminate(a + (size_t)i * (size + 1), pivot, k, size);
    }
#endif
}

int main(int argc, char **argv)
{
    int size = argc > 1 ? dimension(argv[1]) : 0;
    int threads = argc > 2 ? dimension(argv[2]) : 1;
    if (!size || !threads || threads > 256 || argc > 4) {
        fprintf(stderr, "Usage: %s N [threads (1-256)] [seed]\n", argv[0]);
        return EXIT_FAILURE;
    }
    unsigned seed = argc == 4 ? (unsigned)strtoul(argv[3], NULL, 10) : 1;
    double *a = allocate((size_t)size * (size + 1), sizeof(*a));
    initialize(a, size, seed);
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    run(a, size, threads);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("elimination_seconds=%.6f\n",
           (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
    int failed = solve_and_check(a, size);
    free(a);
    return failed;
}
