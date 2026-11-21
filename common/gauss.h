#ifndef GAUSS_H
#define GAUSS_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Adapted from Assignments/gauss.c. Store b in the last column. */
static inline void *allocate(size_t count, size_t size)
{
    void *p = calloc(count, size);
    if (!p) { perror("calloc"); exit(EXIT_FAILURE); }
    return p;
}

static inline int dimension(const char *arg)
{
    char *end;
    long n = strtol(arg, &end, 10);
    return end != arg && !*end && n >= 1 && n <= 10000 ? (int)n : 0;
}

static inline double known_solution(int j)
{
    return (j % 11 - 5) / 5.0;
}

static inline void initialize(double *a, int n, unsigned seed)
{
    srand(seed);
    for (int i = 0; i < n; ++i) {
        double *row = a + (size_t)i * (n + 1);
        double sum = 0;
        for (int j = 0; j < n; ++j) {
            row[j] = 2.0 * rand() / RAND_MAX - 1;
            if (j != i) sum += fabs(row[j]);
        }
        /* Strict diagonal dominance keeps no-pivot test systems well behaved. */
        row[i] = sum + 1;
        for (int j = 0; j < n; ++j) row[n] += row[j] * known_solution(j);
    }
}

static inline void normalize(double *row, int k, int n)
{
    double pivot = row[k];
    for (int j = k + 1; j <= n; ++j) row[j] /= pivot;
    row[k] = 1;
}

static inline void eliminate(double *row, const double *pivot, int k, int n)
{
    double multiplier = row[k];
    for (int j = k + 1; j <= n; ++j) row[j] -= multiplier * pivot[j];
    row[k] = 0;
}

static inline int solve_and_check(const double *a, int n)
{
    double *x = allocate(n, sizeof(*x));
    double error = 0;
    for (int i = n - 1; i >= 0; --i) {
        const double *row = a + (size_t)i * (n + 1);
        x[i] = row[n];
        for (int j = i + 1; j < n; ++j) x[i] -= row[j] * x[j];
        double difference = fabs(x[i] - known_solution(i));
        error = isfinite(difference) ? fmax(error, difference) : INFINITY;
    }
    if (n <= 10) {
        printf("x =");
        for (int i = 0; i < n; ++i) printf(" %.9g", x[i]);
        puts("");
    }
    printf("max solution error = %.3g\n", error);
    free(x);
    return error > 1e-9;
}
#endif
