#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static void fill_matrix(double *m, int n) {
    for (int i = 0; i < n * n; ++i) m[i] = (double)(i % 100) / 10.0;
}

static void matmul_serial(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int k = 0; k < n; ++k) s += A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
        }
    }
}

static void matmul_omp(const double *A, const double *B, double *C, int n, int dynamic, int chunk) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n * n; ++i) C[i] = 0.0;

    if (dynamic) {
        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double s = 0.0;
                for (int k = 0; k < n; ++k) s += A[i * n + k] * B[k * n + j];
                C[i * n + j] = s;
            }
        }
    } else {
        #pragma omp parallel for schedule(static, 1)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double s = 0.0;
                for (int k = 0; k < n; ++k) s += A[i * n + k] * B[k * n + j];
                C[i * n + j] = s;
            }
        }
    }
    (void)chunk;
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 512;

    double *A = (double *)malloc((size_t)n * n * sizeof(double));
    double *B = (double *)malloc((size_t)n * n * sizeof(double));
    double *C = (double *)malloc((size_t)n * n * sizeof(double));
    if (!A || !B || !C) {
        fprintf(stderr, "Allocation failed for n=%d\n", n);
        return 1;
    }

    fill_matrix(A, n);
    fill_matrix(B, n);

    double t0 = omp_get_wtime();
    matmul_serial(A, B, C, n);
    double t1 = omp_get_wtime();
    printf("Serial time: %.6f s\n", t1 - t0);

    t0 = omp_get_wtime();
    matmul_omp(A, B, C, n, 0, 1);
    t1 = omp_get_wtime();
    printf("OpenMP static time: %.6f s\n", t1 - t0);

    t0 = omp_get_wtime();
    matmul_omp(A, B, C, n, 1, 1);
    t1 = omp_get_wtime();
    printf("OpenMP dynamic time: %.6f s\n", t1 - t0);

    free(A); free(B); free(C);
    return 0;
}
