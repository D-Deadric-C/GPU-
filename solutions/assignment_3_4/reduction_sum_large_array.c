#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 50000000;
    double *arr = (double *)malloc((size_t)n * sizeof(double));
    if (!arr) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (int i = 0; i < n; ++i) arr[i] = 1.0;

    double s_seq = 0.0;
    double t0 = omp_get_wtime();
    for (int i = 0; i < n; ++i) s_seq += arr[i];
    double t1 = omp_get_wtime();

    double s_par = 0.0;
    double t2 = omp_get_wtime();
    #pragma omp parallel for reduction(+:s_par)
    for (int i = 0; i < n; ++i) s_par += arr[i];
    double t3 = omp_get_wtime();

    printf("Sequential sum: %.2f, time: %.6f s\n", s_seq, t1 - t0);
    printf("Parallel reduction sum: %.2f, time: %.6f s\n", s_par, t3 - t2);

    free(arr);
    return 0;
}
