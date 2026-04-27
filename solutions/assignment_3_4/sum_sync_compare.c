#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 10000000;
    int *arr = (int *)malloc((size_t)n * sizeof(int));
    if (!arr) return 1;

    for (int i = 0; i < n; ++i) arr[i] = 1;

    long long sum_unsync = 0;
    double t0 = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        sum_unsync += arr[i];
    }
    double t1 = omp_get_wtime();

    long long sum_critical = 0;
    double t2 = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        #pragma omp critical
        sum_critical += arr[i];
    }
    double t3 = omp_get_wtime();

    long long sum_atomic = 0;
    double t4 = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        #pragma omp atomic
        sum_atomic += arr[i];
    }
    double t5 = omp_get_wtime();

    long long sum_reduction = 0;
    double t6 = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum_reduction)
    for (int i = 0; i < n; ++i) {
        sum_reduction += arr[i];
    }
    double t7 = omp_get_wtime();

    printf("Unsync sum (race): %lld, time: %.6f s\n", sum_unsync, t1 - t0);
    printf("Critical sum: %lld, time: %.6f s\n", sum_critical, t3 - t2);
    printf("Atomic sum: %lld, time: %.6f s\n", sum_atomic, t5 - t4);
    printf("Reduction sum: %lld, time: %.6f s\n", sum_reduction, t7 - t6);

    free(arr);
    return 0;
}
