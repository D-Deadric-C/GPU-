#include <stdio.h>
#include <omp.h>

int main(void) {
    const int N = 20;
    double t0 = omp_get_wtime();

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            int tid = omp_get_thread_num();
            printf("Series of 2 by thread %d:\n", tid);
            for (int i = 1; i <= N; ++i) printf("%d ", 2 * i);
            printf("\n");
        }

        #pragma omp section
        {
            int tid = omp_get_thread_num();
            printf("Series of 4 by thread %d:\n", tid);
            for (int i = 1; i <= N; ++i) printf("%d ", 4 * i);
            printf("\n");
        }
    }

    double t1 = omp_get_wtime();
    printf("Parallel sections time: %.6f s\n", t1 - t0);
    return 0;
}
