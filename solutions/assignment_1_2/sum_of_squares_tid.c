#include <stdio.h>
#include <omp.h>

int main(void) {
    int total = 0;

    #pragma omp parallel reduction(+:total)
    {
        int tid = omp_get_thread_num();
        int sq = tid * tid;
        printf("Thread %d square = %d\n", tid, sq);
        total += sq;
    }

    printf("Sum of squares of thread IDs = %d\n", total);
    return 0;
}
