#include <stdio.h>
#include <omp.h>

int main(void) {
    int Aryabhatta = 10;

    #pragma omp parallel private(Aryabhatta)
    {
        int tid = omp_get_thread_num();
        Aryabhatta = 10;
        printf("Thread %d -> %d * %d = %d\n", tid, tid, Aryabhatta, tid * Aryabhatta);
    }

    return 0;
}
