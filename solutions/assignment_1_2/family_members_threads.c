#include <stdio.h>
#include <omp.h>

int main(void) {
    const char *family[] = {"Father", "Mother", "Brother", "Sister", "Grandmother", "Grandfather"};
    int n = (int)(sizeof(family) / sizeof(family[0]));

    #pragma omp parallel num_threads(n)
    {
        int tid = omp_get_thread_num();
        if (tid < n) {
            printf("Thread %d -> %s\n", tid, family[tid]);
        }
    }

    return 0;
}
