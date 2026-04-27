#include <stdio.h>
#include <omp.h>

int main(void) {
    const int N = 20;
    int final_partial = 0;

    #pragma omp parallel
    {
        int local = 0;

        #pragma omp for lastprivate(final_partial)
        for (int i = 1; i <= N; ++i) {
            local += i;
            final_partial = local;
        }

        #pragma omp critical
        {
            printf("Thread %d local partial sum = %d\n", omp_get_thread_num(), local);
        }
    }

    int total = (N * (N + 1)) / 2;
    printf("Lastprivate-captured final_partial = %d\n", final_partial);
    printf("Correct total sum (1..%d) = %d\n", N, total);
    return 0;
}
