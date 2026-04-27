#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return 1; \
    } \
} while(0)

__global__ void mat_add(const float* A, const float* B, float* C, int n2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n2) C[i] = A[i] + B[i];
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 2048;
    int n2 = n * n;
    size_t bytes = (size_t)n2 * sizeof(float);

    float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes);
    for (int i = 0; i < n2; ++i) { hA[i] = 1.0f; hB[i] = 2.0f; }

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc((void**)&dA, bytes));
    CHECK_CUDA(cudaMalloc((void**)&dB, bytes));
    CHECK_CUDA(cudaMalloc((void**)&dC, bytes));
    CHECK_CUDA(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    int thread_options[] = {64, 128, 256, 512, 1024};

    for (int t = 0; t < 5; ++t) {
        int threads = thread_options[t];
        int blocks = (n2 + threads - 1) / threads;

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        mat_add<<<blocks, threads>>>(dA, dB, dC, n2);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        printf("threads/block=%d -> %.3f ms\\n", threads, ms);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB);
    return 0;
}
