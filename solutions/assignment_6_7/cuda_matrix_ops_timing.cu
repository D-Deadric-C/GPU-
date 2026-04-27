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

__global__ void mat_add(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = n * n;
    if (idx < size) C[idx] = A[idx] + B[idx];
}

__global__ void mat_mul(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 1024;
    size_t bytes = (size_t)n * n * sizeof(float);

    float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes), *hC = (float*)malloc(bytes);
    if (!hA || !hB || !hC) return 1;

    for (int i = 0; i < n * n; ++i) {
        hA[i] = (float)(i % 100) / 10.0f;
        hB[i] = (float)((i + 7) % 100) / 10.0f;
    }

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc((void**)&dA, bytes));
    CHECK_CUDA(cudaMalloc((void**)&dB, bytes));
    CHECK_CUDA(cudaMalloc((void**)&dC, bytes));
    CHECK_CUDA(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int threads = 256;
    int blocks = (n * n + threads - 1) / threads;

    CHECK_CUDA(cudaEventRecord(start));
    mat_add<<<blocks, threads>>>(dA, dB, dC, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float add_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&add_ms, start, stop));

    dim3 t2(16, 16);
    dim3 b2((n + t2.x - 1) / t2.x, (n + t2.y - 1) / t2.y);

    CHECK_CUDA(cudaEventRecord(start));
    mat_mul<<<b2, t2>>>(dA, dB, dC, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float mul_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&mul_ms, start, stop));

    CHECK_CUDA(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    printf("Matrix size: %d x %d\\n", n, n);
    printf("GPU matrix addition time: %.3f ms\\n", add_ms);
    printf("GPU matrix multiplication time: %.3f ms\\n", mul_ms);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
