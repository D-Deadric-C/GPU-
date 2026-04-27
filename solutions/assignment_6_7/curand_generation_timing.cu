#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return 1; \
    } \
} while(0)

#define CHECK_CURAND(call) do { \
    curandStatus_t st = (call); \
    if (st != CURAND_STATUS_SUCCESS) { \
        fprintf(stderr, "CURAND error %s:%d: %d\\n", __FILE__, __LINE__, st); \
        return 1; \
    } \
} while(0)

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 1 << 20;
    float *d = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d, (size_t)n * sizeof(float)));

    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CURAND(curandGenerateUniform(gen, d, n));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("Generated %d random floats using CURAND in %.3f ms\\n", n, ms);

    CHECK_CURAND(curandDestroyGenerator(gen));
    CHECK_CUDA(cudaFree(d));
    return 0;
}
