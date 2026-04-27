#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return 1; \
    } \
} while(0)

int main() {
    int count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&count));
    printf("CUDA device count: %d\\n", count);

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp p;
        CHECK_CUDA(cudaGetDeviceProperties(&p, i));
        printf("\\nDevice %d: %s\\n", i, p.name);
        printf("  Compute capability: %d.%d\\n", p.major, p.minor);
        printf("  Global memory: %.2f GB\\n", p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocessors: %d\\n", p.multiProcessorCount);
        printf("  Max threads/block: %d\\n", p.maxThreadsPerBlock);
        printf("  Warp size: %d\\n", p.warpSize);
    }

    return 0;
}
