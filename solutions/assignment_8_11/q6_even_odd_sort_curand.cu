#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>

#define CHECK_CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1;} } while(0)
#define CHECK_CURAND(call) do { curandStatus_t s=(call); if(s!=CURAND_STATUS_SUCCESS){fprintf(stderr,"CURAND %s:%d %d\\n",__FILE__,__LINE__,s); return 1;} } while(0)

__global__ void even_odd(float* arr, int n, int odd){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = 2 * idx + odd;
    if(i + 1 < n && arr[i] > arr[i + 1]){
        float t = arr[i];
        arr[i] = arr[i + 1];
        arr[i + 1] = t;
    }
}

int main(int argc,char** argv){
    int n = (argc>1)?atoi(argv[1]):1<<16;
    float *d;
    CHECK_CUDA(cudaMalloc((void**)&d,(size_t)n*sizeof(float)));

    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 456ULL));
    CHECK_CURAND(curandGenerateUniform(gen, d, n));

    int threads = 256;
    int blocks = ((n / 2) + threads - 1) / threads;
    cudaEvent_t ev_start, ev_stop; CHECK_CUDA(cudaEventCreate(&ev_start)); CHECK_CUDA(cudaEventCreate(&ev_stop));

    CHECK_CUDA(cudaEventRecord(ev_start));
    for(int pass=0; pass<n; ++pass){
        even_odd<<<blocks,threads>>>(d, n, pass & 1);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaEventRecord(ev_stop)); CHECK_CUDA(cudaEventSynchronize(ev_stop));

    float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms,ev_start,ev_stop));
    printf("Q6 even-odd sort n=%d time=%.3f ms\\n",n,ms);

    curandDestroyGenerator(gen);
    cudaFree(d);
    return 0;
}
