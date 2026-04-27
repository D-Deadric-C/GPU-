#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>

#define BLOCK 256
#define CHECK_CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1;} } while(0)
#define CHECK_CURAND(call) do { curandStatus_t s=(call); if(s!=CURAND_STATUS_SUCCESS){fprintf(stderr,"CURAND %s:%d %d\\n",__FILE__,__LINE__,s); return 1;} } while(0)

__global__ void odd_even_pass(float* a, int n, int phase){
    int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x) + phase;
    if(i + 1 < n && a[i] > a[i+1]){
        float t = a[i]; a[i] = a[i+1]; a[i+1] = t;
    }
}

int main(int argc,char** argv){
    int n = (argc>1)?atoi(argv[1]):1<<16;
    float *d;
    CHECK_CUDA(cudaMalloc((void**)&d,(size_t)n*sizeof(float)));

    curandGenerator_t g;
    CHECK_CURAND(curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(g, 123ULL));
    CHECK_CURAND(curandGenerateUniform(g,d,n));

    int threads=BLOCK;
    int blocks=(n/2 + threads - 1)/threads;
    cudaEvent_t ev_start, ev_stop; CHECK_CUDA(cudaEventCreate(&ev_start)); CHECK_CUDA(cudaEventCreate(&ev_stop));
    CHECK_CUDA(cudaEventRecord(ev_start));
    for(int pass=0; pass<n; ++pass){
        odd_even_pass<<<blocks,threads>>>(d,n,pass & 1);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaEventRecord(ev_stop)); CHECK_CUDA(cudaEventSynchronize(ev_stop));
    float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms,ev_start,ev_stop));
    printf("Q5 bubble/odd-even style sort n=%d time=%.3f ms\\n",n,ms);

    curandDestroyGenerator(g);
    cudaFree(d);
    return 0;
}
