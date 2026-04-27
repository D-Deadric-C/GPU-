#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>

#define TILE 16
#define CHECK_CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1;} } while(0)
#define CHECK_CURAND(call) do { curandStatus_t s=(call); if(s!=CURAND_STATUS_SUCCESS){fprintf(stderr,"CURAND %s:%d %d\\n",__FILE__,__LINE__,s); return 1;} } while(0)

__global__ void tiled_matmul(const float* A,const float* B,float* C,int n){
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    int row = blockIdx.y*TILE + threadIdx.y;
    int col = blockIdx.x*TILE + threadIdx.x;
    float sum = 0.0f;
    for(int t=0; t<(n+TILE-1)/TILE; ++t){
        int aCol = t*TILE + threadIdx.x;
        int bRow = t*TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row<n && aCol<n) ? A[row*n+aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow<n && col<n) ? B[bRow*n+col] : 0.0f;
        __syncthreads();
        for(int k=0;k<TILE;++k) sum += As[threadIdx.y][k]*Bs[k][threadIdx.x];
        __syncthreads();
    }
    if(row<n && col<n) C[row*n+col]=sum;
}

int main(int argc,char** argv){
    int n = (argc>1)?atoi(argv[1]):1024;
    size_t bytes = (size_t)n*n*sizeof(float);
    float *dA,*dB,*dC;
    CHECK_CUDA(cudaMalloc((void**)&dA,bytes));
    CHECK_CUDA(cudaMalloc((void**)&dB,bytes));
    CHECK_CUDA(cudaMalloc((void**)&dC,bytes));

    curandGenerator_t g;
    CHECK_CURAND(curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(g, 2026ULL));
    CHECK_CURAND(curandGenerateUniform(g,dA,n*n));
    CHECK_CURAND(curandGenerateUniform(g,dB,n*n));

    dim3 block(TILE,TILE), grid((n+TILE-1)/TILE,(n+TILE-1)/TILE);
    cudaEvent_t ev_start, ev_stop; CHECK_CUDA(cudaEventCreate(&ev_start)); CHECK_CUDA(cudaEventCreate(&ev_stop));
    CHECK_CUDA(cudaEventRecord(ev_start));
    tiled_matmul<<<grid,block>>>(dA,dB,dC,n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(ev_stop)); CHECK_CUDA(cudaEventSynchronize(ev_stop));
    float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms,ev_start,ev_stop));
    printf("Q1 tiled matmul CURAND n=%d time=%.3f ms\\n", n, ms);

    curandDestroyGenerator(g);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
