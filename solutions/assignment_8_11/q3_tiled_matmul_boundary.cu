#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE 16
#define CHECK_CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1;} } while(0)

__global__ void tiled_boundary(const float* A,const float* B,float* C,int n){
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    int row = blockIdx.y*TILE + threadIdx.y;
    int col = blockIdx.x*TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t=0; t<(n+TILE-1)/TILE; ++t){
        int ai = row*n + (t*TILE + threadIdx.x);
        int bi = (t*TILE + threadIdx.y)*n + col;
        As[threadIdx.y][threadIdx.x] = (row<n && t*TILE+threadIdx.x<n) ? A[ai] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (col<n && t*TILE+threadIdx.y<n) ? B[bi] : 0.0f;
        __syncthreads();
        for(int k=0;k<TILE;++k) sum += As[threadIdx.y][k]*Bs[k][threadIdx.x];
        __syncthreads();
    }
    if(row<n && col<n) C[row*n+col] = sum;
}

int main(int argc,char** argv){
    int n = (argc>1)?atoi(argv[1]):1000;
    size_t bytes=(size_t)n*n*sizeof(float);
    float *hA=(float*)malloc(bytes),*hB=(float*)malloc(bytes);
    for(int i=0;i<n*n;++i){ hA[i]=(float)(i%13); hB[i]=(float)(i%11); }
    float *dA,*dB,*dC;
    CHECK_CUDA(cudaMalloc((void**)&dA,bytes)); CHECK_CUDA(cudaMalloc((void**)&dB,bytes)); CHECK_CUDA(cudaMalloc((void**)&dC,bytes));
    CHECK_CUDA(cudaMemcpy(dA,hA,bytes,cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(dB,hB,bytes,cudaMemcpyHostToDevice));
    dim3 block(TILE,TILE), grid((n+TILE-1)/TILE,(n+TILE-1)/TILE);
    cudaEvent_t ev_start, ev_stop; CHECK_CUDA(cudaEventCreate(&ev_start)); CHECK_CUDA(cudaEventCreate(&ev_stop));
    CHECK_CUDA(cudaEventRecord(ev_start));
    tiled_boundary<<<grid,block>>>(dA,dB,dC,n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(ev_stop)); CHECK_CUDA(cudaEventSynchronize(ev_stop));
    float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms,ev_start,ev_stop));
    printf("Q3 tiled matmul boundary n=%d time=%.3f ms\\n", n, ms);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); free(hA); free(hB);
    return 0;
}
