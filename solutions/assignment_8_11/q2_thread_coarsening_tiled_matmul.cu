#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE 16
#define COARSE 2
#define CHECK_CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1;} } while(0)

__global__ void coarsened_matmul(const float* A,const float* B,float* C,int n){
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE*COARSE];

    int row = blockIdx.y*TILE + threadIdx.y;
    int colBase = (blockIdx.x*TILE + threadIdx.x) * COARSE;
    float sum[COARSE] = {0};

    for(int t=0; t<(n+TILE-1)/TILE; ++t){
        int aCol = t*TILE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row<n && aCol<n) ? A[row*n+aCol] : 0.0f;

        #pragma unroll
        for(int c=0;c<COARSE;++c){
            int col = colBase + c;
            int bRow = t*TILE + threadIdx.y;
            Bs[threadIdx.y][threadIdx.x*COARSE + c] = (bRow<n && col<n) ? B[bRow*n+col] : 0.0f;
        }
        __syncthreads();

        for(int k=0;k<TILE;++k){
            float a = As[threadIdx.y][k];
            #pragma unroll
            for(int c=0;c<COARSE;++c) sum[c] += a * Bs[k][threadIdx.x*COARSE + c];
        }
        __syncthreads();
    }

    if(row<n){
        #pragma unroll
        for(int c=0;c<COARSE;++c){
            int col = colBase + c;
            if(col<n) C[row*n+col] = sum[c];
        }
    }
}

int main(int argc,char** argv){
    int n = (argc>1)?atoi(argv[1]):1024;
    size_t bytes=(size_t)n*n*sizeof(float);
    float *hA=(float*)malloc(bytes), *hB=(float*)malloc(bytes);
    for(int i=0;i<n*n;++i){ hA[i]=1.0f+(i%7); hB[i]=1.0f+(i%5); }
    float *dA,*dB,*dC;
    CHECK_CUDA(cudaMalloc((void**)&dA,bytes)); CHECK_CUDA(cudaMalloc((void**)&dB,bytes)); CHECK_CUDA(cudaMalloc((void**)&dC,bytes));
    CHECK_CUDA(cudaMemcpy(dA,hA,bytes,cudaMemcpyHostToDevice)); CHECK_CUDA(cudaMemcpy(dB,hB,bytes,cudaMemcpyHostToDevice));

    dim3 block(TILE,TILE), grid((n + TILE*COARSE -1)/(TILE*COARSE), (n+TILE-1)/TILE);
    cudaEvent_t ev_start, ev_stop; CHECK_CUDA(cudaEventCreate(&ev_start)); CHECK_CUDA(cudaEventCreate(&ev_stop));
    CHECK_CUDA(cudaEventRecord(ev_start));
    coarsened_matmul<<<grid,block>>>(dA,dB,dC,n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(ev_stop)); CHECK_CUDA(cudaEventSynchronize(ev_stop));
    float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms,ev_start,ev_stop));
    printf("Q2 coarsened tiled matmul n=%d time=%.3f ms\\n",n,ms);

    cudaFree(dA); cudaFree(dB); cudaFree(dC); free(hA); free(hB);
    return 0;
}
