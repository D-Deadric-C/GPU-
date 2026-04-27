#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1;} } while(0)

__global__ void conv1d(const float* in, const float* mask, float* out, int n, int m){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    int r = m / 2;
    float sum = 0.0f;
    for(int k = -r; k <= r; ++k){
        int idx = i + k;
        if(idx >= 0 && idx < n) sum += in[idx] * mask[k + r];
    }
    out[i] = sum;
}

int main(int argc,char** argv){
    int n=(argc>1)?atoi(argv[1]):1<<20;
    int m=5;
    size_t bn=(size_t)n*sizeof(float), bm=(size_t)m*sizeof(float);
    float *hIn=(float*)malloc(bn), hMask[5]={1,2,3,2,1};
    for(int i=0;i<n;++i) hIn[i]=(float)(i%10);

    float *dIn,*dMask,*dOut;
    CHECK_CUDA(cudaMalloc((void**)&dIn,bn)); CHECK_CUDA(cudaMalloc((void**)&dMask,bm)); CHECK_CUDA(cudaMalloc((void**)&dOut,bn));
    CHECK_CUDA(cudaMemcpy(dIn,hIn,bn,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dMask,hMask,bm,cudaMemcpyHostToDevice));

    int threads=256, blocks=(n+threads-1)/threads;
    cudaEvent_t ev_start, ev_stop; CHECK_CUDA(cudaEventCreate(&ev_start)); CHECK_CUDA(cudaEventCreate(&ev_stop));
    CHECK_CUDA(cudaEventRecord(ev_start));
    conv1d<<<blocks,threads>>>(dIn,dMask,dOut,n,m);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(ev_stop)); CHECK_CUDA(cudaEventSynchronize(ev_stop));
    float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms,ev_start,ev_stop));
    printf("Q7 1D convolution n=%d time=%.3f ms\\n",n,ms);

    cudaFree(dIn); cudaFree(dMask); cudaFree(dOut); free(hIn);
    return 0;
}
