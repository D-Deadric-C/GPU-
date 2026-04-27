#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1;} } while(0)

__global__ void reduce_sum(const float* in, float* out, int n){
    extern __shared__ float s[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float x = 0.0f;
    if(i < (unsigned)n) x += in[i];
    if(i + blockDim.x < (unsigned)n) x += in[i + blockDim.x];
    s[tid] = x;
    __syncthreads();
    for(unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if(tid == 0) out[blockIdx.x] = s[0];
}

int main(int argc,char** argv){
    int n = (argc>1)?atoi(argv[1]):1<<20;
    size_t bytes=(size_t)n*sizeof(float);
    float *h=(float*)malloc(bytes);
    for(int i=0;i<n;++i) h[i]=1.0f;
    float *dIn,*dTmp;
    CHECK_CUDA(cudaMalloc((void**)&dIn,bytes));
    CHECK_CUDA(cudaMemcpy(dIn,h,bytes,cudaMemcpyHostToDevice));

    int threads=256;
    int curN=n;
    cudaEvent_t ev_start, ev_stop; CHECK_CUDA(cudaEventCreate(&ev_start)); CHECK_CUDA(cudaEventCreate(&ev_stop));
    CHECK_CUDA(cudaEventRecord(ev_start));

    while(curN > 1){
        int blocks = (curN + threads*2 - 1) / (threads*2);
        CHECK_CUDA(cudaMalloc((void**)&dTmp, (size_t)blocks * sizeof(float)));
        reduce_sum<<<blocks,threads,threads*sizeof(float)>>>(dIn,dTmp,curN);
        CHECK_CUDA(cudaGetLastError());
        cudaFree(dIn);
        dIn = dTmp;
        curN = blocks;
    }

    float result=0;
    CHECK_CUDA(cudaMemcpy(&result,dIn,sizeof(float),cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(ev_stop)); CHECK_CUDA(cudaEventSynchronize(ev_stop));
    float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms,ev_start,ev_stop));
    printf("Q4 reduction sum=%f time=%.3f ms\\n",result,ms);

    cudaFree(dIn); free(h);
    return 0;
}
