#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define MASK_W 3
#define TILE 16
#define CHECK_CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1;} } while(0)

__constant__ float dMask[MASK_W * MASK_W];

__global__ void conv2d_const(const float* in, float* out, int w, int h){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= w || y >= h) return;

    int r = MASK_W / 2;
    float sum = 0.0f;
    for(int j=-r; j<=r; ++j){
        for(int i=-r; i<=r; ++i){
            int xx = x + i, yy = y + j;
            if(xx >= 0 && xx < w && yy >= 0 && yy < h){
                sum += in[yy * w + xx] * dMask[(j + r) * MASK_W + (i + r)];
            }
        }
    }
    out[y * w + x] = sum;
}

int main(int argc,char** argv){
    int w=(argc>1)?atoi(argv[1]):1024;
    int h=(argc>2)?atoi(argv[2]):1024;
    size_t bytes=(size_t)w*h*sizeof(float);

    float *hIn=(float*)malloc(bytes), *hOut=(float*)malloc(bytes);
    float mask[MASK_W*MASK_W] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
    for(int i=0;i<w*h;++i) hIn[i]=(float)(i%255);

    float *dIn,*dOut;
    CHECK_CUDA(cudaMalloc((void**)&dIn,bytes)); CHECK_CUDA(cudaMalloc((void**)&dOut,bytes));
    CHECK_CUDA(cudaMemcpy(dIn,hIn,bytes,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(dMask,mask,sizeof(mask)));

    dim3 block(TILE,TILE), grid((w+TILE-1)/TILE,(h+TILE-1)/TILE);
    cudaEvent_t ev_start, ev_stop; CHECK_CUDA(cudaEventCreate(&ev_start)); CHECK_CUDA(cudaEventCreate(&ev_stop));
    CHECK_CUDA(cudaEventRecord(ev_start));
    conv2d_const<<<grid,block>>>(dIn,dOut,w,h);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(ev_stop)); CHECK_CUDA(cudaEventSynchronize(ev_stop));
    float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms,ev_start,ev_stop));
    CHECK_CUDA(cudaMemcpy(hOut,dOut,bytes,cudaMemcpyDeviceToHost));

    printf("Q8 2D convolution constant memory %dx%d time=%.3f ms\\n",w,h,ms);

    cudaFree(dIn); cudaFree(dOut); free(hIn); free(hOut);
    return 0;
}
