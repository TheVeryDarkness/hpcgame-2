#include <cassert>
#include <math.h>
#include <stdio.h>
#include <malloc.h>
#include <stdint.h>
#include <chrono>
#include <iostream>

// https://zhuanlan.zhihu.com/p/663607169
#define CHECK_CUDA(call)                                \
    do                                                  \
    {                                                   \
        const cudaError_t error_code = call;            \
        if (error_code != cudaSuccess)                  \
        {                                               \
            printf("CUDA Error:\n");                    \
            printf("    File:       %s\n", __FILE__);   \
            printf("    Line:       %d\n", __LINE__);   \
            printf("    Error code: %d\n", error_code); \
            printf("    Error text: %s\n",              \
                   cudaGetErrorString(error_code));     \
            exit(1);                                    \
        }                                               \
    } while (0)

typedef double d_t;
struct d3_t {
    d_t x, y, z;
};

__device__ d_t norm(d3_t x) {
    return sqrt(x.x * x.x + x.y * x.y + x.z * x.z);
}

__device__ d3_t operator-(d3_t a, d3_t b) {
    return {a.x-b.x,a.y-b.y,a.z-b.z};
}

/**
 * @brief 
 * 
 * @param src 
 * @param mir 
 * @param sen 
 * @param data 
 * @param mirn 
 * @param senn 
 */
template<int64_t mirn, int64_t senn>
__global__ void kernel(const d3_t src, const d3_t* mir, const d3_t* sen, d_t* data) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    {
        d_t a=0;
        d_t b=0;
        #pragma unroll
        for (int64_t j = 0; j < mirn; j++) {
            d_t l = norm(mir[j] - src) + norm(mir[j] - sen[i]);
            a += cos(6.283185307179586 * 2000 * l);
            b += sin(6.283185307179586 * 2000 * l);
        }
        data[i] = sqrt(a * a + b * b);
    }
}

int main(){
    // // These variables are used to convert occupancy to warps
    // int device;
    // cudaDeviceProp prop;

    // cudaGetDevice(&device);
    // cudaGetDeviceProperties(&prop, device);
    
    FILE* fi;
    fi = fopen("in.data", "rb");
    d3_t src;
    int64_t mirn,senn;

    fread(&src, 1, sizeof(d3_t), fi);
    
    fread(&mirn, 1, sizeof(int64_t), fi);
    assert(mirn == 1048576);
    d3_t* mir = (d3_t*)malloc(mirn * sizeof(d3_t));
    fread(mir, 1, mirn * sizeof(d3_t), fi);

    fread(&senn, 1, sizeof(int64_t), fi);
    assert(senn == 1048576);
    d3_t* sen = (d3_t*)malloc(senn * sizeof(d3_t));
    fread(sen, 1, senn * sizeof(d3_t), fi);

    fclose(fi);

    d_t* data = (d_t*)malloc(senn * sizeof(d_t));

    d3_t* d_mir, * d_sen;
    d_t* d_data;

    cudaMalloc(&d_mir, mirn * sizeof(d3_t));
    cudaMalloc(&d_sen, senn * sizeof(d3_t));
    cudaMalloc(&d_data, senn * sizeof(d_t));

    cudaMemcpy(d_mir, mir, mirn * sizeof(d3_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sen, sen, senn * sizeof(d3_t), cudaMemcpyHostToDevice);

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    const int blockSize = 256;
    const int numBlocks = (senn + blockSize - 1) / blockSize;        // Occupancy in terms of active blocks

    kernel<1048576, 1048576><<<numBlocks, blockSize>>>(src, d_mir, d_sen, d_data);

    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &numBlocks,
    //     kernel,
    //     blockSize,
    //     0
    // );

    // const int activeWarps = numBlocks * blockSize / prop.warpSize;
    // const int maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    // std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
    
    CHECK_CUDA(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << std::endl;

    cudaMemcpy(data, d_data, senn * sizeof(d_t), cudaMemcpyDeviceToHost);

    cudaFree(d_mir);
    cudaFree(d_sen);
    cudaFree(d_data);

    CHECK_CUDA(cudaGetLastError());

    fi = fopen("out.data", "wb");
    fwrite(data, 1, senn * sizeof(d_t), fi);
    fclose(fi);

    return 0;
}
