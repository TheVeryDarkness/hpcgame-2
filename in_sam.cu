#include <cassert>
#include <math.h>
#include <stdio.h>
#include <malloc.h>
#include <stdint.h>
#include <chrono>
#include <iostream>

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
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

// 将 src 存储为常量内存以优化加载
__constant__ d3_t d_src;

// 每个线程块共享内存的大小
#define SHARED_MEM_SIZE 1024

template<int64_t mirn>
__global__ void kernel(const d3_t* mir, const d3_t* sen, d_t* data, int64_t senn) {
    __shared__ d3_t shared_mir[SHARED_MEM_SIZE]; // 使用共享内存缓存 mir 数据

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // 检查线程是否越界
    if (i >= senn) return;

    d_t a = 0;
    d_t b = 0;

    // 每次加载部分 mir 数据到共享内存
    for (int64_t tile = 0; tile < mirn; tile += SHARED_MEM_SIZE) {
        int64_t idx = tile + threadIdx.x;

        if (idx < mirn) {
            shared_mir[threadIdx.x] = mir[idx];
        }
        __syncthreads();

        // 遍历当前 tile 的共享内存数据
        for (int64_t j = 0; j < SHARED_MEM_SIZE && (tile + j) < mirn; j++) {
            d_t l = norm(shared_mir[j] - d_src) + norm(shared_mir[j] - sen[i]);
            d_t angle = 12566.3706143592 * l; // 2π * 2000
            a += cos(angle);
            b += sin(angle);
        }
        __syncthreads();
    }

    // 计算最终结果
    data[i] = sqrt(a * a + b * b);
}

int main() {
    FILE* fi;
    fi = fopen("in.data", "rb");
    d3_t src;
    int64_t mirn, senn;

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

    CHECK_CUDA(cudaMalloc(&d_mir, mirn * sizeof(d3_t)));
    CHECK_CUDA(cudaMalloc(&d_sen, senn * sizeof(d3_t)));
    CHECK_CUDA(cudaMalloc(&d_data, senn * sizeof(d_t)));

    CHECK_CUDA(cudaMemcpy(d_mir, mir, mirn * sizeof(d3_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sen, sen, senn * sizeof(d3_t), cudaMemcpyHostToDevice));

    // 将 src 加载到常量内存
    CHECK_CUDA(cudaMemcpyToSymbol(d_src, &src, sizeof(d3_t)));

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    int blockSize;
    int minGridSize;
    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel<1048576>, 0, 0));

    int numBlocks = (senn + blockSize - 1) / blockSize;

    kernel<1048576><<<numBlocks, blockSize>>>(d_mir, d_sen, d_data, senn);

    CHECK_CUDA(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << std::endl;

    CHECK_CUDA(cudaMemcpy(data, d_data, senn * sizeof(d_t), cudaMemcpyDeviceToHost));

    cudaFree(d_mir);
    cudaFree(d_sen);
    cudaFree(d_data);

    CHECK_CUDA(cudaGetLastError());

    fi = fopen("out.data", "wb");
    fwrite(data, 1, senn * sizeof(d_t), fi);
    fclose(fi);

    free(mir);
    free(sen);
    free(data);

    return 0;
}