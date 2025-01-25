#include <cassert>
#include <math.h>
#include <stdio.h>
#include <malloc.h>
#include <stdint.h>
#include <chrono>
#include <iostream>
#include <cuda_fp16.h>

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

typedef float f_t;
typedef double d_t;
// // https://blog.csdn.net/bruce_0712/article/details/65444997
// struct __align__(16) d3_t {
//     d_t x, y, z;
// };

struct d3_t {
    d_t x, y, z;
};

struct f3_t {
    f_t x, y, z;

    __device__ f3_t(d3_t d) : x(d.x), y(d.y), z(d.z) {}
    __device__ f3_t(f_t x, f_t y, f_t z) : x(x), y(y), z(z) {}
    f3_t() = default;
};

// __device__ d_t norm(d3_t x) {
//     return sqrt(x.x * x.x + x.y * x.y + x.z * x.z);
// }

// __device__ d3_t operator-(d3_t a, d3_t b) {
//     return {a.x-b.x,a.y-b.y,a.z-b.z};
// }

static inline __device__ f_t sub_norm(f_t x, f_t y, f_t z, f3_t base) {
    const f_t dx = x - base.x;
    const f_t dy = y - base.y;
    const f_t dz = z - base.z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

template<int64_t mirn, int64_t senn>
__global__ void kernel(const d3_t src, const f_t* mir_x, const f_t* mir_y, const f_t* mir_z, const d3_t* sen, d_t* data) {

    static_assert(mirn == 1048576);
    static_assert(senn == 1048576);
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const f3_t sen_i = sen[i];

    {
        f_t a=0;
        f_t b=0;
        // #pragma unroll
        for (int64_t j = 0; j < mirn; j++) {
            const f_t mir_x_j = mir_x[j];
            const f_t mir_y_j = mir_y[j];
            const f_t mir_z_j = mir_z[j];
            // d_t l = norm(mir[j] - src) + norm(mir[j] - sen_i);
            f_t l = sub_norm(mir_x_j, mir_y_j, mir_z_j, src) + sub_norm(mir_x_j, mir_y_j, mir_z_j, sen_i);
            a += cos(6.283185307179586 * 2000 * l);
            b += sin(6.283185307179586 * 2000 * l);
        }
        data[i] = sqrt(a * a + b * b);
    }
}

// // 每个线程块共享内存的大小
// constexpr static inline int64_t SHARED_MEM_SIZE = 1024;

// template<int64_t mirn, int64_t senn>
// __global__ void kernel(const d3_t src, const d3_t* mir, const d3_t* sen, d_t* data) {
//     __shared__ d3_t shared_mir[SHARED_MEM_SIZE]; // 使用共享内存缓存 mir 数据

//     static_assert(mirn == 1048576);
//     static_assert(senn == 1048576);
//     static_assert(mirn % SHARED_MEM_SIZE == 0);
//     int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

//     d_t a = 0;
//     d_t b = 0;
//     // 每次加载部分 mir 数据到共享内存
//     for (int64_t tile = 0; tile < mirn; tile += SHARED_MEM_SIZE) {
//         int64_t idx = tile + threadIdx.x;

//         if (idx < mirn) {
//             shared_mir[threadIdx.x] = mir[idx];
//         }
//         __syncthreads();

//         #pragma unroll
//         for (int64_t j = 0; j < SHARED_MEM_SIZE; j++) {
//             const d_t l = norm(shared_mir[j] - src) + norm(shared_mir[j] - sen[i]);
//             a += cos(6.283185307179586 * 2000 * l);
//             b += sin(6.283185307179586 * 2000 * l);
//         }
//         __syncthreads();
//     }
//     // 计算最终结果
//     data[i] = sqrt(a * a + b * b);
// }

int main(){
    // // These variables are used to convert occupancy to warps
    // int device;
    // cudaDeviceProp prop;

    // cudaGetDevice(&device);
    // cudaGetDeviceProperties(&prop, device);
    
    FILE* fi;
    fi = fopen("in.data", "rb");
    double src[3];
    int64_t mirn,senn;

    constexpr uint8_t d3_size = sizeof(double) * 3;

    fread(src, 1, d3_size, fi);

    const d3_t src_d3 = {src[0], src[1], src[2]};

    fread(&mirn, 1, sizeof(int64_t), fi);
    assert(mirn == 1048576);

    // d3_t* mir;
    // CHECK_CUDA(cudaMallocHost(&mir, mirn * sizeof(d3_t)));

    f_t* mir_x, * mir_y, * mir_z;
    CHECK_CUDA(cudaMallocHost(&mir_x, mirn * sizeof(f_t)));
    CHECK_CUDA(cudaMallocHost(&mir_y, mirn * sizeof(f_t)));
    CHECK_CUDA(cudaMallocHost(&mir_z, mirn * sizeof(f_t)));

    for (int i = 0; i < mirn; i++) {
        double mir[3];
        fread(mir, 1, d3_size, fi);
        mir_x[i] = mir[0];
        mir_y[i] = mir[1];
        mir_z[i] = mir[2];
    }
    // for (int i = 0; i < mirn; i++) {
    //     fread(&mir[i], 1, d3_size, fi);
    // }
    // fread(mir, 1, mirn * sizeof(d3_t), fi);

    fread(&senn, 1, sizeof(int64_t), fi);
    assert(senn == 1048576);
    d3_t* sen;
    CHECK_CUDA(cudaMallocHost(&sen, senn * sizeof(d3_t)));
    // for (int i = 0; i < senn; i++) {
    //     fread(&sen[i], 1, d3_size, fi);
    // }
    fread(sen, 1, senn * sizeof(d3_t), fi);

    fclose(fi);

    // d3_t* d_mir;
    f_t* d_mir_x, *d_mir_y, *d_mir_z;
    d3_t * d_sen;
    d_t* d_data;

    CHECK_CUDA(cudaMalloc(&d_data, senn * sizeof(d_t)));
    // CHECK_CUDA(cudaMalloc(&d_mir, mirn * sizeof(d3_t)));
    CHECK_CUDA(cudaMalloc(&d_sen, senn * sizeof(d3_t)));

    CHECK_CUDA(cudaMalloc(&d_mir_x, mirn * sizeof(f_t)));
    CHECK_CUDA(cudaMalloc(&d_mir_y, mirn * sizeof(f_t)));
    CHECK_CUDA(cudaMalloc(&d_mir_z, mirn * sizeof(f_t)));

    // CHECK_CUDA(cudaMemcpy(d_mir, mir, mirn * sizeof(d3_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mir_x, mir_x, mirn * sizeof(f_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mir_y, mir_y, mirn * sizeof(f_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mir_z, mir_z, mirn * sizeof(f_t), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_sen, sen, senn * sizeof(d3_t), cudaMemcpyHostToDevice));

    // std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    const int blockSize = 256;
    const int numBlocks = (senn + blockSize - 1) / blockSize;        // Occupancy in terms of active blocks

    kernel<1048576, 1048576><<<numBlocks, blockSize>>>(src_d3, d_mir_x, d_mir_y, d_mir_z, d_sen, d_data);

    // CHECK_CUDA(cudaDeviceSynchronize());

    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
    //           << std::endl;

    d_t* data;
    CHECK_CUDA(cudaMallocHost(&data, senn * sizeof(d_t)));

    CHECK_CUDA(cudaMemcpy(data, d_data, senn * sizeof(d_t), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaGetLastError());

    // cudaFree(d_mir);
    // cudaFree(d_sen);
    // cudaFree(d_data);

    fi = fopen("out.data", "wb");
    fwrite(data, 1, senn * sizeof(d_t), fi);
    fclose(fi);

    // cudaFreeHost(mir);
    // cudaFreeHost(sen);
    // cudaFreeHost(data);

    return 0;
}
