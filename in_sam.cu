#include <math.h>
#include <stdio.h>
#include <malloc.h>
#include <stdint.h>
#include <cuda_fp16.h>

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

__device__ __host__ d3_t operator*(d3_t a, d_t b) {
    return {a.x*b,a.y*b,a.z*b};
}

__device__ __host__ d3_t &operator*=(d3_t &a, d_t b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
}

typedef float f_t;
struct f3_t {
    f_t x, y, z;

    __device__ __host__ f3_t(d3_t a): x(a.x), y(a.y), z(a.z) {}
    __device__ __host__ f3_t(f_t x, f_t y, f_t z): x(x), y(y), z(z) {}
};

// __device__ f_t norm(f3_t x) {
//     return sqrt(x.x * x.x + x.y * x.y + x.z * x.z);
// }

// __device__ f3_t operator-(f3_t a, f3_t b) {
//     return {a.x-b.x,a.y-b.y,a.z-b.z};
// }

__device__ __host__ f3_t operator*(f3_t a, f_t b) {
    return {a.x*b,a.y*b,a.z*b};
}

constexpr d_t coeff = 6.283185307179586 * 2000;

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
__global__ void kernel(const d3_t src, d3_t mir[senn], d3_t sen[senn], d_t data[senn]) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < senn) {
        const d3_t sen_i = sen[i];
        d_t a = 0;
        d_t b = 0;
        for (int64_t j = 0; j < mirn; j++) {
            const d3_t mir_j = mir[j];
            d_t l = norm(mir_j - src) + norm(mir_j - sen_i);

            d_t tmp_a, tmp_b;
            sincos(l, &tmp_b, &tmp_a);
            a += tmp_a;
            b += tmp_b;

            // f_t tmp_a, tmp_b;
            // sincosf(l, &tmp_b, &tmp_a);
            // a += tmp_a;
            // b += tmp_b;

            // a += cos(l);
            // b += sin(l);

            // a += (1 - 0.5 * l * l);
            // b += l;

            // a += (1 - 0.5 * l * l);
            // b += l * (1 - l * l / 6);
        }
        data[i] = sqrt(a * a + b * b);
    }
}

int main(){
    FILE* fi;
    fi = fopen("in.data", "rb");
    d3_t src;
    int64_t mirn,senn; // number of mirrors and sensors, both being 2^20
    d3_t* mir, * sen;

    fread(&src, 1, sizeof(d3_t), fi);
    
    fread(&mirn, 1, sizeof(int64_t), fi);
    mir = (d3_t*)malloc(mirn * sizeof(d3_t));
    fread(mir, 1, mirn * sizeof(d3_t), fi);
    for (int64_t i = 0; i < mirn; i++) {
        mir[i] *= coeff;
    }

    fread(&senn, 1, sizeof(int64_t), fi);
    sen = (d3_t*)malloc(senn * sizeof(d3_t));
    fread(sen, 1, senn * sizeof(d3_t), fi);
    for (int64_t i = 0; i < senn; i++) {
        sen[i] *= coeff;
    }

    fclose(fi);

    d_t* data = (d_t*)malloc(senn * sizeof(d_t));

    d3_t* d_mir, * d_sen;
    d_t* d_data;

    cudaMalloc(&d_mir, mirn * sizeof(d3_t));
    cudaMalloc(&d_sen, senn * sizeof(d3_t));
    cudaMalloc(&d_data, senn * sizeof(d_t));

    cudaMemcpy(d_mir, mir, mirn * sizeof(d3_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sen, sen, senn * sizeof(d3_t), cudaMemcpyHostToDevice);

    const int blockSize = 256;
    // assert(senn % blockSize == 0);
    const int numBlocks = senn / blockSize; 

    kernel<1048576, 1048576><<<numBlocks, blockSize>>>(src * coeff, d_mir, d_sen, d_data);

    cudaMemcpy(data, d_data, senn * sizeof(d_t), cudaMemcpyDeviceToHost);

    cudaFree(d_mir);
    cudaFree(d_sen);
    cudaFree(d_data);

    fi = fopen("out.data", "wb");
    fwrite(data, 1, senn * sizeof(d_t), fi);
    fclose(fi);

    return 0;
}