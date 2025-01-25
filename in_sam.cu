#include <math.h>
#include <stdio.h>
#include <malloc.h>
#include <stdint.h>

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
__global__ void kernel(d3_t src, d3_t* mir, d3_t* sen, d_t* data, int64_t mirn, int64_t senn) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < senn) {
        d_t a=0;
        d_t b=0;
        for (int64_t j = 0; j < mirn; j++) {
            d_t l = norm(mir[j] - src) + norm(mir[j] - sen[i]);
            a += cos(6.283185307179586 * 2000 * l);
            b += sin(6.283185307179586 * 2000 * l);
        }
        data[i] = sqrt(a * a + b * b);
    }
}

int main(){
    FILE* fi;
    fi = fopen("in.data", "rb");
    d3_t src;
    int64_t mirn,senn;
    d3_t* mir, * sen;

    fread(&src, 1, sizeof(d3_t), fi);
    
    fread(&mirn, 1, sizeof(int64_t), fi);
    mir = (d3_t*)malloc(mirn * sizeof(d3_t));
    fread(mir, 1, mirn * sizeof(d3_t), fi);

    fread(&senn, 1, sizeof(int64_t), fi);
    sen = (d3_t*)malloc(senn * sizeof(d3_t));
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

    kernel<<<(senn + 255) / 256, 256>>>(src, d_mir, d_sen, d_data, mirn, senn);

    cudaMemcpy(data, d_data, senn * sizeof(d_t), cudaMemcpyDeviceToHost);

    cudaFree(d_mir);
    cudaFree(d_sen);
    cudaFree(d_data);

    fi = fopen("out.data", "wb");
    fwrite(data, 1, senn * sizeof(d_t), fi);
    fclose(fi);

    return 0;
}