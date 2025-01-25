#include <cuda_runtime.h>
#include "./include/universal/number/posit/posit.hpp"

using fp_t = sw::universal::posit<64, 3>;

// 该函数实现 C = A * B，其中 A 是 n*m 矩阵，B 是 m*k 矩阵，C 是 n*k 矩阵
// 你需要使用 CUDA 来加速矩阵乘法
// 你可以假设 n, m, k 都是 32 的倍数
// uint64_t 是用 POSIT 来表示的浮点数。

void cuda_posit_gemm_d(const uint64_t *dA, const uint64_t *dB, uint64_t *dC,
                       int n, int m, int k) {
    const fp_t *A = (const fp_t *)dA;
    const fp_t *B = (const fp_t *)dB;
    fp_t *C = (fp_t *)dC;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            fp_t sum = 0;
            for (int l = 0; l < m; l++) {
                fp_t a = A[i * m + l];
                fp_t b = B[l * k + j];
                sum += a * b;
            }
            C[i * k + j] = sum;
        }
    }
}
