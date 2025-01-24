#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>

using namespace std;

// 该函数用于求解模 3 意义下的线性方程组
//
// 参数：
//   a: 线性方程组的系数矩阵，a[i][j] 表示第 i 个方程中第 j 个未知数的系数，大小为 (n, n+1)
// 返回值：
//   一个 vector<int32_t>，表示线性方程组的解，大小为 n
static inline vector<int32_t> solve_linear_system(const int32_t n, const int32_t n1, const int32_t n2, const vector<int32_t> &m, const vector<int32_t> &im) {
    // 方向数组
    constexpr int nl[5][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}, {0, 0}};

    // 创建矩阵和向量
    vector<vector<int32_t>> a(n, vector<int32_t>(n + 1, 0));

    // 填充矩阵和向量
    for (int i = 0; i < n2; ++i) {
        for (int j = 0; j < n1; ++j) {
            int ci = im[i * n1 + j];
            if (ci >= 0) {
                // y[ci] = 3 - m[i * n1 + j];
                a[ci][n] = 3 - m[i * n1 + j];
                for (const auto& direction : nl) {
                    int i_ = i + direction[0];
                    int j_ = j + direction[1];
                    if (i_ >= 0 && i_ < n2 && j_ >= 0 && j_ < n1) {
                        int ci_ = im[i_ * n1 + j_];
                        if (ci_ >= 0) {
                            a[ci_][ci] = 1;
                        }
                    }
                }
            }
        }
    }

    for (int32_t i = 0; i < n; ++i) {
        // 找到第 i 个方程中第 i 个未知数的系数不为 0 的方程
        int32_t j = i;
        while (j < n && a[j][i] == 0) {
            ++j;
        }
        assert(j < n);

        // 交换第 i 个方程和第 j 个方程
        if (i != j) {
            swap(a[i], a[j]);
        }

        // 将第 i 个方程中第 i 个未知数的系数变为 1
        if (a[i][i] == 2) {
            for (int32_t j = i; j <= n; ++j) {
                a[i][j] *= 2;
                a[i][j] %= 3;
            }
        }

        // 将第 j 个方程中第 i 个未知数的系数变为 0
        for (int32_t j = 0; j < n; ++j) {
            if (j != i) {
                for (int32_t k = i; k <= n; ++k) {
                    a[j][k] -= a[i][k];
                    a[j][k] = (a[j][k] + 3) % 3;
                }
            }
        }
    }
}

int main() {
    // 读取数据
    ifstream infile("in.data", ios::binary);
    assert(infile.is_open());

    int32_t n1, n2;
    infile.read(reinterpret_cast<char*>(&n1), sizeof(int32_t));
    infile.read(reinterpret_cast<char*>(&n2), sizeof(int32_t));

    vector<int32_t> m(n2 * n1);
    infile.read(reinterpret_cast<char*>(m.data()), n2 * n1 * sizeof(int32_t));
    infile.close();

    vector<int32_t> im(n2 * n1, -1);
    // 非零元素的个数
    int count = 0;

    // 填充im数组
    for (int i = 0; i < n2; ++i) {
        for (int j = 0; j < n1; ++j) {
            if (m[i * n1 + j] != 0) {
                im[i * n1 + j] = count++;
            }
        }
    }

    // 求解线性方程
    // 注意：C++没有内置的求解线性方程组的功能，您可以使用Eigen或其他库。
    // 这里我们假设有一个函数solve_linear_system来处理这个问题。
    // vector<int32_t> x_t(ci);
    vector<int32_t> x_t = solve_linear_system(count, n1, n2, m, im);

    // 填充输出数组
    vector<int32_t> x(n2 * n1, 0);
    for (int i = 0; i < n2; ++i) {
        for (int j = 0; j < n1; ++j) {
            int ci = im[i * n1 + j];
            if (ci >= 0) {
                x[i * n1 + j] = x_t[ci];
            }
        }
    }

    // 写入输出文件
    ofstream outfile("out.data", ios::binary);
    assert(outfile.is_open());
    outfile.write(reinterpret_cast<char*>(x.data()), x.size() * sizeof(int32_t));
    outfile.close();

    return 0;
}