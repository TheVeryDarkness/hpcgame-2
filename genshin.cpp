#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>

using namespace std;

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
    int ci = 0;

    // 填充im数组
    for (int i = 0; i < n2; ++i) {
        for (int j = 0; j < n1; ++j) {
            if (m[i * n1 + j] != 0) {
                im[i * n1 + j] = ci++;
            }
        }
    }

    // 方向数组
    constexpr static inline int nl[5][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}, {0, 0}};

    // 创建矩阵和向量
    vector<vector<int32_t>> a(ci, vector<int32_t>(ci, 0));
    vector<int32_t> y(ci, 0);

    // 填充矩阵和向量
    for (int i = 0; i < n2; ++i) {
        for (int j = 0; j < n1; ++j) {
            int ci = im[i * n1 + j];
            if (ci >= 0) {
                y[ci] = 3 - m[i * n1 + j];
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

    // 求解线性方程
    // 注意：C++没有内置的求解线性方程组的功能，您可以使用Eigen或其他库。
    // 这里我们假设有一个函数solve_linear_system来处理这个问题。
    vector<int32_t> x_t(ci);
    // x_t = solve_linear_system(a, y);

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