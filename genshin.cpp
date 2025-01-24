#include <iostream>
#include <fstream>
#include <vector>
#include <valarray>
#include <cassert>

using namespace std;

using cell_t = int8_t;

#define NDEBUG
#ifndef NDEBUG
static inline void show_matrix(const vector<vector<cell_t>> &a, const vector<cell_t> &y) {
    for (int i = 0; i < a.size(); ++i) {
        for (int j = 0; j < a[i].size(); ++j) {
            cout << a[i][j] << " ";
        }
        cout << "= " << y[i] << endl;
    }
}
#endif

// 该函数用于求解模 3 意义下的线性方程组
//
// 参数：
//   a: 线性方程组的系数矩阵，a[i][j] 表示第 i 个方程中第 j 个未知数的系数，大小为 (n, n+1)
// 返回值：
//   一个 vector<cell_t>，表示线性方程组的解，大小为 n
template<int32_t n1, int32_t n2>
static inline vector<cell_t> solve_linear_system(const int32_t n, const vector<int32_t> &m, const vector<int32_t> &im) {
    // 方向数组
    constexpr int nl[5][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}, {0, 0}};

    // 创建矩阵和向量

    // 模 3 意义下的线性方程组的系数矩阵
    vector<valarray<cell_t>> a(n, valarray<cell_t>(n));
    vector<cell_t> y(n, 0);
    assert(n1 * n2 == m.size());
    assert(n1 * n2 == im.size());

    // 填充矩阵和向量
    for (int i = 0; i < n2; ++i) {
        for (int j = 0; j < n1; ++j) {
            const int ci = im[i * n1 + j];
            assert(ci < n);
            if (ci >= 0) {
                y[ci] = 3 - m[i * n1 + j];
                for (const auto& direction : nl) {
                    const int i_ = i + direction[0];
                    const int j_ = j + direction[1];
                    if (i_ >= 0 && i_ < n2 && j_ >= 0 && j_ < n1) {
                        const int ci_ = im[i_ * n1 + j_];
                        assert(ci_ < n);
                        if (ci_ >= 0) {
                            a[ci_][ci] = 1;
                        }
                    }
                }
            }
        }
    }

#ifndef NDEBUG
    show_matrix(a, y);
#endif

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
            swap(y[i], y[j]);
        }

        // 将第 i 个方程中第 i 个未知数的系数变为 1
        if (a[i][i] == 2) {
            for (int32_t j = i; j < n; ++j) {
                a[i][j] *= 2;
                a[i][j] %= 3;
            }
            y[i] *= 2;
            y[i] %= 3;
        }
        assert(a[i][i] == 1);

        // 将第 j 个方程中第 i 个未知数的系数变为 0
        #pragma omp parallel for schedule(static)
        for (int32_t j = 0; j < n; ++j) {
            if (j != i) {
                if (a[j][i] == 2) {
                    a[j] += a[i];
                    a[j] %= 3;
                    y[j] += y[i];
                    y[j] %= 3;
                } else if (a[j][i] == 1) {
                    a[j] += a[i];
                    a[j] += a[i];
                    a[j] %= 3;
                    y[j] += 2 * y[i];
                    y[j] %= 3;
                }
            }
        }

#ifndef NDEBUG
        cout << "第" << i << "次消元" << endl;
        show_matrix(a, y);
#endif
    }

    // 返回解
    return y;
}

template<int32_t n1, int32_t n2>
static inline void solve(ifstream &infile) {
    vector<int32_t> m(n2 * n1);
    infile.read(reinterpret_cast<char*>(m.data()), n2 * n1 * sizeof(int32_t));
    infile.close();

    // 非零元素对应的未知数编号
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
    vector<cell_t> x_t = solve_linear_system<n1, n2>(count, m, im);
    assert(x_t.size() == count);

#ifndef NDEBUG
    for (int i = 0; i < count; ++i) {
        cout << x_t[i] << ' ';
    }
    cout << endl;
#endif

    // 填充输出数组
    vector<int32_t> x(n2 * n1, 0);
    for (int i = 0; i < n2; ++i) {
        for (int j = 0; j < n1; ++j) {
            int ci = im[i * n1 + j];
            if (ci >= 0) {
                x[i * n1 + j] = x_t[ci];
#ifndef NDEBUG
                cout << "x[" << i << "][" << j << "] = " << x[i * n1 + j] << endl;
#endif
            }
        }
    }

    // 写入输出文件
    ofstream outfile("out.data", ios::binary);
    assert(outfile.is_open());
    outfile.write(reinterpret_cast<char*>(x.data()), x.size() * sizeof(int32_t));
    outfile.close();
}

int main() {
    // 读取数据
    ifstream infile("in.data", ios::binary);
    assert(infile.is_open());

    int32_t n1, n2;
    infile.read(reinterpret_cast<char*>(&n1), sizeof(int32_t));
    infile.read(reinterpret_cast<char*>(&n2), sizeof(int32_t));

    if (n1 == 4) {
        solve<4, 4>(infile);
    } else if (n1 == 512) {
        if (n2 == 512) {
            solve<512, 512>(infile);
        } else if (n2 == 1024) {
            solve<512, 1024>(infile);
        }
    } else if (n1 == 128) {
        if (n2 == 128) {
            solve<128, 128>(infile);
        }
    }
    return 0;
}