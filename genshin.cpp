#include <array>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>

using namespace std;

using cell_t = int8_t;

// #define _DEBUG

#ifdef _DEBUG
#undef _DEBUG
#endif
#define NDEBUG

struct sparse_matrix_pair {
    int32_t index;
    cell_t value;
    sparse_matrix_pair(int32_t index, cell_t value) : index(index), value(value) {}

    bool operator<(const sparse_matrix_pair &rhs) const {
        return index < rhs.index;
    }
    bool operator>(const sparse_matrix_pair &rhs) const {
        return index > rhs.index;
    }

    sparse_matrix_pair &operator+=(const sparse_matrix_pair &rhs) {
        value += rhs.value;
        value %= 3;
        return *this;
    }
    sparse_matrix_pair &operator-=(const sparse_matrix_pair &rhs) {
        value -= rhs.value;
        value += 3;
        value %= 3;
        return *this;
    }
};

class sparse_matrix_pairs {
    vector<sparse_matrix_pair> pairs;
    cell_t y;

    void sort_indices() {
#ifdef _DEBUG
        cout << "before sort_indices: ";
        print();
        cout << endl;
#endif
        assert(!pairs.empty());
        for (int32_t i = 0; i < pairs.size(); ) {
            if (pairs[i].value == 0) {
                swap(pairs[i], pairs.back());
                pairs.pop_back();
            } else {
                ++i;
            }
        }
        assert(!pairs.empty());
        sort(pairs.begin(), pairs.end());
#ifdef _DEBUG
        cout << "after sort_indices: ";
        print();
        cout << endl;
#endif
    }

public:
    sparse_matrix_pairs(vector<sparse_matrix_pair> indices, cell_t y) : pairs(std::move(indices)), y(y) {
        assert(is_sorted(pairs.begin(), pairs.end()));
    }
    // sparse_matrix_pairs &operator-=(const sparse_matrix_pairs &rhs) {
    //     pairs.reserve(pairs.size() + rhs.pairs.size());
    //     auto lhs_it = pairs.begin();
    //     auto rhs_it = rhs.pairs.begin();
    //     while (lhs_it != pairs.end() && rhs_it != rhs.pairs.end()) {
    //         if (*lhs_it < *rhs_it) {
    //             ++lhs_it;
    //         } else if (*lhs_it > *rhs_it) {
    //             pairs.insert(lhs_it, *rhs_it);
    //             ++rhs_it;
    //         } else {
    //             lhs_it->value -= rhs_it->value;
    //             lhs_it->value += 3;
    //             lhs_it->value %= 3;
    //             ++lhs_it;
    //             ++rhs_it;
    //         }
    //     }
    //     while (rhs_it != rhs.pairs.end()) {
    //         pairs.push_back(*rhs_it);
    //         ++rhs_it;
    //     }

    //     sort_indices();
    //     return *this;
    // }

    bool operator<(const sparse_matrix_pairs &rhs) const {
        return pairs < rhs.pairs;
    }

    void normalize() {
        assert(is_sorted(pairs.begin(), pairs.end()));
        assert(!pairs.empty());
        if (pairs[0].value == 2) {
            for (auto &pair : pairs) {
                pair.value *= 2;
                pair.value %= 3;
            }
            y *= 2;
            y %= 3;
        }
        assert(pairs[0].value == 1);
    }

    // Use `(this, yi)` to eliminate `(rhs, yj)`.
    void eliminate(sparse_matrix_pairs &rhs) const {
#ifdef _DEBUG
        this->print();
        cout << " eliminates ";
        rhs.print();
#endif
        assert(this != &rhs);
        assert(!pairs.empty());
        assert(!rhs.pairs.empty());
        assert(pairs[0].value == 1);
        const int32_t i0 = 0;
        int32_t j0 = 0;
        while (pairs[i0].index > rhs.pairs[j0].index) {
            ++j0;
            if (j0 >= rhs.pairs.size()) {
#ifdef _DEBUG
                cout << " skipped" << endl;
#endif
                return;
            }
        }
        if (pairs[i0].index < rhs.pairs[j0].index) {
#ifdef _DEBUG
            cout << " skipped" << endl;
#endif
            return;
        }
        assert(pairs[i0].index == rhs.pairs[j0].index);

        const cell_t delta = 3 - rhs.pairs[j0].value;
        rhs.y += y * delta;
        rhs.y %= 3;
        rhs.pairs.erase(rhs.pairs.begin() + j0);

        int32_t i = i0 + 1;
        for (int32_t j = j0; i < pairs.size() && j < rhs.pairs.size(); ) {
            if (pairs[i].index < rhs.pairs[j].index) {
                rhs.pairs.emplace(rhs.pairs.begin() + j, pairs[i].index, pairs[i].value * delta % 3);
                ++i;
            } else if (pairs[i].index > rhs.pairs[j].index) {
                ++j;
            } else {
                // pairs[i].index == rhs.pairs[j].index
                rhs.pairs[j].value += pairs[i].value * delta;
                rhs.pairs[j].value %= 3;
                ++i;
                ++j;
            }
        }
        while (i < pairs.size()) {
            rhs.pairs.emplace_back(pairs[i].index, pairs[i].value * delta % 3);
            ++i;
        }
        
#ifdef _DEBUG
        cout << " to ";
        rhs.print();
        cout << endl;
#endif
        rhs.sort_indices();
    }

    void print() const {
        for (const auto &pair : pairs) {
            cout << '(' << pair.index << " = " << (int32_t)pair.value << ')';
        }
        cout << " = " << (int32_t)y;
    }

    cell_t get_y() const {
        return y;
    }
};

void show_matrix(const vector<sparse_matrix_pairs> &a) {
    for (int i = 0; i < a.size(); ++i) {
        cout << i << ": ";
        a[i].print();
        cout << endl;
    }
}

// #define NDEBUG

// 该函数用于求解模 3 意义下的线性方程组
//
// 参数：
//   a: 线性方程组的系数矩阵，a[i][j] 表示第 i 个方程中第 j 个未知数的系数，大小为 (n, n+1)
// 返回值：
//   一个 vector<cell_t>，表示线性方程组的解，大小为 n
template<int32_t n1, int32_t n2>
static inline vector<cell_t> solve_linear_system(const int32_t n, const vector<int32_t> &m, const vector<int32_t> &im) {
    // 方向数组
    constexpr int nl[5][2] = {{-1, 0}, {0, -1}, {0, 0}, {0, 1}, {1, 0}};

    // 创建矩阵和向量

    // 模 3 意义下的线性方程组的系数矩阵
    vector<sparse_matrix_pairs> a;
    a.reserve(n);
    assert(n1 * n2 == m.size());
    assert(n1 * n2 == im.size());

    // 填充矩阵和向量
    for (int i = 0; i < n2; ++i) {
        for (int j = 0; j < n1; ++j) {
            const int ci = im[i * n1 + j];
            assert(ci < n);
            if (ci >= 0) {
                vector<sparse_matrix_pair> pairs;
                for (const auto& direction : nl) {
                    const int i_ = i + direction[0];
                    const int j_ = j + direction[1];
                    if (i_ >= 0 && i_ < n2 && j_ >= 0 && j_ < n1) {
                        const int ci_ = im[i_ * n1 + j_];
                        assert(ci_ < n);
                        if (ci_ >= 0) {
                            pairs.emplace_back(ci_, 1);
                        }
                    }
                }
                a.push_back(sparse_matrix_pairs(std::move(pairs), 3 - m[i * n1 + j]));
            }
        }
    }

    sort(a.begin(), a.end());

#ifdef _DEBUG
    show_matrix(a);
#endif

    for (int32_t i = 0; i < n; ++i) {
        // 将第 i 个方程中第 i 个未知数的系数变为 1
        a[i].normalize();

        // 将第 j 个方程中第 i 个未知数的系数变为 0
        // #pragma omp parallel for schedule(static)
        for (int32_t j = 0; j < n; ++j) {
            if (j != i) {
                a[i].eliminate(a[j]);
            }
        }

        sort(a.begin(), a.end());

#ifdef _DEBUG
        cout << "第" << i << "次消元" << endl;
        show_matrix(a);
#endif
    }

    // 返回解
    vector<cell_t> y(n);
    for (int32_t i = 0; i < n; ++i) {
        y[i] = a[i].get_y();
    }
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

#ifdef _DEBUG
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
#ifdef _DEBUG
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