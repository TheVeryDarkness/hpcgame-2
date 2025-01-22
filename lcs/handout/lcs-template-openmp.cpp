#include <utility>
#include <cstdlib>
#include <algorithm>

typedef int element_t;

// dp[i][j] = length of the LCS of arr_1[0..i] and arr_2[0..j]
// dp[i][j] = dp[i - 1][j - 1] + 1 if arr_1[i] == arr_2[j]
//          = max(dp[i - 1][j], dp[i][j - 1]) otherwise

template<size_t len_1, size_t len_2>
static inline size_t lcs_inner(element_t arr_1[len_1], element_t arr_2[len_2]) {
    static_assert(len_1 <= len_2);
    static_assert(len_1 >= 2);
    size_t* buf_2 = (size_t*)calloc(len_2 + 1, sizeof(size_t));
    size_t* buf_1 = (size_t*)calloc(len_2 + 1, sizeof(size_t));
    size_t* buf_0 = (size_t*)calloc(len_2 + 1, sizeof(size_t));

    // At the end of the t-th iteration:
    // buf_0[x] now represents the length of the LCS of dp[x][t - x]
    // buf_1[x] now represents the length of the LCS of dp[x][t - 1 - x]
    // buf_2[x] now represents the length of the LCS of dp[x][t - 2 - x]

    // See the following diagram for the meaning of x and t (cell values are x):
    //
    //       3
    //     2 2
    //   1 1 1
    // 0 0 0
    //     ^buf_2
    //   ^buf_1
    // ^buf_0, t = 3
    for (size_t t = 2; t <= len_1; ++t) {
        #pragma omp parallel for schedule(static)
        for (size_t x = 1; x < t; ++x) {
            const size_t i = x;
            const size_t j = t - x;
            if (arr_1[i - 1] == arr_2[j - 1]) {
                buf_0[x] = buf_2[x - 1] + 1;
            } else {
                buf_0[x] = std::max(buf_1[x], buf_1[x - 1]);
            }
        }
        std::swap(buf_1, buf_2);
        std::swap(buf_0, buf_1);
    }

    // See the following diagram for the meaning of x and t (cell values are x):
    //
    //       3 ? ?
    //     2 2 ?
    //   1 1 1
    // 0 0 0
    //     ^buf_2
    //   ^buf_1
    // ^buf_0, t = 6
    for (size_t t = len_1 + 1; t <= len_2; ++t) {
        #pragma omp parallel for schedule(static)
        for (size_t x = 1; x < len_1; ++x) {
            const size_t i = x;
            const size_t j = t - x;
            if (arr_1[i - 1] == arr_2[j - 1]) {
                buf_0[x] = buf_2[x - 1] + 1;
            } else {
                buf_0[x] = std::max(buf_1[x], buf_1[x - 1]);
            }
        }
        std::swap(buf_1, buf_2);
        std::swap(buf_0, buf_1);
    }

    // See the following diagram for the meaning of x and t (cell values are x):
    //
    //     2 ? ?
    //   1 2 ?
    // 0 1 1
    // ? ?
    // ?
    //     ^buf_2
    //   ^buf_1
    // ^buf_0, t = 9
    for (size_t t = len_2 + 1; t <= len_1 + len_2; ++t) {
        #pragma omp parallel for schedule(static)
        for (size_t x = 1; x < len_2; ++x) {
            const size_t i = t - len_2 + x;
            const size_t j = len_2 - x;
            if (arr_1[i - 1] == arr_2[j - 1]) {
                buf_0[x] = buf_2[x - 1] + 1;
            } else {
                buf_0[x] = std::max(buf_1[x], buf_1[x + 1]);
            }
        }
        std::swap(buf_1, buf_2);
        std::swap(buf_0, buf_1);
    }

    size_t result = buf_0[0];

    free(buf_2);
    free(buf_1);
    free(buf_0);

    return result;
}

size_t lcs(element_t* arr_1, element_t* arr_2, size_t len_1, size_t len_2) {
    // 65536 65536
    // 65536 262144
    // 262144 262144
    // 262144 1048576
    // 1048576 1048576
    switch (len_1) {
    case 256: // For testing purposes
        if (len_2 == 256)
            return lcs_inner<256, 256>(arr_1, arr_2);
        else return 0;
    case 65536:
        switch (len_2) {
        case 65536:
            return lcs_inner<65536, 65536>(arr_1, arr_2);
        case 262144:
            return lcs_inner<65536, 262144>(arr_1, arr_2);
        default:
            return 0;
        }
    case 262144:
        switch (len_2) {
        case 262144:
            return lcs_inner<262144, 262144>(arr_1, arr_2);
        case 1048576:
            return lcs_inner<262144, 1048576>(arr_1, arr_2);
        default:
            return 0;
        }
    case 1048576:
        switch (len_2) {
        case 1048576:
            return lcs_inner<1048576, 1048576>(arr_1, arr_2);
        default:
            return 0;
        }
    default:
        return 0;
    }
}
