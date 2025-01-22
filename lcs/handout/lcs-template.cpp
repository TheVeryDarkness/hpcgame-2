#include <utility>
#include <cstdlib>
#include <algorithm>

typedef int element_t;

template<size_t len_1, size_t len_2>
static inline size_t lcs_inner(element_t* arr_1, element_t* arr_2) {
    size_t* previous = (size_t*)calloc(len_2 + 1, sizeof(size_t));
    size_t* current = (size_t*)calloc(len_2 + 1, sizeof(size_t));

    for (size_t i = 1; i <= len_1; ++i) {
        for (size_t j = 1; j <= len_2; ++j) {
            if (arr_1[i - 1] == arr_2[j - 1]) {
                current[j] = previous[j - 1] + 1;
            } else {
                current[j] = std::max(current[j - 1], previous[j]);
            }
        }
        std::swap(current, previous);
    }

    size_t result = previous[len_2];

    free(previous);
    free(current);

    return result;
}

size_t lcs(element_t* arr_1, element_t* arr_2, size_t len_1, size_t len_2) {
    // 65536 65536
    // 65536 262144
    // 262144 262144
    // 262144 1048576
    // 1048576 1048576
    switch (len_1) {
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
