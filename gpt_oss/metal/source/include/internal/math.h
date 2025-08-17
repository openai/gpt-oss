#pragma once

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

inline static size_t math_ceil_div(size_t numer, size_t denom) {
    return (numer + denom - 1) / denom;
}

inline static size_t math_max(size_t a, size_t b) {
    return a >= b ? a : b;
}

inline static size_t math_min(size_t a, size_t b) {
    return a < b ? a : b;
}

inline static size_t math_sub_sat(size_t a, size_t b) {
    return a > b ? a - b : 0;
}

static size_t math_round_down_po2(size_t number, size_t multiple) {
    assert(multiple != 0);
    assert((multiple & (multiple - 1)) == 0);

    return number & -multiple;
}

static size_t math_round_up_po2(size_t number, size_t multiple) {
    assert(multiple != 0);
    assert((multiple & (multiple - 1)) == 0);

    const size_t multiple_mask = multiple - 1;
    if ((number & multiple_mask) != 0) {
        number |= multiple_mask;
        number += 1;
    }
    return number;
}
