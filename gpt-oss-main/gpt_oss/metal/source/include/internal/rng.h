#pragma once

#include <stdint.h>

inline static uint32_t rng_squares32(uint64_t offset, uint64_t seed) {
    const uint64_t y = offset * seed;
    const uint64_t z = y + seed;

    /* Round 1 */
    uint64_t x = y * y + y;
    x = (x >> 32) | (x << 32);

    /* Round 2 */
    x = x * x + z;
    x = (x >> 32) | (x << 32);

    /* Round 3 */
    x = x * x + y;
    x = (x >> 32) | (x << 32);

    /* Round 4 */
    x = x * x + z;
    return (uint32_t) (x >> 32);
}
