#pragma once

#include <stdint.h>

#include <internal/macros.h>


typedef struct GPTOSS_DENSELY_PACKED_STRUCTURE {
    GPTOSS_ALIGN(2) uint16_t bits;
} gptoss_bfloat16;
static_assert(sizeof(gptoss_bfloat16) == 2, "bfloat16 size is not 2 bytes");


typedef struct GPTOSS_DENSELY_PACKED_STRUCTURE {
    GPTOSS_ALIGN(2) uint16_t bits;
} gptoss_float16;
static_assert(sizeof(gptoss_float16) == 2, "float16 size is not 2 bytes");


typedef struct GPTOSS_DENSELY_PACKED_STRUCTURE {
    GPTOSS_ALIGN(1) uint8_t bits;
} gptoss_float8ue8m0;
static_assert(sizeof(gptoss_float8ue8m0) == 1, "gptoss_float8ue8m0 size is not 1 bytes");


typedef struct GPTOSS_DENSELY_PACKED_STRUCTURE {
    GPTOSS_ALIGN(1) uint8_t bits;
} gptoss_float8e5m2;
static_assert(sizeof(gptoss_float8e5m2) == 1, "float8e5m2 size is not 1 bytes");


typedef struct GPTOSS_DENSELY_PACKED_STRUCTURE {
    GPTOSS_ALIGN(1) uint8_t bits;
} gptoss_float8e4m3;
static_assert(sizeof(gptoss_float8e4m3) == 1, "gptoss_float8e4m3 size is not 1 bytes");


typedef struct GPTOSS_DENSELY_PACKED_STRUCTURE {
    GPTOSS_ALIGN(1) uint8_t bits;
} gptoss_float4e2m1x2;
static_assert(sizeof(gptoss_float4e2m1x2) == 1, "gptoss_float4e2m1x2 size is not 1 bytes");
