#include <gtest/gtest.h>

#include <cstddef>

#include "embeddings-kernel-tester.hpp"


using gptoss::EmbeddingsKernelTester;

constexpr std::size_t kThreadgroupSize = 64;


TEST(BF16_F32_EMBEDDINGS, single_token_single_tile) {
    EmbeddingsKernelTester()
        .num_channels(kThreadgroupSize)
        .threadgroup_size(kThreadgroupSize)
        .TestBF16_F32();
}

TEST(BF16_F32_EMBEDDINGS, single_token_multi_tile) {
    EmbeddingsKernelTester()
        .num_channels(kThreadgroupSize * 4 + 16)
        .threadgroup_size(kThreadgroupSize)
        .TestBF16_F32();
}

TEST(BF16_F32_EMBEDDINGS, multiple_tokens) {
    EmbeddingsKernelTester()
        .num_channels(kThreadgroupSize * 4 + 16)
        .num_tokens(3)
        .threadgroup_size(kThreadgroupSize)
        .TestBF16_F32();
}
