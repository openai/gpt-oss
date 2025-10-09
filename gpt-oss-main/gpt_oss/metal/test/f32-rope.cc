#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "rope-kernel-tester.hpp"


using gptoss::RoPEKernelTester;

constexpr float kFrequencyBase = 50000.0f;
constexpr std::uint32_t kHeadDim = 64;  // fixed in the kernel
constexpr std::uint32_t kTokenOffset = 7;


TEST(F32_ROPE, single_simdgroup) {
    RoPEKernelTester()
        .head_dim(kHeadDim)
        .num_q_heads(1)
        .num_kv_heads(0)
        .token_offset(kTokenOffset)
        .frequency_base(kFrequencyBase)
        .threadgroup_size(kHeadDim / 2)
        .TestF32();
}

TEST(F32_ROPE, single_threadgroup) {
    constexpr std::size_t threadgroup_size = 64;
    constexpr std::uint32_t num_heads = threadgroup_size / (kHeadDim / 2);

    RoPEKernelTester()
        .head_dim(kHeadDim)
        .num_q_heads(num_heads)
        .num_kv_heads(0)
        .token_offset(kTokenOffset)
        .frequency_base(kFrequencyBase)
        .threadgroup_size(threadgroup_size)
        .TestF32();
}

TEST(F32_ROPE, multiple_threadgroups) {
    constexpr std::uint32_t num_threadgroups = 3;
    constexpr std::size_t threadgroup_size = 64;
    constexpr std::uint32_t num_heads = num_threadgroups * (threadgroup_size / (kHeadDim / 2));

    RoPEKernelTester()
        .head_dim(kHeadDim)
        .num_q_heads(num_heads)
        .num_kv_heads(0)
        .token_offset(kTokenOffset)
        .frequency_base(kFrequencyBase)
        .threadgroup_size(threadgroup_size)
        .TestF32();
}

TEST(F32_ROPE, multiple_tokens) {
    constexpr std::uint32_t num_tokens = 2;
    constexpr std::uint32_t num_threadgroups = 3;
    constexpr std::size_t threadgroup_size = 64;
    constexpr std::uint32_t num_heads = num_threadgroups * (threadgroup_size / (kHeadDim / 2));

    RoPEKernelTester()
        .head_dim(kHeadDim)
        .num_tokens(2)
        .num_q_heads(num_heads)
        .num_kv_heads(0)
        .token_offset(kTokenOffset)
        .frequency_base(kFrequencyBase)
        .threadgroup_size(threadgroup_size)
        .TestF32();
}
