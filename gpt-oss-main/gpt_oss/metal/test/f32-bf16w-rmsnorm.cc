#include <gtest/gtest.h>

#include <cstdint>

#include "rmsnorm-kernel-tester.hpp"


using gptoss::RMSNormKernelTester;

constexpr std::uint32_t kThreadgroupSize = 1024;  // fixed in the kernel
constexpr std::uint32_t kVectorSize = 4;  // fixed in the kernel

TEST(F32_BF16W_RMSNORM, single_iteration) {
    RMSNormKernelTester()
        .num_channels(kThreadgroupSize)
        .TestF32_BF16W();
}

TEST(F32_BF16W_RMSNORM, multiple_iterations) {
    RMSNormKernelTester()
        .num_channels(kThreadgroupSize * 2)
        .TestF32_BF16W();
}

TEST(F32_BF16W_RMSNORM, partial_iteration) {
    RMSNormKernelTester()
        .num_channels(kThreadgroupSize * 2 + kVectorSize)
        .TestF32_BF16W();
}

TEST(F32_BF16W_RMSNORM, multiple_tokens) {
    RMSNormKernelTester()
        .num_tokens(3)
        .num_channels(kThreadgroupSize * 2 + kVectorSize)
        .TestF32_BF16W();
}
