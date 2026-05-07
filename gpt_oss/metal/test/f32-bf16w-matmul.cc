#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "matmul-kernel-tester.hpp"


using gptoss::MatMulKernelTester;

constexpr size_t kSimdgroupSize = 32;  // fixed in the kernel

TEST(F32_BF16W_MATMUL, single_simdgroup_single_iteration) {
    MatMulKernelTester()
        .num_rows(1)
        .num_cols(kSimdgroupSize * 4)
        .threadgroup_size(kSimdgroupSize)
        .TestF32_BF16W();
}

TEST(F32_BF16W_MATMUL, single_simdgroup_multiple_iteration) {
    MatMulKernelTester()
        .num_rows(1)
        .num_cols((2 * kSimdgroupSize + 1) * 4)
        .threadgroup_size(kSimdgroupSize)
        .TestF32_BF16W();
}

TEST(F32_BF16W_MATMUL, single_threadgroup) {
    constexpr std::size_t threadgroup_size = 2 * kSimdgroupSize;

    MatMulKernelTester()
        .num_rows(threadgroup_size / kSimdgroupSize)
        .num_cols((2 * kSimdgroupSize + 1) * 4)
        .threadgroup_size(threadgroup_size)
        .TestF32_BF16W();
}

TEST(F32_BF16W_MATMUL, multiple_threadgroups) {
    constexpr std::size_t threadgroup_size = 2 * kSimdgroupSize;
    constexpr std::uint32_t num_threadgroups = 3;

    MatMulKernelTester()
        .num_rows(num_threadgroups * threadgroup_size / kSimdgroupSize)
        .num_cols((2 * kSimdgroupSize + 1) * 4)
        .threadgroup_size(threadgroup_size)
        .TestF32_BF16W();
}

TEST(F32_BF16W_MATMUL, multiple_tokens) {
    constexpr std::size_t threadgroup_size = 2 * kSimdgroupSize;
    constexpr std::uint32_t num_threadgroups = 3;

    MatMulKernelTester()
        .num_rows(num_threadgroups * threadgroup_size / kSimdgroupSize)
        .num_cols((2 * kSimdgroupSize + 1) * 4)
        .num_tokens(2)
        .threadgroup_size(threadgroup_size)
        .TestF32_BF16W();
}

TEST(F32_BF16W_DENSE_MATMUL_QKV, seq_len_1024) {
    MatMulKernelTester()
        .num_tokens(1024)
        .num_rows(5120)
        .num_cols(2880)
        .TestF32_BF16W(
            MatMulKernelTester::MatMulKernelType::PREFILL_QKV_OPTIMIZED);
}

TEST(F32_BF16W_DENSE_MATMUL_ATTN_OUTPUT, seq_len_1024) {
    MatMulKernelTester()
        .num_tokens(1024)
        .num_rows(2880)
        .num_cols(4096)
        .TestF32_BF16W(
            MatMulKernelTester::MatMulKernelType::PREFILL_ATTN_OUTPUT_OPTIMIZED);
}

TEST(F32_BF16W_DENSE_MATMUL_MLP_GATE, seq_len_1024) {
    MatMulKernelTester()
        .num_tokens(1024)
        .num_rows(128)
        .num_cols(2880)
        .TestF32_BF16W(
            MatMulKernelTester::MatMulKernelType::PREFILL_MLP_GATE_OPTIMIZED);
}