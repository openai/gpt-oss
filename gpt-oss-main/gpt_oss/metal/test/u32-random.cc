#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "fill-random-kernel-tester.hpp"


using gptoss::FillRandomKernelTester;

constexpr std::size_t kThreadgroupSize = 128;

TEST(U32_FILL_RANDOM, single_threadgroup_single_iteration) {
    FillRandomKernelTester()
        .num_elements(kThreadgroupSize)
        .threadgroup_size(kThreadgroupSize)
        .max_threadgroups(1)
        .TestU32();
}

TEST(U32_FILL_RANDOM, single_threadgroup_multiple_iterations) {
    constexpr std::size_t num_iterations = 3;

    FillRandomKernelTester()
        .num_elements(num_iterations * kThreadgroupSize)
        .threadgroup_size(kThreadgroupSize)
        .max_threadgroups(1)
        .TestU32();
}

TEST(U32_FILL_RANDOM, multiple_threadgroups_multiple_iterations) {
    constexpr std::size_t num_iterations = 3;
    constexpr std::size_t num_threadgroups = 2;

    FillRandomKernelTester()
        .num_elements(num_iterations * num_threadgroups * kThreadgroupSize)
        .threadgroup_size(kThreadgroupSize)
        .max_threadgroups(num_threadgroups)
        .TestU32();
}

TEST(U32_FILL_RANDOM, excessive_threadgroups) {
    FillRandomKernelTester()
        .num_elements(kThreadgroupSize)
        .threadgroup_size(kThreadgroupSize)
        .max_threadgroups(2)
        .TestU32();
}

TEST(U32_FILL_RANDOM, nonuniform_range) {
    constexpr std::size_t num_iterations = 3;
    constexpr std::size_t num_threadgroups = 2;

    FillRandomKernelTester()
        .num_elements((num_iterations * num_threadgroups + 1) * kThreadgroupSize)
        .threadgroup_size(kThreadgroupSize)
        .max_threadgroups(num_threadgroups)
        .TestU32();
}

TEST(U32_FILL_RANDOM, partial_range) {
    constexpr std::size_t num_iterations = 3;
    constexpr std::size_t num_threadgroups = 2;

    FillRandomKernelTester()
        .num_elements(num_iterations * num_threadgroups * kThreadgroupSize + 1)
        .threadgroup_size(kThreadgroupSize)
        .max_threadgroups(num_threadgroups)
        .TestU32();
}
