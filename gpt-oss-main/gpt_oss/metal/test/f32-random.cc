#include <gtest/gtest.h>

#include <cmath>

#include <internal/metal.hpp>
#include <internal/metal-kernels.h>
#include <internal/rng.hpp>

using gptoss::Check;
using namespace gptoss::metal;


constexpr uint64_t kSeed = UINT64_C(1019827666124465388);
constexpr uint64_t kOffset = UINT64_C(12345678901234567890);
constexpr float kMin = -1.0f;
constexpr float kMax = +1.5f;
constexpr float kScale = (kMax - kMin) * 0.5f;
constexpr float kBias = (kMin + kMax) * 0.5f;
constexpr size_t kThreadgroupSize = 128;

TEST(F32_FILL_RANDOM, single_threadgroup_single_iteration) {
    constexpr size_t num_bytes = kThreadgroupSize * 16;
    constexpr size_t num_elements = num_bytes / sizeof(uint32_t);

    Device device;
    CommandQueue command_queue{device};
    CommandBuffer command_buffer{command_queue};
    Library library{device};
    Function f32_fill_random_fn{library, "gptoss_f32_fill_random"};
    Buffer buffer{device, num_elements * sizeof(float)};

    Check(gptoss_metal_command_buffer_encode_launch_f32_fill_random(
            command_buffer.handle(),
            f32_fill_random_fn.handle(),
            /*threadgroup_size=*/kThreadgroupSize,
            /*max_threadgroups=*/1,
            /*output_buffer=*/buffer.handle(),
            /*output_offset=*/0,
            num_elements, kSeed, kOffset, kMin, kMax),
        "gptoss_metal_command_buffer_encode_launch_f32_fill_random");

    command_buffer.commit();
    command_buffer.wait_completion();

    const float* output_ptr = static_cast<const float*>(buffer.ptr());
    for (size_t i = 0; i < num_elements; i++) {
        const uint32_t ref_word = gptoss::rng::squares32(kOffset + i, kSeed);
        const float ref_float = static_cast<int32_t>(ref_word) * 0x1.0p-31f;
        const float ref_value = std::fma(ref_float, kScale, kBias);
        ASSERT_EQ(output_ptr[i], ref_value)
            << "at position " << i << " / " << num_elements;
    }
}

TEST(F32_FILL_RANDOM, single_threadgroup_multiple_iterations) {
    constexpr size_t num_iterations = 3;
    constexpr size_t num_bytes = num_iterations * kThreadgroupSize * 16;
    constexpr size_t num_elements = num_bytes / sizeof(uint32_t);

    Device device;
    CommandQueue command_queue{device};
    CommandBuffer command_buffer{command_queue};
    Library library{device};
    Function f32_fill_random_fn{library, "gptoss_f32_fill_random"};
    Buffer buffer{device, num_elements * sizeof(float)};

    Check(gptoss_metal_command_buffer_encode_launch_f32_fill_random(
            command_buffer.handle(),
            f32_fill_random_fn.handle(),
            /*threadgroup_size=*/kThreadgroupSize,
            /*max_threadgroups=*/1,
            /*output_buffer=*/buffer.handle(),
            /*output_offset=*/0,
            num_elements, kSeed, kOffset, kMin, kMax),
        "gptoss_metal_command_buffer_encode_launch_f32_fill_random");

    command_buffer.commit();
    command_buffer.wait_completion();

    const float* output_ptr = static_cast<const float*>(buffer.ptr());
    for (size_t i = 0; i < num_elements; i++) {
        const uint32_t ref_word = gptoss::rng::squares32(kOffset + i, kSeed);
        const float ref_float = static_cast<int32_t>(ref_word) * 0x1.0p-31f;
        const float ref_value = std::fma(ref_float, kScale, kBias);
        ASSERT_EQ(output_ptr[i], ref_value)
            << "at position " << i << " / " << num_elements;
    }
}

TEST(F32_FILL_RANDOM, multiple_threadgroups_multiple_iterations) {
    constexpr size_t num_iterations = 3;
    constexpr size_t num_threadgroups = 2;
    constexpr size_t num_bytes = num_iterations * num_threadgroups * kThreadgroupSize * 16;
    constexpr size_t num_elements = num_bytes / sizeof(uint32_t);

    Device device;
    CommandQueue command_queue{device};
    CommandBuffer command_buffer{command_queue};
    Library library{device};
    Function f32_fill_random_fn{library, "gptoss_f32_fill_random"};
    Buffer buffer{device, num_elements * sizeof(float)};

    Check(gptoss_metal_command_buffer_encode_launch_f32_fill_random(
            command_buffer.handle(),
            f32_fill_random_fn.handle(),
            /*threadgroup_size=*/kThreadgroupSize,
            /*max_threadgroups=*/num_threadgroups,
            /*output_buffer=*/buffer.handle(),
            /*output_offset=*/0,
            num_elements, kSeed, kOffset, kMin, kMax),
        "gptoss_metal_command_buffer_encode_launch_f32_fill_random");

    command_buffer.commit();
    command_buffer.wait_completion();

    const float* output_ptr = static_cast<const float*>(buffer.ptr());
    for (size_t i = 0; i < num_elements; i++) {
        const uint32_t ref_word = gptoss::rng::squares32(kOffset + i, kSeed);
        const float ref_float = static_cast<int32_t>(ref_word) * 0x1.0p-31f;
        const float ref_value = std::fma(ref_float, kScale, kBias);
        ASSERT_EQ(output_ptr[i], ref_value)
            << "at position " << i << " / " << num_elements;
    }
}

TEST(F32_FILL_RANDOM, excessive_threadgroups) {
    constexpr size_t num_bytes = kThreadgroupSize * 16;
    constexpr size_t num_elements = num_bytes / sizeof(uint32_t);

    Device device;
    CommandQueue command_queue{device};
    CommandBuffer command_buffer{command_queue};
    Library library{device};
    Function f32_fill_random_fn{library, "gptoss_f32_fill_random"};
    Buffer buffer{device, num_elements * sizeof(float)};

    Check(gptoss_metal_command_buffer_encode_launch_f32_fill_random(
            command_buffer.handle(),
            f32_fill_random_fn.handle(),
            /*threadgroup_size=*/kThreadgroupSize,
            /*max_threadgroups=*/2,
            /*output_buffer=*/buffer.handle(),
            /*output_offset=*/0,
            num_elements, kSeed, kOffset, kMin, kMax),
        "gptoss_metal_command_buffer_encode_launch_f32_fill_random");

    command_buffer.commit();
    command_buffer.wait_completion();

    const float* output_ptr = static_cast<const float*>(buffer.ptr());
    for (size_t i = 0; i < num_elements; i++) {
        const uint32_t ref_word = gptoss::rng::squares32(kOffset + i, kSeed);
        const float ref_float = static_cast<int32_t>(ref_word) * 0x1.0p-31f;
        const float ref_value = std::fma(ref_float, kScale, kBias);
        ASSERT_EQ(output_ptr[i], ref_value)
            << "at position " << i << " / " << num_elements;
    }
}

TEST(F32_FILL_RANDOM, nonuniform_range) {
    constexpr size_t num_iterations = 3;
    constexpr size_t num_threadgroups = 2;
    constexpr size_t num_bytes = (num_iterations * num_threadgroups + 1) * kThreadgroupSize * 16;
    constexpr size_t num_elements = num_bytes / sizeof(uint32_t);

    Device device;
    CommandQueue command_queue{device};
    CommandBuffer command_buffer{command_queue};
    Library library{device};
    Function f32_fill_random_fn{library, "gptoss_f32_fill_random"};
    Buffer buffer{device, num_elements * sizeof(float)};

    Check(gptoss_metal_command_buffer_encode_launch_f32_fill_random(
            command_buffer.handle(),
            f32_fill_random_fn.handle(),
            /*threadgroup_size=*/kThreadgroupSize,
            /*max_threadgroups=*/num_threadgroups,
            /*output_buffer=*/buffer.handle(),
            /*output_offset=*/0,
            num_elements, kSeed, kOffset, kMin, kMax),
        "gptoss_metal_command_buffer_encode_launch_f32_fill_random");

    command_buffer.commit();
    command_buffer.wait_completion();

    const float* output_ptr = static_cast<const float*>(buffer.ptr());
    for (size_t i = 0; i < num_elements; i++) {
        const uint32_t ref_word = gptoss::rng::squares32(kOffset + i, kSeed);
        const float ref_float = static_cast<int32_t>(ref_word) * 0x1.0p-31f;
        const float ref_value = std::fma(ref_float, kScale, kBias);
        ASSERT_EQ(output_ptr[i], ref_value)
            << "at position " << i << " / " << num_elements;
    }
}

TEST(F32_FILL_RANDOM, partial_range) {
    constexpr size_t num_iterations = 3;
    constexpr size_t num_threadgroups = 2;
    constexpr size_t num_bytes = (num_iterations * num_threadgroups * kThreadgroupSize + 1) * 16;
    constexpr size_t num_elements = num_bytes / sizeof(uint32_t);

    Device device;
    CommandQueue command_queue{device};
    CommandBuffer command_buffer{command_queue};
    Library library{device};
    Function f32_fill_random_fn{library, "gptoss_f32_fill_random"};
    Buffer buffer{device, num_elements * sizeof(float)};

    Check(gptoss_metal_command_buffer_encode_launch_f32_fill_random(
            command_buffer.handle(),
            f32_fill_random_fn.handle(),
            /*threadgroup_size=*/kThreadgroupSize,
            /*max_threadgroups=*/num_threadgroups,
            /*output_buffer=*/buffer.handle(),
            /*output_offset=*/0,
            num_elements, kSeed, kOffset, kMin, kMax),
        "gptoss_metal_command_buffer_encode_launch_f32_fill_random");

    command_buffer.commit();
    command_buffer.wait_completion();

    const float* output_ptr = static_cast<const float*>(buffer.ptr());
    for (size_t i = 0; i < num_elements; i++) {
        const uint32_t ref_word = gptoss::rng::squares32(kOffset + i, kSeed);
        const float ref_float = static_cast<int32_t>(ref_word) * 0x1.0p-31f;
        const float ref_value = std::fma(ref_float, kScale, kBias);
        ASSERT_EQ(output_ptr[i], ref_value)
            << "at position " << i << " / " << num_elements;
    }
}
