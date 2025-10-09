#include <gtest/gtest.h>

#include <cmath>
#include <ios>

#include <internal/metal.hpp>
#include <internal/metal-kernels.h>

using gptoss::Check;
using namespace gptoss::metal;

constexpr size_t kThreadgroupSize = 32;


static float fp4e2m1_to_fp32[16] = {
    +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

TEST(MF4_F32_CONVERT, single_threadgroup_single_iteration) {
    constexpr size_t num_blocks = kThreadgroupSize;
    constexpr size_t num_elements = num_blocks * 32;
    constexpr size_t num_bytes = num_elements / 2;

    Device device;
    CommandQueue command_queue{device};
    CommandBuffer command_buffer{command_queue};
    Library library{device};
    Function mf4_f32_convert_fn{library, "gptoss_mf4_f32_convert"};
    Buffer block_buffer{device, num_bytes};
    Buffer scale_buffer{device, num_blocks * sizeof(uint8_t)};
    Buffer output_buffer{device, num_elements * sizeof(float)};

    uint8_t* block_ptr = static_cast<uint8_t*>(block_buffer.ptr());
    std::memset(block_ptr, 0, num_bytes);
    for (size_t b = 0; b < num_blocks; b++) {
        for (size_t i = 0; i < 32; i++) {
            const uint8_t nibble = (i + b) & 0x0F;
            const uint8_t byte = nibble << ((i % 2) * 4);
            block_ptr[b * 16 + i / 2] |= byte;
        }
    }

    uint8_t* scale_ptr = static_cast<uint8_t*>(scale_buffer.ptr());
    for (size_t b = 0; b < num_blocks; b++) {
        scale_ptr[b] = 127 - b;
    }

    Check(gptoss_metal_command_buffer_encode_launch_mf4_f32_convert(
            command_buffer.handle(),
            mf4_f32_convert_fn.handle(),
            /*threadgroup_size=*/kThreadgroupSize,
            /*max_threadgroups=*/1,
            block_buffer.handle(),
            scale_buffer.handle(),
            output_buffer.handle(),
            num_elements),
        "gptoss_metal_command_buffer_encode_launch_mf4_f32_convert");

    command_buffer.commit();
    command_buffer.wait_completion();

    const float* output_ptr = static_cast<const float*>(output_buffer.ptr());
    for (size_t b = 0; b < num_blocks; b++) {
        for (size_t i = 0; i < 32; i++) {
            const uint8_t byte = block_ptr[b * 16 + i / 2];
            const uint8_t nibble = (byte >> ((i % 2) * 4)) & 0x0F;
            const float ref_scale = std::ldexp(1.0f, static_cast<int>(scale_ptr[b]) - 127);
            const float ref_value = fp4e2m1_to_fp32[nibble] * ref_scale;
            ASSERT_EQ(output_ptr[b * 32 + i], ref_value)
                << "at position " << i << " / 32"
                << ", block " << b << " / " << num_blocks
                << ", FP4e2m1 value " << std::hex << uint32_t(nibble);
        }
    }
}

TEST(MF4_F32_CONVERT, multiple_threadgroups_multiple_iterations) {
    constexpr size_t num_threadgroups = 2;
    constexpr size_t num_blocks = num_threadgroups * (kThreadgroupSize + 1);
    constexpr size_t num_elements = num_blocks * 32;
    constexpr size_t num_bytes = num_elements / 2;

    Device device;
    CommandQueue command_queue{device};
    CommandBuffer command_buffer{command_queue};
    Library library{device};
    Function mf4_f32_convert_fn{library, "gptoss_mf4_f32_convert"};
    Buffer block_buffer{device, num_bytes};
    Buffer scale_buffer{device, num_blocks * sizeof(uint8_t)};
    Buffer output_buffer{device, num_elements * sizeof(float)};

    uint8_t* block_ptr = static_cast<uint8_t*>(block_buffer.ptr());
    std::memset(block_ptr, 0, num_bytes);
    for (size_t b = 0; b < num_blocks; b++) {
        for (size_t i = 0; i < 32; i++) {
            const uint8_t nibble = (i + b) & 0x0F;
            const uint8_t byte = nibble << ((i % 2) * 4);
            block_ptr[b * 16 + i / 2] |= byte;
        }
    }

    uint8_t* scale_ptr = static_cast<uint8_t*>(scale_buffer.ptr());
    for (size_t b = 0; b < num_blocks; b++) {
        scale_ptr[b] = 200 - b;
    }

    Check(gptoss_metal_command_buffer_encode_launch_mf4_f32_convert(
            command_buffer.handle(),
            mf4_f32_convert_fn.handle(),
            /*threadgroup_size=*/kThreadgroupSize,
            /*max_threadgroups=*/num_threadgroups,
            block_buffer.handle(),
            scale_buffer.handle(),
            output_buffer.handle(),
            num_elements),
        "gptoss_metal_command_buffer_encode_launch_mf4_f32_convert");

    command_buffer.commit();
    command_buffer.wait_completion();

    const float* output_ptr = static_cast<const float*>(output_buffer.ptr());
    for (size_t b = 0; b < num_blocks; b++) {
        for (size_t i = 0; i < 32; i++) {
            const uint8_t byte = block_ptr[b * 16 + i / 2];
            const uint8_t nibble = (byte >> ((i % 2) * 4)) & 0x0F;
            const float ref_scale = std::ldexp(1.0f, static_cast<int>(scale_ptr[b]) - 127);
            const float ref_value = fp4e2m1_to_fp32[nibble] * ref_scale;
            ASSERT_EQ(output_ptr[b * 32 + i], ref_value)
                << "at position " << i << " / 32"
                << ", block " << b << " / " << num_blocks
                << ", FP4e2m1 value " << std::hex << uint32_t(nibble);
        }
    }
}
