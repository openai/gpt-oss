#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <internal/datatype.hpp>
#include <internal/metal.hpp>
#include <internal/metal-kernels.h>


namespace gptoss {

class RoPEKernelTester {
public:
    RoPEKernelTester() { }

    RoPEKernelTester(const RoPEKernelTester&) = delete;
    RoPEKernelTester(RoPEKernelTester&&) = delete;
    RoPEKernelTester& operator=(const RoPEKernelTester&) = delete;
    RoPEKernelTester& operator=(RoPEKernelTester&&) = delete;

    [[nodiscard]]
    RoPEKernelTester& threadgroup_size(std::size_t threadgroup_size) {
        threadgroup_size_ = threadgroup_size;
        return *this;
    }

    std::size_t threadgroup_size() const {
        return threadgroup_size_;
    }

    [[nodiscard]]
    RoPEKernelTester& head_dim(std::uint32_t head_dim) {
        head_dim_ = head_dim;
        return *this;
    }

    std::uint32_t head_dim() const {
        return head_dim_;
    }

    [[nodiscard]]
    RoPEKernelTester& num_q_heads(std::uint32_t num_q_heads) {
        num_q_heads_ = num_q_heads;
        return *this;
    }

    std::uint32_t num_q_heads() const {
        return num_q_heads_;
    }

    [[nodiscard]]
    RoPEKernelTester& num_kv_heads(std::uint32_t num_kv_heads) {
        num_kv_heads_ = num_kv_heads;
        return *this;
    }

    std::uint32_t num_kv_heads() const {
        return num_kv_heads_;
    }

    std::uint32_t num_qk_heads() const {
        return num_q_heads() + num_kv_heads();
    }

    std::uint32_t num_qkv_heads() const {
        return num_q_heads() + 2 * num_kv_heads();
    }

    [[nodiscard]]
    RoPEKernelTester& num_tokens(std::uint32_t num_tokens) {
        num_tokens_ = num_tokens;
        return *this;
    }

    std::uint32_t num_tokens() const {
        return num_tokens_;
    }

    [[nodiscard]]
    RoPEKernelTester& token_offset(std::uint32_t token_offset) {
        token_offset_ = token_offset;
        return *this;
    }

    std::uint32_t token_offset() const {
        return token_offset_;
    }

    [[nodiscard]]
    RoPEKernelTester& frequency_base(float frequency_base) {
        frequency_base_ = frequency_base;
        return *this;
    }

    float frequency_base() const {
        return frequency_base_;
    }

    void Validate() const {
        ASSERT_NE(head_dim(), 0);
        ASSERT_EQ(head_dim() % 2, 0);
        ASSERT_NE(num_q_heads(), 0);
        ASSERT_NE(num_tokens(), 0);
    }

    void TestF32() const {
        Validate();

        metal::Buffer activations_buffer{device_, (num_tokens() * num_qkv_heads() + num_qk_heads()) * head_dim() * sizeof(float)};
        metal::Buffer ref_activations_buffer{device_, (num_tokens() * num_qkv_heads() + num_qk_heads()) * head_dim() * sizeof(float)};
        // KV cache buffer layout: [num_kv_heads][max_tokens][2 (K,V)][head_dim]
        const std::uint32_t max_tokens = num_tokens();
        const std::uint32_t kv_heads_for_alloc = std::max<std::uint32_t>(1, num_kv_heads());
        const std::size_t kv_bytes = static_cast<std::size_t>(kv_heads_for_alloc) * max_tokens * 2 * head_dim() * sizeof(float);
        metal::Buffer kv_cache_buffer{device_, kv_bytes};
        metal::Buffer control_buffer{device_, sizeof(gptoss_control)};
        std::memset(control_buffer.ptr(), 0, sizeof(gptoss_control));

        metal::CommandBuffer command_buffer{command_queue_};

        command_buffer.encode_launch_f32_fill_random(
            f32_fill_random_fn_,
            /*threadgroup_size=*/0,
            /*max_threadgroups=*/kFillRandomMaxThreadgroups,
            /*output_buffer=*/activations_buffer,
            /*output_offset=*/0,
            (num_tokens() * num_qkv_heads() + num_qk_heads()) * head_dim(),
            kSeed, /*offset=*/0, /*min=*/-1.0f, /*max=*/1.0);

        command_buffer.encode_launch_f32_fill_random(
            f32_fill_random_fn_,
            /*threadgroup_size=*/0,
            /*max_threadgroups=*/kFillRandomMaxThreadgroups,
            /*output_buffer=*/ref_activations_buffer,
            /*output_offset=*/0,
            (num_tokens() * num_qkv_heads() + num_qk_heads()) * head_dim(),
            kSeed, /*offset=*/0, /*min=*/-1.0f, /*max=*/1.0);

        Check(gptoss_metal_command_buffer_encode_launch_f32_rope(
                command_buffer.handle(),
                f32_rope_fn_.handle(),
                threadgroup_size(),
                activations_buffer.handle(),
                /*activations_offset=*/0,
                kv_cache_buffer.handle(),
                /*kv_offset=*/0,
                control_buffer.handle(),
                /*control_offset=*/0,
                frequency_base(),
                /*interpolation_scale=*/1.0f,
                /*yarn_offset=*/0.0f,
                /*yarn_scale=*/1.0f,
                /*yarn_multiplier=*/1.0f,
                /*num_tokens=*/num_tokens(),
                /*num_q_heads=*/num_q_heads(),
                /*num_kv_heads=*/num_kv_heads(),
                head_dim(),
                /*max_tokens=*/max_tokens,
                /*token_offset=*/token_offset()),
            "gptoss_metal_command_buffer_encode_launch_f32_rope");

        command_buffer.commit();
        command_buffer.wait_completion();

        const float* ref_activations_ptr = static_cast<const float*>(ref_activations_buffer.ptr());
        const float* activations_ptr = static_cast<const float*>(activations_buffer.ptr());
        const float* kv_ptr = static_cast<const float*>(kv_cache_buffer.ptr());
        for (std::uint32_t t = 0; t < num_tokens(); t++) {
            // Validate rotated Q written in-place in activations
            for (std::uint32_t h = 0; h < num_q_heads(); h++) {
                for (std::uint32_t d = 0; d < head_dim(); d += 2) {
                    const double inv_freq = 1.0 /
                        std::pow(static_cast<double>(frequency_base()), static_cast<double>(d) / static_cast<double>(head_dim()));
                    const double phi = static_cast<double>(t + token_offset()) * inv_freq;
                    const double cos_phi = std::cos(phi);
                    const double sin_phi = std::sin(phi);
                    const double real = static_cast<double>(ref_activations_ptr[(t * num_qkv_heads() + h) * head_dim() + d]);
                    const double imag = static_cast<double>(ref_activations_ptr[(t * num_qkv_heads() + h) * head_dim() + d + 1]);
                    const double ref_real = real * cos_phi - imag * sin_phi;
                    const double ref_imag = real * sin_phi + imag * cos_phi;
                    ASSERT_NEAR(
                            static_cast<double>(activations_ptr[(t * num_qkv_heads() + h) * head_dim() + d]),
                            ref_real,
                            std::abs(ref_real) * 1.0e-4)
                        << "at token " << t << " / " << num_tokens();
                    ASSERT_NEAR(
                            static_cast<double>(activations_ptr[(t * num_qkv_heads() + h) * head_dim() + d + 1]),
                            ref_imag,
                            std::abs(ref_imag) * 1.0e-4)
                        << "at token " << t << " / " << num_tokens();

                }
            }
        }
    }

private:
    static constexpr uint64_t kSeed{UINT64_C(1019827666124465388)};
    static constexpr std::size_t kFillRandomMaxThreadgroups = 10;

    metal::Device device_{};
    metal::CommandQueue command_queue_{device_};
    metal::Library library_{device_};
    metal::Function f32_fill_random_fn_{library_, "gptoss_f32_fill_random"};
    metal::Function f32_rope_fn_{library_, "gptoss_f32_rope"};
    std::size_t threadgroup_size_{32};
    std::uint32_t head_dim_{64};
    std::uint32_t num_q_heads_{1};
    std::uint32_t num_kv_heads_{1};
    std::uint32_t num_tokens_{1};
    std::uint32_t token_offset_{0};
    float frequency_base_{50000.0f};
};

}  // namespace gptoss
