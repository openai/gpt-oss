#pragma once

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include <internal/datatype.hpp>
#include <internal/metal.hpp>
#include <internal/metal-kernels.h>


namespace gptoss {

class EmbeddingsKernelTester {
public:
    EmbeddingsKernelTester() { }

    EmbeddingsKernelTester(const EmbeddingsKernelTester&) = delete;
    EmbeddingsKernelTester(EmbeddingsKernelTester&&) = delete;
    EmbeddingsKernelTester& operator=(const EmbeddingsKernelTester&) = delete;
    EmbeddingsKernelTester& operator=(EmbeddingsKernelTester&&) = delete;

    [[nodiscard]]
    EmbeddingsKernelTester& num_channels(std::uint32_t num_channels) {
        num_channels_ = num_channels;
        return *this;
    }

    std::uint32_t num_channels() const {
        return num_channels_;
    }

    [[nodiscard]]
    EmbeddingsKernelTester& num_tokens(std::uint32_t num_tokens) {
        num_tokens_ = num_tokens;
        return *this;
    }

    std::uint32_t num_tokens() const {
        return num_tokens_;
    }

    std::uint32_t vocabulary_size() const {
        return num_tokens() + 1;
    }

    [[nodiscard]]
    EmbeddingsKernelTester& threadgroup_size(std::size_t threadgroup_size) {
        threadgroup_size_ = threadgroup_size;
        return *this;
    }

    std::size_t threadgroup_size() const {
        return threadgroup_size_;
    }

    void Validate() const {
        ASSERT_NE(num_channels(), 0);
        ASSERT_NE(num_tokens(), 0);
        ASSERT_NE(threadgroup_size(), 0);
        ASSERT_EQ(threadgroup_size() % 32, 0);
    }

    void TestBF16_F32() const {
        Validate();

        metal::CommandBuffer command_buffer{command_queue_};
        metal::Buffer token_buffer{device_, sizeof(std::uint32_t)};
        metal::Buffer weight_buffer{device_, vocabulary_size() * num_channels() * sizeof(gptoss_bfloat16)};
        metal::Buffer output_buffer{device_, num_channels() * sizeof(float)};
        metal::Buffer control_buffer{device_, sizeof(gptoss_control)};
        std::memset(control_buffer.ptr(), 0, sizeof(gptoss_control));

        std::uint32_t* token_ptr = static_cast<std::uint32_t*>(token_buffer.ptr());
        for (std::uint32_t t = 0; t < num_tokens(); t++) {
            token_ptr[t] = t + 1;
        }

        Check(gptoss_metal_command_buffer_encode_launch_bf16_f32_embeddings(
                command_buffer.handle(),
                bf16_f32_embeddings_fn.handle(),
                threadgroup_size(),
                token_buffer.handle(),
                /*token_offset=*/0,
                weight_buffer.handle(),
                /*weight_offset=*/0,
                output_buffer.handle(),
                /*output_offset=*/0,
                control_buffer.handle(),
                /*control_offset=*/0,
                num_tokens(),
                num_channels()),
            "gptoss_metal_command_buffer_encode_launch_bf16_f32_embeddings");

        command_buffer.commit();
        command_buffer.wait_completion();

        const gptoss_bfloat16* weight_ptr = static_cast<const gptoss_bfloat16*>(weight_buffer.ptr());
        const float* output_ptr = static_cast<const float*>(output_buffer.ptr());
        for (std::uint32_t t = 0; t < num_tokens(); t++) {
            const std::uint32_t token = token_ptr[t];
            for (std::uint32_t i = 0; i < num_channels(); i++) {
                const gptoss_bfloat16 input_val = weight_ptr[token * num_channels() + i];
                const float ref_output = upcast<float>(input_val);
                const float output = output_ptr[t * num_channels() + i];
                ASSERT_EQ(output, ref_output)
                    << "at token " << t << ", position " << i << " / " << num_channels() << ", input " << std::uint32_t(input_val.bits);
            }
        }
    }

private:
    metal::Device device_{};
    metal::CommandQueue command_queue_{device_};
    metal::Library library_{device_};
    metal::Function bf16_f32_embeddings_fn{library_, "gptoss_bf16_f32_embeddings"};
    std::uint32_t num_tokens_{1};
    std::uint32_t num_channels_{1};
    std::size_t threadgroup_size_{32};
};

}  // namespace gptoss
