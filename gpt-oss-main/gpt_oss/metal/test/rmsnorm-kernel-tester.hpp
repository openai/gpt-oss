#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <internal/datatype.hpp>
#include <internal/metal.hpp>
#include <internal/metal-kernels.h>


namespace gptoss {

class RMSNormKernelTester {
public:
    RMSNormKernelTester() { }

    RMSNormKernelTester(const RMSNormKernelTester&) = delete;
    RMSNormKernelTester(RMSNormKernelTester&&) = delete;
    RMSNormKernelTester& operator=(const RMSNormKernelTester&) = delete;
    RMSNormKernelTester& operator=(RMSNormKernelTester&&) = delete;

    [[nodiscard]]
    RMSNormKernelTester& num_channels(std::uint32_t num_channels) {
        num_channels_ = num_channels;
        return *this;
    }

    std::uint32_t num_channels() const {
        return num_channels_;
    }

    [[nodiscard]]
    RMSNormKernelTester& num_tokens(std::uint32_t num_tokens) {
        num_tokens_ = num_tokens;
        return *this;
    }

    std::uint32_t num_tokens() const {
        return num_tokens_;
    }

    [[nodiscard]]
    RMSNormKernelTester& epsilon(float epsilon) {
        epsilon_ = epsilon;
        return *this;
    }

    float epsilon() const {
        return epsilon_;
    }

    void Validate() const {
        ASSERT_NE(num_channels(), 0);
        ASSERT_NE(num_tokens(), 0);
        ASSERT_GE(epsilon(), 0.0f);
    }

    void TestF32_BF16W() const {
        Validate();

        metal::Buffer input_buffer{device_, num_tokens() * num_channels() * sizeof(float)};
        metal::Buffer weight_buffer{device_, num_channels() * sizeof(gptoss_bfloat16)};
        metal::Buffer output_buffer{device_, num_tokens() * num_channels() * sizeof(float)};
        metal::Buffer control_buffer{device_, sizeof(gptoss_control)};
        std::memset(control_buffer.ptr(), 0, sizeof(gptoss_control));

        metal::CommandBuffer command_buffer{command_queue_};

        command_buffer.encode_launch_f32_fill_random(
            f32_fill_random_fn_,
            /*threadgroup_size=*/0,
            /*max_threadgroups=*/kFillRandomMaxThreadgroups,
            /*output_buffer=*/input_buffer, /*output_offset=*/0,
            num_channels(), kSeed, /*offset=*/0, /*min=*/-1.0f, /*max=*/1.0);

        command_buffer.encode_launch_bf16_fill_random(
            bf16_fill_random_fn_,
            /*threadgroup_size=*/0,
            /*max_threadgroups=*/kFillRandomMaxThreadgroups,
            /*output_buffer=*/weight_buffer, /*output_offset=*/0,
            num_channels(), kSeed + 1, /*offset=*/0, /*min=*/-1.0f, /*max=*/1.0);

        Check(gptoss_metal_command_buffer_encode_launch_f32_bf16w_rmsnorm(
                command_buffer.handle(),
                f32_bf16w_rmsnorm_fn_.handle(),
                input_buffer.handle(),
                /*input_offset=*/0,
                weight_buffer.handle(),
                /*weight_offset=*/0,
                output_buffer.handle(),
                /*output_offset=*/0,
                control_buffer.handle(),
                /*control_offset=*/0,
                num_tokens(),
                num_channels(),
                epsilon()),
            "gptoss_metal_command_buffer_encode_launch_f32_bf16w_rmsnorm");

        command_buffer.commit();
        command_buffer.wait_completion();

        const float* input_ptr = static_cast<const float*>(input_buffer.ptr());
        const gptoss_bfloat16* weight_ptr = static_cast<const gptoss_bfloat16*>(weight_buffer.ptr());
        const float* output_ptr = static_cast<const float*>(output_buffer.ptr());
        for (std::uint32_t t = 0; t < num_tokens(); t++) {
            double sumsq = 0.0;
            for (std::uint32_t c = 0; c < num_channels(); c++) {
                const double val = static_cast<double>(input_ptr[t * num_channels() + c]);
                sumsq = std::fma(val, val, sumsq);
            }
            const double avgsq = sumsq / static_cast<double>(num_channels());
            const double scale = 1.0 / std::sqrt(avgsq + epsilon());
            for (std::uint32_t c = 0; c < num_channels(); c++) {
                const double input_val = upcast<double>(input_ptr[t * num_channels() + c]);
                const double weight_val = upcast<double>(weight_ptr[c]);
                const double ref_output = scale * input_val * weight_val;
                const double output = upcast<double>(output_ptr[t * num_channels() + c]);
                ASSERT_NEAR(output, ref_output, 1.0e-5 * std::abs(ref_output))
                    << "at channel " << c << " / " << num_channels() << ", token " << t << " / " << num_tokens()
                    << ", input " << input_val << ", weight " << weight_val << ", scale " << scale;
            }
        }
    }

private:
    static constexpr std::uint64_t kSeed{UINT64_C(1019827666124465388)};
    static constexpr std::size_t kFillRandomMaxThreadgroups = 10;

    metal::Device device_{};
    metal::CommandQueue command_queue_{device_};
    metal::Library library_{device_};
    metal::Function f32_fill_random_fn_{library_, "gptoss_f32_fill_random"};
    metal::Function bf16_fill_random_fn_{library_, "gptoss_bf16_fill_random"};
    metal::Function f32_bf16w_rmsnorm_fn_{library_, "gptoss_f32_bf16w_rmsnorm"};
    std::uint32_t num_tokens_{1};
    std::uint32_t num_channels_{1};
    float epsilon_{1.0e-5f};
};

}  // namespace gptoss
