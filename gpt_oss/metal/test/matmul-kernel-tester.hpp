#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <internal/datatype.hpp>
#include <internal/metal.hpp>
#include <internal/metal-kernels.h>

namespace gptoss {

template <typename T>
::testing::AssertionResult
IsNearAbsRel(const char* a_expr, const char* b_expr, const char* abs_expr,
             const char* rel_expr, T a, T b, T abs_tol, T rel_tol = 1.0) {

    using std::abs;
    if (!std::isfinite(a) || !std::isfinite(b)) {
        return ::testing::AssertionFailure()
               << "Non-finite value(s): " << a_expr << "=" << a << ", " << b_expr
               << "=" << b;
        // At least one of abs_tol and rel_tol must be provided
    }
    const T diff = abs(a - b);
    const T rel = rel_tol * std::max(abs(a), abs(b));
    const T thr = std::max(abs_tol, rel);

    if (diff <= thr)
        return ::testing::AssertionSuccess();

    return ::testing::AssertionFailure()
           << a_expr << " vs " << b_expr << " differ by " << diff
           << " > max(abs_tol=" << abs_tol << ", rel_tol*max(|a|,|b|)=" << rel
           << ") with " << abs_expr << "=" << abs_tol << ", " << rel_expr << "="
           << rel_tol << ". \n"
           << a_expr << "=" << a << ". \n"
           << b_expr << "=" << b;
}

#define ASSERT_NEAR_ABS_REL(a, b, abs_tol, rel_tol) \
    ASSERT_PRED_FORMAT4(IsNearAbsRel<double>, a, b, abs_tol, rel_tol)

class MatMulKernelTester {
public:
    MatMulKernelTester() { }

    MatMulKernelTester(const MatMulKernelTester&) = delete;
    MatMulKernelTester(MatMulKernelTester&&) = delete;
    MatMulKernelTester& operator=(const MatMulKernelTester&) = delete;
    MatMulKernelTester& operator=(MatMulKernelTester&&) = delete;

    [[nodiscard]]
    MatMulKernelTester& num_rows(std::uint32_t num_rows) {
        num_rows_ = num_rows;
        return *this;
    }

    std::uint32_t num_rows() const {
        return num_rows_;
    }

    [[nodiscard]]
    MatMulKernelTester& num_cols(std::uint32_t num_cols) {
        num_cols_ = num_cols;
        return *this;
    }

    std::uint32_t num_cols() const {
        return num_cols_;
    }

    [[nodiscard]]
    MatMulKernelTester& num_tokens(std::uint32_t num_tokens) {
        num_tokens_ = num_tokens;
        return *this;
    }

    std::uint32_t num_tokens() const {
        return num_tokens_;
    }

    [[nodiscard]]
    MatMulKernelTester& threadgroup_size(std::size_t threadgroup_size) {
        threadgroup_size_ = threadgroup_size;
        return *this;
    }

    std::size_t threadgroup_size() const {
        return threadgroup_size_;
    }

    void Validate(std::uint32_t vec_size) const {
        ASSERT_NE(num_rows(), 0);
        ASSERT_NE(num_cols(), 0);
        ASSERT_EQ(num_cols() % vec_size, 0);
        ASSERT_NE(num_tokens(), 0);
        ASSERT_NE(threadgroup_size(), 0);
    }

    enum class MatMulKernelType {
        DECODE_OPTIMIZED,
        PREFILL_QKV_OPTIMIZED,
        PREFILL_ATTN_OUTPUT_OPTIMIZED,
        PREFILL_MLP_GATE_OPTIMIZED,
    };

    void TestF32_BF16W(MatMulKernelType kernel_type = MatMulKernelType::DECODE_OPTIMIZED) const {
        Validate(/*vec_size=*/4);

        metal::CommandBuffer command_buffer_initialize{command_queue_};
        metal::Buffer input_buffer{device_, num_tokens() * num_cols() * sizeof(float)};
        metal::Buffer weight_buffer{device_, num_rows() * num_cols() * sizeof(gptoss_bfloat16)};
        metal::Buffer bias_buffer{device_, num_rows() * sizeof(gptoss_bfloat16)};
        metal::Buffer output_buffer{device_, num_tokens() * num_rows() * sizeof(float)};
        metal::Buffer output_buffer_copy{device_, num_tokens() * num_rows() * sizeof(float)};
        // KV cache buffer for PREFILL_QKV_OPTIMIZED: assume head_dim=64, num_kv_heads=8
        const std::uint32_t kHeadDim = 64;
        const std::uint32_t kNumKvHeads = 8;
        metal::Buffer kv_cache_buffer{device_, static_cast<std::size_t>(kNumKvHeads) * num_tokens() * 2 * kHeadDim * sizeof(float)};
        metal::Buffer control_buffer{device_, sizeof(gptoss_control)};
        std::memset(control_buffer.ptr(), 0, sizeof(gptoss_control));

        command_buffer_initialize.encode_launch_f32_fill_random(
            f32_fill_random_fn_,
            /*threadgroup_size=*/0,
            /*max_threadgroups=*/kFillRandomMaxThreadgroups,
            /*output_buffer=*/input_buffer,
            /*output_offset=*/0,
            num_tokens() * num_cols(), kSeed, /*offset=*/0, /*min=*/-1.0f, /*max=*/1.0);

        command_buffer_initialize.encode_launch_bf16_fill_random(
            bf16_fill_random_fn_,
            /*threadgroup_size=*/0,
            /*max_threadgroups=*/kFillRandomMaxThreadgroups,
            /*output_buffer=*/weight_buffer,
            /*output_offset=*/0,
            num_rows() * num_cols(), kSeed + 1, /*offset=*/0, /*min=*/-1.0f, /*max=*/1.0);

        command_buffer_initialize.encode_launch_bf16_fill_random(
            bf16_fill_random_fn_,
            /*threadgroup_size=*/0,
            /*max_threadgroups=*/kFillRandomMaxThreadgroups,
            /*output_buffer=*/bias_buffer,
            /*output_offset=*/0,
            num_rows(), kSeed + 2, /*offset=*/0, /*min=*/-1.0f, /*max=*/1.0);

        // Fill output buffer with random values to test matmul with add.
        command_buffer_initialize.encode_launch_f32_fill_random(
            f32_fill_random_fn_,
            /*threadgroup_size=*/0,
            /*max_threadgroups=*/kFillRandomMaxThreadgroups,
            /*output_buffer=*/output_buffer,
            /*output_offset=*/0, num_tokens() * num_rows(), kSeed + 3,
            /*offset=*/0,
            /*min=*/-1.0f, /*max=*/1.0);
        command_buffer_initialize.commit();
        command_buffer_initialize.wait_completion();
        if (kernel_type ==
            MatMulKernelType::PREFILL_ATTN_OUTPUT_OPTIMIZED) {
            // Copy output buffer to output buffer copy to use when calculating reference.
            const uint64_t bytes =
                uint64_t(num_tokens()) * uint64_t(num_rows()) * sizeof(float);

            void* src = output_buffer.ptr();
            void* dst = output_buffer_copy.ptr();
            assert(src && dst && "Buffers must be CPU-mappable for memcpy");

            std::memcpy(reinterpret_cast<std::byte*>(dst),
                        reinterpret_cast<const std::byte*>(src), bytes);
        }

        metal::CommandBuffer command_buffer_compute{command_queue_};
        switch (kernel_type) {
        case MatMulKernelType::DECODE_OPTIMIZED:
            Check(gptoss_metal_command_buffer_encode_launch_f32_bf16w_matmul(
                      command_buffer_compute.handle(), f32_bf16w_matmul_fn_.handle(),
                      /*threadgroup_size=*/threadgroup_size(), input_buffer.handle(),
                      /*input_offset=*/0, weight_buffer.handle(),
                      /*weight_offset=*/0, bias_buffer.handle(),
                      /*bias_offset=*/0, output_buffer.handle(),
                      /*output_offset=*/0, control_buffer.handle(),
                      /*control_offset=*/0, num_tokens(), num_cols(), num_rows()),
                  "gptoss_metal_command_buffer_encode_launch_f32_bf16w_matmul");
            break;
        case MatMulKernelType::PREFILL_QKV_OPTIMIZED:
            Check(
                gptoss_metal_command_buffer_encode_launch_f32_bf16w_dense_matmul_qkv(
                    command_buffer_compute.handle(),
                    f32_bf16w_dense_matmul_qkv_fn_.handle(), input_buffer.handle(),
                    /*input_offset=*/0, weight_buffer.handle(),
                    /*weight_offset=*/0, bias_buffer.handle(),
                    /*bias_offset=*/0, output_buffer.handle(),
                    /*output_offset=*/0, kv_cache_buffer.handle(),
                    /*kv_offset=*/0, control_buffer.handle(),
                    /*control_offset=*/0, num_tokens(), num_cols(), num_rows(),
                    /*max_tokens=*/num_tokens(), /*token_offset=*/0),
                "gptoss_metal_command_buffer_encode_launch_f32_bf16w_dense_matmul_qkv");
            break;
        case MatMulKernelType::PREFILL_ATTN_OUTPUT_OPTIMIZED:
            Check(
                gptoss_metal_command_buffer_encode_launch_f32_bf16w_dense_matmul_attn_output(
                    command_buffer_compute.handle(),
                    f32_bf16w_dense_matmul_attn_output_fn_.handle(),
                    input_buffer.handle(),
                    /*input_offset=*/0, weight_buffer.handle(),
                    /*weight_offset=*/0, bias_buffer.handle(),
                    /*bias_offset=*/0, output_buffer.handle(),
                    /*output_offset=*/0, control_buffer.handle(),
                    /*control_offset=*/0, num_tokens(), num_cols(), num_rows()),
                "gptoss_metal_command_buffer_encode_launch_f32_bf16w_dense_matmul_attn_output");
            break;
        case MatMulKernelType::PREFILL_MLP_GATE_OPTIMIZED:
            Check(
                gptoss_metal_command_buffer_encode_launch_f32_bf16w_dense_matmul_mlp_gate(
                    command_buffer_compute.handle(),
                    f32_bf16w_dense_matmul_mlp_gate_fn_.handle(),
                    input_buffer.handle(),
                    /*input_offset=*/0, weight_buffer.handle(),
                    /*weight_offset=*/0, bias_buffer.handle(),
                    /*bias_offset=*/0, output_buffer.handle(),
                    /*output_offset=*/0, control_buffer.handle(),
                    /*control_offset=*/0, num_tokens(), num_cols(), num_rows()),
                "gptoss_metal_command_buffer_encode_launch_f32_bf16w_dense_matmul_mlp_gate");
            break;
        }
        command_buffer_compute.commit();
        command_buffer_compute.wait_completion();
        const float* input_ptr = static_cast<const float*>(input_buffer.ptr());
        const gptoss_bfloat16* weight_ptr = static_cast<const gptoss_bfloat16*>(weight_buffer.ptr());
        const gptoss_bfloat16* bias_ptr = static_cast<const gptoss_bfloat16*>(bias_buffer.ptr());
        const float* output_ptr = static_cast<const float*>(output_buffer.ptr());
        const float* kv_ptr = static_cast<const float*>(kv_cache_buffer.ptr());
        const float* output_ptr_copy = static_cast<const float*>(output_buffer_copy.ptr());
        for (size_t t = 0; t < num_tokens(); t++) {
            for (size_t r = 0; r < num_rows(); r++) {
                double ref_sum = upcast<double>(bias_ptr[r]);
                for (size_t c = 0; c < num_cols(); c++) {
                    const double ref_weight = upcast<double>(weight_ptr[r * num_cols() + c]);
                    const double input_value = upcast<double>(input_ptr[t * num_cols() + c]);
                    ref_sum = std::fma(input_value, ref_weight, ref_sum);
                }

                if (kernel_type ==
                    MatMulKernelType::PREFILL_ATTN_OUTPUT_OPTIMIZED) {
                    ref_sum += upcast<double>(output_ptr_copy[t * num_rows() + r]);
                }
                if (kernel_type == MatMulKernelType::PREFILL_QKV_OPTIMIZED) {
                    // In this optimized path, V rows are written to the kv cache at index 1.
                    // Assume num_q_heads=64, num_kv_heads=8, head_dim=64 and QKV packed as [Q][K][V].
                    const std::size_t v_rows_start = (64 + 8) * 64; // rows offset where V begins
                    if (r >= v_rows_start) {
                        const std::size_t v_row_index = r - v_rows_start;
                        const std::size_t kv_head = v_row_index / kHeadDim;
                        const std::size_t d = v_row_index % kHeadDim;
                        const std::size_t kv_base = ((kv_head * num_tokens() + t) * 2 + 1) * kHeadDim;
                        ASSERT_NEAR_ABS_REL(upcast<double>(kv_ptr[kv_base + d]), ref_sum, 2.0e-4, 1.0e-4)
                            << "token " << t << ", v-row " << r;
                        continue;
                    }
                }
                ASSERT_NEAR_ABS_REL(upcast<double>(output_ptr[t * num_rows() + r]),
                                    ref_sum, 2.0e-4, 1.0e-4)
                    << "token " << t;
            }
        }
    }

private:
    static constexpr std::uint64_t kSeed{UINT64_C(1019827666124465388)};
    static constexpr std::size_t kFillRandomMaxThreadgroups = 10;
    static constexpr float fp4e2m1_to_fp32[16] = {
        +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };

    metal::Device device_{};
    metal::CommandQueue command_queue_{device_};
    metal::Library library_{device_};
    metal::Function f32_fill_random_fn_{library_, "gptoss_f32_fill_random"};
    metal::Function bf16_fill_random_fn_{library_, "gptoss_bf16_fill_random"};
    metal::Function f32_bf16w_matmul_fn_{library_, "gptoss_f32_bf16w_matmul"};
    metal::Function f32_bf16w_dense_matmul_qkv_fn_{library_, "gptoss_f32_bf16w_dense_matmul_qkv"};
    metal::Function f32_bf16w_dense_matmul_attn_output_fn_{library_, "gptoss_f32_bf16w_dense_matmul_attn_output"};
    metal::Function f32_bf16w_dense_matmul_mlp_gate_fn_{library_, "gptoss_f32_bf16w_dense_matmul_mlp_gate"};
    std::uint32_t num_tokens_{1};
    std::uint32_t num_rows_{1};
    std::uint32_t num_cols_{32};
    std::size_t threadgroup_size_{32};
};

}  // namespace gptoss
