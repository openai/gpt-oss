#pragma once

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include <internal/datatype.hpp>
#include <internal/metal.hpp>
#include <internal/metal-kernels.h>
#include <internal/rng.hpp>


namespace gptoss {

class FillRandomKernelTester {
public:
    FillRandomKernelTester() { }

    FillRandomKernelTester(const FillRandomKernelTester&) = delete;
    FillRandomKernelTester(FillRandomKernelTester&&) = delete;
    FillRandomKernelTester& operator=(const FillRandomKernelTester&) = delete;
    FillRandomKernelTester& operator=(FillRandomKernelTester&&) = delete;

    [[nodiscard]]
    FillRandomKernelTester& num_elements(std::uint32_t num_elements) {
        num_elements_ = num_elements;
        return *this;
    }

    std::uint32_t num_elements() const {
        return num_elements_;
    }

    [[nodiscard]]
    FillRandomKernelTester& threadgroup_size(std::size_t threadgroup_size) {
        threadgroup_size_ = threadgroup_size;
        return *this;
    }

    std::size_t threadgroup_size() const {
        return threadgroup_size_;
    }

    [[nodiscard]]
    FillRandomKernelTester& max_threadgroups(std::size_t max_threadgroups) {
        max_threadgroups_ = max_threadgroups;
        return *this;
    }

    std::size_t max_threadgroups() const {
        return max_threadgroups_;
    }

    void Validate() const {
        ASSERT_NE(num_elements(), 0);
        ASSERT_NE(threadgroup_size(), 0);
        ASSERT_NE(max_threadgroups(), 0);
    }

    void TestU32() const {
        Validate();

        metal::Buffer output_buffer{device_, num_elements() * sizeof(std::uint32_t)};

        metal::CommandBuffer command_buffer{command_queue_};
        command_buffer.encode_launch_u32_fill_random(
            u32_fill_random_fn_,
            threadgroup_size(),
            max_threadgroups(),
            output_buffer,
            /*output_offset=*/0,
            num_elements(), kSeed, kOffset);

        command_buffer.commit();
        command_buffer.wait_completion();

        const std::uint32_t* output_ptr = static_cast<const std::uint32_t*>(output_buffer.ptr());
        for (std::size_t i = 0; i < num_elements(); i++) {
            const std::uint32_t ref_value = gptoss::rng::squares32(kOffset + i, kSeed);
            ASSERT_EQ(output_ptr[i], ref_value)
                << "at position " << i << " / " << num_elements();
        }
    }

private:
    static constexpr uint64_t kSeed{UINT64_C(1019827666124465388)};
    static constexpr uint64_t kOffset{UINT64_C(12345678901234567890)};

    metal::Device device_{};
    metal::CommandQueue command_queue_{device_};
    metal::Library library_{device_};
    metal::Function f32_fill_random_fn_{library_, "gptoss_f32_fill_random"};
    metal::Function bf16_fill_random_fn_{library_, "gptoss_bf16_fill_random"};
    metal::Function u32_fill_random_fn_{library_, "gptoss_u32_fill_random"};
    std::uint32_t num_elements_{1};
    std::size_t threadgroup_size_{32};
    std::size_t max_threadgroups_{1};
};

}  // namespace gptoss
