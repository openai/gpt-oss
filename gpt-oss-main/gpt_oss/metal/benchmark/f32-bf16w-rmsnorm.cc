#include <gpt-oss.h>
#include <internal/datatype.h>
#include <internal/metal.hpp>
#include <internal/metal-kernels.h>

#include <cstring>

#include <benchmark/benchmark.h>

using gptoss::Check;
using namespace gptoss::metal;

constexpr float kEpsilon = 1.0e-5f;
constexpr uint64_t kSeed = UINT64_C(1019827666124465388);

static void f32_bf16w_rnsnorm(benchmark::State& state) {
    const size_t num_tokens = 1;
    const size_t num_channels = state.range(0);

    Device device;
    CommandQueue command_queue{device};
    Library library{device};
    Function f32_fill_random_fn{library, "gptoss_f32_fill_random"};
    Function bf16_fill_random_fn{library, "gptoss_bf16_fill_random"};
    Function f32_bf16w_rmsnorm_fn{library, "gptoss_f32_bf16w_rmsnorm"};
    Buffer input_buffer{device, num_tokens * num_channels * sizeof(float)};
    Buffer weight_buffer{device, num_channels * sizeof(gptoss_bfloat16)};
    Buffer output_buffer{device, num_tokens * num_channels * sizeof(float)};
    Buffer control_buffer{device, sizeof(gptoss_control)};
    std::memset(control_buffer.ptr(), 0, sizeof(gptoss_control));

    {
        CommandBuffer command_buffer{command_queue};

        size_t offset = 0;
        Check(gptoss_metal_command_buffer_encode_launch_f32_fill_random(
                command_buffer.handle(),
                f32_fill_random_fn.handle(),
                /*threadgroup_size=*/0,
                /*max_threadgroups=*/10,
                /*output_buffer=*/input_buffer.handle(),
                /*output_offset=*/0,
                num_channels, kSeed, offset, /*min=*/-1.0f, /*max=*/1.0),
            "gptoss_metal_command_buffer_encode_launch_f32_fill_random");
        offset += num_channels;

        Check(gptoss_metal_command_buffer_encode_launch_bf16_fill_random(
                command_buffer.handle(),
                bf16_fill_random_fn.handle(),
                /*threadgroup_size=*/0,
                /*max_threadgroups=*/10,
                /*output_buffer=*/weight_buffer.handle(),
                /*output_offset=*/0,
                num_channels, kSeed, offset, /*min=*/-1.0f, /*max=*/1.0),
            "gptoss_metal_command_buffer_encode_launch_bf16_fill_random");
        offset += num_channels;

        command_buffer.commit();
        command_buffer.wait_completion();
    }

    for (auto _ : state) {
        CommandBuffer command_buffer{command_queue};

        Check(gptoss_metal_command_buffer_encode_launch_f32_bf16w_rmsnorm(
                command_buffer.handle(),
                f32_bf16w_rmsnorm_fn.handle(),
                input_buffer.handle(),
                /*input_offset=*/0,
                weight_buffer.handle(),
                /*weight_offset=*/0,
                output_buffer.handle(),
                /*output_offset=*/0,
                control_buffer.handle(),
                /*control_offset=*/0,
                num_tokens,
                num_channels,
                kEpsilon),
            "gptoss_metal_command_buffer_encode_launch_f32_bf16w_rmsnorm");

        command_buffer.commit();
        const double elapsed_seconds = command_buffer.wait_completion();
        state.SetIterationTime(elapsed_seconds);
    }

    const size_t num_elements = num_tokens * num_channels;
    state.counters["elements"] =
        benchmark::Counter(state.iterations() * num_elements,
                           benchmark::Counter::kIsRate);

    const int64_t bytes_per_iteration = input_buffer.size() + weight_buffer.size() + output_buffer.size();
    state.counters["bytes"] =
        benchmark::Counter(state.iterations() * bytes_per_iteration,
                           benchmark::Counter::kIsRate);
}

BENCHMARK(f32_bf16w_rnsnorm)->Arg(2880)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
