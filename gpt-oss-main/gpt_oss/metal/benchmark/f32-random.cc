#include <gpt-oss.h>
#include <internal/metal.hpp>
#include <internal/metal-kernels.h>

#include <benchmark/benchmark.h>

using gptoss::Check;
using namespace gptoss::metal;

static void f32_fill_random(benchmark::State& state) {
    const size_t numel = state.range(0);

    Device device;
    CommandQueue command_queue{device};
    Library library{device};
    Function f32_fill_random_fn{library, "gptoss_f32_fill_random"};
    Buffer buffer{device, numel * sizeof(float)};

    constexpr uint64_t seed = UINT64_C(1019827666124465388);
    constexpr uint64_t offset = UINT64_C(12345678901234567890);
    const float min = -1.0f;
    const float max = 7.0f;
    for (auto _ : state) {
        CommandBuffer command_buffer{command_queue};

        Check(gptoss_metal_command_buffer_encode_launch_f32_fill_random(
                command_buffer.handle(),
                f32_fill_random_fn.handle(),
                /*threadgroup_size=*/0,
                /*max_threadgroups=*/120,
                /*output_buffer=*/buffer.handle(),
                /*output_offset=*/0,
                numel, seed, offset, min, max),
            "gptoss_metal_command_buffer_encode_launch_f32_fill_random");

        command_buffer.commit();
        const double elapsed_seconds = command_buffer.wait_completion();
        state.SetIterationTime(elapsed_seconds);
    }
    
    const int64_t elements_per_iteration = numel;
    state.counters["elements"] =
        benchmark::Counter(state.iterations() * elements_per_iteration,
                           benchmark::Counter::kIsRate);

    const int64_t bytes_per_iteration = numel * sizeof(float);
    state.counters["bytes"] =
        benchmark::Counter(state.iterations() * bytes_per_iteration,
                           benchmark::Counter::kIsRate);
}

constexpr int64_t giga = INT64_C(1073741824);
BENCHMARK(f32_fill_random)->Arg(2 * giga)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
