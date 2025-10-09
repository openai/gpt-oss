#include <gpt-oss.h>
#include <internal/datatype.h>
#include <internal/metal.hpp>
#include <internal/metal-kernels.h>

#include <cstring>

#include <benchmark/benchmark.h>

using gptoss::Check;
using namespace gptoss::metal;

static void mf4_f32_convert(benchmark::State& state) {
    const size_t num_blocks = state.range(0);
    const size_t num_elements = num_blocks * 32;
    const size_t num_bytes = num_elements / 2;

    Device device;
    CommandQueue command_queue{device};
    Library library{device};
    Function mf4_f32_convert_fn{library, "gptoss_mf4_f32_convert"};
    Buffer block_buffer{device, num_bytes};
    Buffer scale_buffer{device, num_blocks * sizeof(gptoss_float8ue8m0)};
    Buffer output_buffer{device, num_elements * sizeof(float)};

    std::memset(block_buffer.ptr(), 0x91, num_bytes);  // force subnormals
    std::memset(scale_buffer.ptr(), 128, num_blocks * sizeof(uint8_t));  // scale = 2.0

    for (auto _ : state) {
        CommandBuffer command_buffer{command_queue};

        Check(gptoss_metal_command_buffer_encode_launch_mf4_f32_convert(
                command_buffer.handle(),
                mf4_f32_convert_fn.handle(),
                /*threadgroup_size=*/0,
                /*max_threadgroups=*/120,
                block_buffer.handle(),
                scale_buffer.handle(),
                output_buffer.handle(),
                num_elements),
            "gptoss_metal_command_buffer_encode_launch_mf4_f32_convert");

        command_buffer.commit();
        const double elapsed_seconds = command_buffer.wait_completion();
        state.SetIterationTime(elapsed_seconds);
    }

    state.counters["blocks"] =
        benchmark::Counter(state.iterations() * num_blocks,
                           benchmark::Counter::kIsRate);

    state.counters["elements"] =
        benchmark::Counter(state.iterations() * num_elements,
                           benchmark::Counter::kIsRate);

    const int64_t bytes_per_iteration = num_bytes + num_blocks + num_elements * sizeof(float);
    state.counters["bytes"] =
        benchmark::Counter(state.iterations() * bytes_per_iteration,
                           benchmark::Counter::kIsRate);
}

constexpr int64_t mega = INT64_C(1048576);
BENCHMARK(mf4_f32_convert)->Arg(256 * mega)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
