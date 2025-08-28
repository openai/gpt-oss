#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include <internal/kernel-args.h>
#include <internal/log.h>
#include <internal/math.h>
#include <internal/metal.h>
#include <internal/metal-kernels.h>


enum gptoss_status gptoss_metal_command_buffer_encode_launch_u32_fill_random(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* u32_fill_random_fn,
    size_t threadgroup_size,
    size_t max_threadgroups,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    uint64_t num_elements,
    uint64_t rng_seed,
    uint64_t rng_offset)
{
    if (command_buffer->object == NULL || u32_fill_random_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = u32_fill_random_fn->max_threadgroup_threads;
    } else if (threadgroup_size > u32_fill_random_fn->max_threadgroup_threads) {
        return gptoss_status_invalid_argument;
    }

    const size_t num_vecs = num_elements;
    const size_t num_vecs_per_threadgroup = math_ceil_div(num_vecs, max_threadgroups * threadgroup_size) * threadgroup_size;
    const size_t num_threadgroups = math_min(max_threadgroups, math_ceil_div(num_vecs, num_vecs_per_threadgroup));
    const struct gptoss_u32_fill_random_args args = {
        .num_vecs = num_vecs,
        .num_vecs_per_threadgroup = num_vecs_per_threadgroup,
        .seed = rng_seed,
        .offset = rng_offset,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, u32_fill_random_fn,
        threadgroup_size, 1, 1,
        num_threadgroups, 1, 1,
        sizeof(args), &args,
        1, &output_buffer, &output_offset,
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_fill_random(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_fill_random_fn,
    size_t threadgroup_size,
    size_t max_threadgroups,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    uint64_t num_elements,
    uint64_t rng_seed,
    uint64_t rng_offset,
    float rng_min,
    float rng_max)
{
    if (command_buffer->object == NULL || f32_fill_random_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = f32_fill_random_fn->max_threadgroup_threads;
    } else if (threadgroup_size > f32_fill_random_fn->max_threadgroup_threads) {
        return gptoss_status_invalid_argument;
    }

    if (rng_min >= rng_max) {
        return gptoss_status_invalid_argument;
    }

    const size_t num_vecs = num_elements;
    const size_t num_vecs_per_threadgroup = math_ceil_div(num_vecs, max_threadgroups * threadgroup_size) * threadgroup_size;
    const size_t num_threadgroups = math_min(max_threadgroups, math_ceil_div(num_vecs, num_vecs_per_threadgroup));
    const struct gptoss_f32_fill_random_args args = {
        .num_vecs = num_vecs,
        .num_vecs_per_threadgroup = num_vecs_per_threadgroup,
        .seed = rng_seed,
        .offset = rng_offset,
        .scale = (rng_max - rng_min) * 0x1.0p-32f,
        .bias = (rng_min + rng_max) * 0.5f,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_fill_random_fn,
        threadgroup_size, 1, 1,
        num_threadgroups, 1, 1,
        sizeof(args), &args,
        1, &output_buffer, &output_offset,
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_bf16_fill_random(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* bf16_fill_random_fn,
    size_t threadgroup_size,
    size_t max_threadgroups,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    uint64_t num_elements,
    uint64_t rng_seed,
    uint64_t rng_offset,
    float rng_min,
    float rng_max)
{
    if (command_buffer->object == NULL || bf16_fill_random_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = bf16_fill_random_fn->max_threadgroup_threads;
    } else if (threadgroup_size > bf16_fill_random_fn->max_threadgroup_threads) {
        return gptoss_status_invalid_argument;
    }

    if (rng_min >= rng_max) {
        return gptoss_status_invalid_argument;
    }

    const size_t num_vecs = num_elements;
    const size_t num_vecs_per_threadgroup = math_ceil_div(num_vecs, max_threadgroups * threadgroup_size) * threadgroup_size;
    const size_t num_threadgroups = math_min(max_threadgroups, math_ceil_div(num_vecs, num_vecs_per_threadgroup));
    const struct gptoss_f32_fill_random_args args = {
        .num_vecs = num_vecs,
        .num_vecs_per_threadgroup = num_vecs_per_threadgroup,
        .seed = rng_seed,
        .offset = rng_offset,
        .scale = (rng_max - rng_min) * 0x1.0p-32f,
        .bias = (rng_min + rng_max) * 0.5f,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, bf16_fill_random_fn,
        threadgroup_size, 1, 1,
        num_threadgroups, 1, 1,
        sizeof(args), &args,
        1, &output_buffer, &output_offset,
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_mf4_f32_convert(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* mf4_f32_convert_fn,
    size_t threadgroup_size,
    size_t max_threadgroups,
    const struct gptoss_metal_buffer* block_buffer,
    const struct gptoss_metal_buffer* scale_buffer,
    const struct gptoss_metal_buffer* output_buffer,
    uint64_t num_elements)
{
    if (command_buffer->object == NULL || mf4_f32_convert_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (num_elements % 32 != 0) {
        return gptoss_status_invalid_argument;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = mf4_f32_convert_fn->max_threadgroup_threads;
    } else if (threadgroup_size > mf4_f32_convert_fn->max_threadgroup_threads) {
        return gptoss_status_invalid_argument;
    }

    const size_t num_vecs = num_elements / 32;
    const size_t num_vecs_per_threadgroup = math_ceil_div(num_vecs, max_threadgroups * threadgroup_size) * threadgroup_size;
    const size_t num_threadgroups = math_min(max_threadgroups, math_ceil_div(num_vecs, num_vecs_per_threadgroup));
    const struct gptoss_convert_args args = {
        .num_vecs = num_vecs,
        .num_vecs_per_threadgroup = num_vecs_per_threadgroup,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, mf4_f32_convert_fn,
        threadgroup_size, 1, 1,
        num_threadgroups, 1, 1,
        sizeof(args), &args,
        3, (const struct gptoss_metal_buffer *[]) {block_buffer, scale_buffer, output_buffer}, NULL,
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_bf16_f32_embeddings(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* bf16_f32_embeddings_fn,
    size_t threadgroup_size,
    const struct gptoss_metal_buffer* token_buffer,
    size_t token_offset,
    const struct gptoss_metal_buffer* weight_buffer,
    size_t weight_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    uint32_t num_tokens,
    uint32_t num_channels)
{
    if (command_buffer->object == NULL || bf16_f32_embeddings_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (num_channels % 4 != 0) {
        return gptoss_status_invalid_argument;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = bf16_f32_embeddings_fn->max_threadgroup_threads;
    } else if (threadgroup_size > bf16_f32_embeddings_fn->max_threadgroup_threads) {
        return gptoss_status_invalid_argument;
    }

    const uint32_t num_vecs = num_channels / 4;
    const struct gptoss_embeddings_args args = {
        .num_vecs = num_vecs,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, bf16_f32_embeddings_fn,
        threadgroup_size, 1, 1,
        num_tokens, 1, 1,
        sizeof(args), &args,
        3,
        (const struct gptoss_metal_buffer *[]) {token_buffer, weight_buffer, output_buffer},
        (const size_t[]) {token_offset, weight_offset, output_offset},
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_bf16w_rmsnorm(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_bf16w_rmsnorm_fn,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* weight_buffer,
    size_t weight_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    uint32_t num_tokens,
    uint32_t num_channels,
    float epsilon)
{
    if (command_buffer->object == NULL || f32_bf16w_rmsnorm_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (num_channels % 4 != 0) {
        return gptoss_status_invalid_argument;
    }

    if (f32_bf16w_rmsnorm_fn->max_threadgroup_threads < 1024) {
        return gptoss_status_unsupported_system;
    }

    if (f32_bf16w_rmsnorm_fn->simdgroup_threads != 32) {
        return gptoss_status_unsupported_system;
    }

    const uint32_t num_vecs = num_channels / 4;
    const struct gptoss_rmsnorm_args args = {
        .num_vecs = num_vecs,
        .num_channels = (float) num_channels,
        .epsilon = epsilon,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_bf16w_rmsnorm_fn,
        /*threadgroup_size=*/1024, 1, 1,
        num_tokens, 1, 1,
        sizeof(args), &args,
        3,
        (const struct gptoss_metal_buffer *[]) {input_buffer, weight_buffer, output_buffer},
        (const size_t[]) {input_offset, weight_offset, output_offset},
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_bf16w_matmul(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_bf16w_matmul_fn,
    size_t threadgroup_size,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* weight_buffer,
    size_t weight_offset,
    const struct gptoss_metal_buffer* bias_buffer,
    size_t bias_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    uint32_t num_tokens,
    uint32_t num_cols,
    uint32_t num_rows)
{
    if (command_buffer->object == NULL || f32_bf16w_matmul_fn->pipeline_state_object == NULL) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul kernel launch: invalid command buffer or pipeline state object");
        return gptoss_status_invalid_state;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = f32_bf16w_matmul_fn->simdgroup_threads;
    } else if (threadgroup_size > f32_bf16w_matmul_fn->max_threadgroup_threads) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul kernel launch: threadgroup size (%zu) exceeds supported maximum (%zu)",
            threadgroup_size, f32_bf16w_matmul_fn->max_threadgroup_threads);
        return gptoss_status_invalid_argument;
    }

    if (num_cols % 4 != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul kernel launch: number of columns (%" PRIu32 ") is not divisible by 4",
            num_cols);
        return gptoss_status_invalid_argument;
    }
    const size_t num_simdgroups = threadgroup_size / f32_bf16w_matmul_fn->simdgroup_threads;
    if (num_rows % num_simdgroups != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul kernel launch: number of rows (%" PRIu32 ") is not divisible by the number of simdgroups (%zu)",
            num_rows, num_simdgroups);
        return gptoss_status_invalid_argument;
    }

    const struct gptoss_matmul_args args = {
        .num_column_vecs = num_cols / 4,
        .num_rows = num_rows,
        .add = 0,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_bf16w_matmul_fn,
        threadgroup_size, 1, 1,
        num_rows / num_simdgroups, num_tokens, 1,
        sizeof(args), &args,
        4,
        (const struct gptoss_metal_buffer *[]) {input_buffer, weight_buffer, bias_buffer, output_buffer},
        (const size_t[]) {input_offset, weight_offset, bias_offset, output_offset},
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_bf16w_matmul_add(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_bf16w_matmul_fn,
    size_t threadgroup_size,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* weight_buffer,
    size_t weight_offset,
    const struct gptoss_metal_buffer* bias_buffer,
    size_t bias_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    uint32_t num_tokens,
    uint32_t num_cols,
    uint32_t num_rows)
{
    if (command_buffer->object == NULL || f32_bf16w_matmul_fn->pipeline_state_object == NULL) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul_add kernel launch: invalid command buffer or pipeline state object");
        return gptoss_status_invalid_state;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = f32_bf16w_matmul_fn->simdgroup_threads;
    } else if (threadgroup_size > f32_bf16w_matmul_fn->max_threadgroup_threads) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul_add kernel launch: threadgroup size (%zu) exceeds supported maximum (%zu)",
            threadgroup_size, f32_bf16w_matmul_fn->max_threadgroup_threads);
        return gptoss_status_invalid_argument;
    }

    if (num_cols % 4 != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul_add kernel launch: number of columns (%" PRIu32 ") is not divisible by 4",
            num_cols);
        return gptoss_status_invalid_argument;
    }
    const size_t num_simdgroups = threadgroup_size / f32_bf16w_matmul_fn->simdgroup_threads;
    if (num_rows % num_simdgroups != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul_add kernel launch: number of rows (%" PRIu32 ") is not divisible by the number of simdgroups (%zu)",
            num_rows, num_simdgroups);
        return gptoss_status_invalid_argument;
    }

    const struct gptoss_matmul_args args = {
        .num_column_vecs = num_cols / 4,
        .num_rows = num_rows,
        .add = 1,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_bf16w_matmul_fn,
        threadgroup_size, 1, 1,
        num_rows / num_simdgroups, num_tokens, 1,
        sizeof(args), &args,
        4,
        (const struct gptoss_metal_buffer *[]) {input_buffer, weight_buffer, bias_buffer, output_buffer},
        (const size_t[]) {input_offset, weight_offset, bias_offset, output_offset},
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_bf16w_unembedding(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_bf16w_unembedding_fn,
    size_t threadgroup_size,
    size_t max_threadgroups,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* weight_buffer,
    size_t weight_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    const struct gptoss_metal_buffer* argmax_buffer,
    size_t argmax_offset,
    uint32_t num_tokens,
    uint32_t num_cols,
    uint32_t num_rows)
{
    if (command_buffer->object == NULL || f32_bf16w_unembedding_fn->pipeline_state_object == NULL) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_unembedding kernel launch: invalid command buffer or pipeline state object");
        return gptoss_status_invalid_state;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = f32_bf16w_unembedding_fn->simdgroup_threads;
    } else if (threadgroup_size > f32_bf16w_unembedding_fn->max_threadgroup_threads) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_unembedding kernel launch: threadgroup size (%zu) exceeds supported maximum (%zu)",
            threadgroup_size, f32_bf16w_unembedding_fn->max_threadgroup_threads);
        return gptoss_status_invalid_argument;
    }

    if (num_cols % 4 != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_unembedding kernel launch: number of columns (%" PRIu32 ") is not divisible by 4",
            num_cols);
        return gptoss_status_invalid_argument;
    }

    const size_t num_simdgroups = threadgroup_size / f32_bf16w_unembedding_fn->simdgroup_threads;
    const size_t num_rows_per_threadgroup = math_ceil_div(num_rows, max_threadgroups * num_simdgroups) * num_simdgroups;
    const size_t num_threadgroups = math_min(max_threadgroups, math_ceil_div(num_rows, num_rows_per_threadgroup));
    const struct gptoss_unembedding_args args = {
        .num_column_vecs = num_cols / 4,
        .num_rows_per_threadgroup = num_rows_per_threadgroup,
        .num_rows = num_rows,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_bf16w_unembedding_fn,
        threadgroup_size, 1, 1,
        num_threadgroups, num_tokens, 1,
        sizeof(args), &args,
        4,
        (const struct gptoss_metal_buffer *[]) {input_buffer, weight_buffer, output_buffer, argmax_buffer},
        (const size_t[]) {input_offset, weight_offset, output_offset, argmax_offset},
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_mf4w_moe_matmul_swiglu(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_mf4w_moe_matmul_swiglu_fn,
    size_t threadgroup_size,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* expert_buffer,
    size_t expert_offset,
    const struct gptoss_metal_buffer* weight_block_buffer,
    size_t weight_block_offset,
    const struct gptoss_metal_buffer* weight_scale_buffer,
    size_t weight_scale_offset,
    const struct gptoss_metal_buffer* bias_buffer,
    size_t bias_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    float swiglu_limit,
    uint32_t expert_stride,
    uint32_t num_tokens,
    uint32_t num_active_experts,
    uint32_t num_cols,
    uint32_t num_rows)
{
    if (command_buffer->object == NULL || f32_mf4w_moe_matmul_swiglu_fn->pipeline_state_object == NULL) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_matmul_swiglu kernel launch: invalid command buffer or pipeline state object");
        return gptoss_status_invalid_state;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = 2 * f32_mf4w_moe_matmul_swiglu_fn->simdgroup_threads;
    } else if (threadgroup_size > f32_mf4w_moe_matmul_swiglu_fn->max_threadgroup_threads) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_matmul_swiglu kernel launch: threadgroup size (%zu) exceeds supported maximum (%zu)",
            threadgroup_size, f32_mf4w_moe_matmul_swiglu_fn->max_threadgroup_threads);
        return gptoss_status_invalid_argument;
    } else if (threadgroup_size % (2 * f32_mf4w_moe_matmul_swiglu_fn->simdgroup_threads)) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_matmul_swiglu kernel launch: threadgroup size (%zu) is not divisible by simdgroup size (%zu) multiplied by 2X",
            threadgroup_size, f32_mf4w_moe_matmul_swiglu_fn->simdgroup_threads);
        return gptoss_status_invalid_argument;
    }

    if (num_cols % 32 != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_matmul_swiglu kernel launch: number of columns (%" PRIu32 ") is not divisible by 32",
            num_cols);
        return gptoss_status_invalid_argument;
    }
    const size_t num_simdgroups = threadgroup_size / f32_mf4w_moe_matmul_swiglu_fn->simdgroup_threads;
    if ((2 * num_rows) % num_simdgroups != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_bf16w_matmul_add kernel launch: "
            "the number of rows (%" PRIu32 ") multiplied by 2X is not divisible by the number of simdgroups (%zu)",
            num_rows, num_simdgroups);
        return gptoss_status_invalid_argument;
    }

    const struct gptoss_moe_matmul_swiglu_args args = {
        .num_column_vecs = num_cols / 32,
        .num_rows = num_rows,
        .num_active_experts = num_active_experts,
        .weight_expert_stride = expert_stride,
        .output_expert_stride = num_rows * num_tokens,
        .swiglu_min = -swiglu_limit,
        .swiglu_max = swiglu_limit,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_mf4w_moe_matmul_swiglu_fn,
        threadgroup_size, 1, 1,
        (2 * num_rows) / num_simdgroups, num_tokens, num_active_experts,
        sizeof(args), &args,
        6,
        (const struct gptoss_metal_buffer *[]) {input_buffer, expert_buffer, weight_block_buffer, weight_scale_buffer, bias_buffer, output_buffer},
        (const size_t[]) {input_offset, expert_offset, weight_block_offset, weight_scale_offset, bias_offset, output_offset},
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_mf4w_moe_matmul(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_mf4w_moe_matmul_fn,
    size_t threadgroup_size,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* expert_buffer,
    size_t expert_offset,
    const struct gptoss_metal_buffer* weight_block_buffer,
    size_t weight_block_offset,
    const struct gptoss_metal_buffer* weight_scale_buffer,
    size_t weight_scale_offset,
    const struct gptoss_metal_buffer* bias_buffer,
    size_t bias_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    uint32_t expert_stride,
    uint32_t num_tokens,
    uint32_t num_active_experts,
    uint32_t num_cols,
    uint32_t num_rows)
{
    if (command_buffer->object == NULL || f32_mf4w_moe_matmul_fn->pipeline_state_object == NULL) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_matmul kernel launch: invalid command buffer or pipeline state object");
        return gptoss_status_invalid_state;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = f32_mf4w_moe_matmul_fn->simdgroup_threads;
    } else if (threadgroup_size > f32_mf4w_moe_matmul_fn->max_threadgroup_threads) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_matmul kernel launch: threadgroup size (%zu) exceeds supported maximum (%zu)",
            threadgroup_size, f32_mf4w_moe_matmul_fn->max_threadgroup_threads);
        return gptoss_status_invalid_argument;
    } else if (threadgroup_size % f32_mf4w_moe_matmul_fn->simdgroup_threads) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_matmul kernel launch: threadgroup size (%zu) is not divisible by simdgroup size (%zu)",
            threadgroup_size, f32_mf4w_moe_matmul_fn->simdgroup_threads);
        return gptoss_status_invalid_argument;
    }

    if (num_cols % 32 != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_matmul kernel launch: number of columns (%" PRIu32 ") is not divisible by 32",
            num_cols);
        return gptoss_status_invalid_argument;
    }
    const size_t num_simdgroups = threadgroup_size / f32_mf4w_moe_matmul_fn->simdgroup_threads;
    if (num_rows % num_simdgroups != 0) {
        GPTOSS_LOG_ERROR("failed to encode f32_mf4w_moe_matmul kernel launch: "
            "the number of rows (%" PRIu32 ") is not divisible by the number of simdgroups (%zu)",
            num_rows, num_simdgroups);
        return gptoss_status_invalid_argument;
    }

    const struct gptoss_moe_matmul_args args = {
        .num_column_vecs = num_cols / 32,
        .num_rows = num_rows,
        .num_active_experts = num_active_experts,
        .input_expert_stride = num_tokens * (num_cols / 32),
        .weight_expert_stride = expert_stride,
        .output_expert_stride = num_rows * num_tokens,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_mf4w_moe_matmul_fn,
        threadgroup_size, 1, 1,
        num_rows / num_simdgroups, num_tokens, num_active_experts,
        sizeof(args), &args,
        6,
        (const struct gptoss_metal_buffer *[]) {input_buffer, expert_buffer, weight_block_buffer, weight_scale_buffer, bias_buffer, output_buffer},
        (const size_t[]) {input_offset, expert_offset, weight_block_offset, weight_scale_offset, bias_offset, output_offset},
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_rope(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_rope_fn,
    size_t threadgroup_size,
    const struct gptoss_metal_buffer* activations_buffer,
    float rope_base,
    float interpolation_scale,
    float yarn_offset,
    float yarn_scale,
    float yarn_multiplier,
    uint32_t num_tokens,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t attn_head_dim,
    uint32_t token_offset)
{
    if (command_buffer->object == NULL || f32_rope_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = f32_rope_fn->max_threadgroup_threads;
    } else if (threadgroup_size > f32_rope_fn->max_threadgroup_threads) {
        return gptoss_status_invalid_argument;
    }

    const size_t num_simdgroups = threadgroup_size / f32_rope_fn->simdgroup_threads;
    const uint32_t num_qk_heads = num_q_heads + num_kv_heads;
    if (num_qk_heads % num_simdgroups != 0) {
        return gptoss_status_invalid_argument;
    }

    const struct gptoss_rope_args args = {
        .token_stride = (num_q_heads + 2 * num_kv_heads) * (attn_head_dim / 2),
        .token_offset = token_offset,
        .freq_scale = -logf(rope_base) / (float) (int32_t) (attn_head_dim / 2),
        .interpolation_scale = interpolation_scale,
        .yarn_offset = yarn_offset,
        .yarn_scale = yarn_scale,
        .yarn_multiplier = yarn_multiplier,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_rope_fn,
        threadgroup_size, 1, 1,
        num_qk_heads / num_simdgroups, num_tokens, 1,
        sizeof(args), &args,
        1, (const struct gptoss_metal_buffer *[]) {activations_buffer}, NULL,
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_accumulate(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_accumulate_fn,
    size_t threadgroup_size,
    size_t max_threadgroups,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* expert_buffer,
    size_t expert_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    uint32_t num_channels,
    uint32_t num_tokens,
    uint32_t num_experts)
{
    if (command_buffer->object == NULL || f32_accumulate_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (num_channels% 4 != 0) {
        return gptoss_status_invalid_argument;
    }

    if (threadgroup_size == 0) {
        threadgroup_size = f32_accumulate_fn->max_threadgroup_threads;
    } else if (threadgroup_size > f32_accumulate_fn->max_threadgroup_threads) {
        return gptoss_status_invalid_argument;
    }

    const size_t num_vecs = num_channels / 4;
    const size_t num_vecs_per_expert = num_vecs * num_tokens;
    const size_t num_vecs_per_threadgroup = math_ceil_div(num_vecs, max_threadgroups * threadgroup_size) * threadgroup_size;
    const size_t num_threadgroups = math_min(max_threadgroups, math_ceil_div(num_vecs, num_vecs_per_threadgroup));
    const struct gptoss_accumulate_args args = {
        .num_vecs_per_expert = num_vecs_per_expert,
        .num_vecs_per_threadgroup = num_vecs_per_threadgroup,
        .num_vecs = num_vecs,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_accumulate_fn,
        threadgroup_size, 1, 1,
        num_threadgroups, num_tokens, 1,
        sizeof(args), &args,
        3,
        (const struct gptoss_metal_buffer *[]) {input_buffer, expert_buffer, output_buffer},
        (const size_t[]) {input_offset, expert_offset, output_offset},
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_topk(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_topk_fn,
    const struct gptoss_metal_buffer* input_buffer,
    size_t input_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    uint32_t num_tokens,
    uint32_t num_experts,
    uint32_t num_active_experts)
{
    if (command_buffer->object == NULL || f32_topk_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (num_experts != 32  && num_experts != 128) {
        return gptoss_status_invalid_argument;
    }

    if (num_active_experts != 4) {
        return gptoss_status_invalid_argument;
    }

    const struct gptoss_topk_args args = { 0 };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_topk_fn,
        /*threadgroup_size=*/32, 1, 1,
        num_tokens, 1, 1,
        sizeof(args), &args,
        2,
        (const struct gptoss_metal_buffer *[]) {input_buffer, output_buffer},
        (const size_t[]) {input_offset, output_offset},
        /*threadgroup_buffer_size=*/0);
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_sdpa(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_sdpa_fn,
    const struct gptoss_metal_buffer* q_buffer,
    size_t q_offset,
    const struct gptoss_metal_buffer* k_buffer,
    size_t k_offset,
    const struct gptoss_metal_buffer* v_buffer,
    size_t v_offset,
    const struct gptoss_metal_buffer* s_buffer,
    size_t s_offset,
    const struct gptoss_metal_buffer* output_buffer,
    size_t output_offset,
    uint32_t window,
    uint32_t num_q_tokens,
    uint32_t num_kv_tokens,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim)
{
    if (command_buffer->object == NULL || f32_sdpa_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    if (num_q_heads != num_kv_heads * 8) {
        GPTOSS_LOG_ERROR("number of Q heads (%" PRIu32 ") must be 8 times the number of KV heads (%" PRIu32 ")",
            num_q_heads, num_kv_heads);
        return gptoss_status_invalid_argument;
    }

    if (head_dim != 64) {
        GPTOSS_LOG_ERROR("attention head dimension (%" PRIu32 ") must be 64", head_dim);
        return gptoss_status_invalid_argument;
    }

    const size_t max_context_tokens = math_min(num_q_tokens + num_kv_tokens + 1, window);
    const size_t threadgroup_size = math_min(f32_sdpa_fn->max_threadgroup_threads,
        max_context_tokens * f32_sdpa_fn->simdgroup_threads);
    const size_t half_threadgroup_size = math_round_down_po2(threadgroup_size / 2, f32_sdpa_fn->simdgroup_threads);

    const struct gptoss_sdpa_args args = {
        .qkv_dim = head_dim * (num_q_heads + 2 * num_kv_heads),
        .num_kv_tokens = num_kv_tokens,
        .window = window,
    };

    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_sdpa_fn,
        threadgroup_size, 1, 1,
        num_q_tokens, num_kv_heads, 1,
        sizeof(args), &args,
        5,
        (const struct gptoss_metal_buffer *[]) {q_buffer, k_buffer, v_buffer, s_buffer, output_buffer},
        (const size_t[]) {q_offset, k_offset, v_offset, s_offset, output_offset},
        /*threadgroup_buffer_size=*/half_threadgroup_size * 8 * 4 * sizeof(float));
}

enum gptoss_status gptoss_metal_command_buffer_encode_launch_f32_softmax(
    const struct gptoss_metal_command_buffer* command_buffer,
    const struct gptoss_metal_function* f32_softmax_fn,
    size_t threadgroup_size,
    size_t max_threadgroups,
    const struct gptoss_metal_buffer* score_buffer,
    size_t score_offset,
    const struct gptoss_metal_buffer* argmax_buffer,
    size_t argmax_offset,
    const struct gptoss_metal_buffer* prob_buffer,
    size_t prob_offset,
    const struct gptoss_metal_buffer* sum_buffer,
    size_t sum_offset,
    uint32_t num_channels,
    uint32_t num_tokens,
    float temperature,
    uint32_t* num_threadgroups_out,
    uint32_t* num_channels_per_threadgroup_out)
{
    *num_threadgroups_out = 0;
    *num_channels_per_threadgroup_out = 0;
    if (command_buffer->object == NULL || f32_softmax_fn->pipeline_state_object == NULL) {
        return gptoss_status_invalid_state;
    }

    const size_t num_vecs = num_channels;
    const size_t num_vecs_per_threadgroup = math_ceil_div(num_vecs, max_threadgroups * threadgroup_size) * threadgroup_size;
    const size_t num_threadgroups = math_min(max_threadgroups, math_ceil_div(num_vecs, num_vecs_per_threadgroup));
    const struct gptoss_softmax_args args = {
        .num_vecs = num_vecs,
        .num_vecs_per_threadgroup = num_vecs_per_threadgroup,
        .max_threadgroups = max_threadgroups,
        .temperature = temperature,
    };

    *num_threadgroups_out = num_threadgroups;
    *num_channels_per_threadgroup_out = num_vecs_per_threadgroup;
    return gptoss_metal_command_buffer_encode_launch_kernel(
        command_buffer, f32_softmax_fn,
        threadgroup_size, 1, 1,
        num_threadgroups, num_tokens, 1,
        sizeof(args), &args,
        4,
        (const struct gptoss_metal_buffer *[]) {score_buffer, argmax_buffer, prob_buffer, sum_buffer},
        (const size_t[]) {score_offset, argmax_offset, prob_offset, sum_offset},
        /*threadgroup_buffer_size=*/0);
}
